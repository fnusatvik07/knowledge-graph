"""Autonomous KG Agent -- builds, maintains, and improves a knowledge graph
with minimal human intervention.

Uses a LangGraph StateGraph with an autonomous loop:
assess -> detect -> plan -> execute -> evaluate -> should_continue -> (loop or done)
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import TypedDict, Literal

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
from shared.llm_clients import chat_completion, chat_completion_structured
from shared.config import get_neo4j_config, get_tavily_api_key

from src.tools.kg_analyzer import (
    get_kg_stats,
    get_coverage_score,
    get_quality_score,
    find_orphan_entities,
    find_duplicate_candidates,
)
from src.tools.source_manager import (
    list_pending_sources,
    evaluate_source,
    mark_processed,
    get_processing_history,
)
from src.planning.gap_detector import detect_coverage_gaps
from src.planning.action_planner import plan_next_action
from src.agents.relevance_scorer import score_relevance
from src.agents.conflict_resolver import resolve_conflicts


# ── Pydantic schemas ─────────────────────────────────────────────────

class ExtractedEntity(BaseModel):
    name: str
    entity_type: str
    description: str


class ExtractedRelationship(BaseModel):
    source: str
    target: str
    relationship_type: str
    description: str


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)


# ── LangGraph state ──────────────────────────────────────────────────

class AutonomousState(TypedDict):
    kg_stats: dict
    coverage_gaps: list[dict]
    pending_sources: list[dict]
    action_history: list[dict]
    quality_score: float
    iteration: int
    max_iterations: int
    quality_threshold: float
    current_action: dict | None
    action_result: str
    previous_quality: float
    should_stop: bool


# ── Helper ───────────────────────────────────────────────────────────

def _get_driver():
    config = get_neo4j_config()
    return GraphDatabase.driver(config["uri"], auth=(config["user"], config["password"]))


# ── Graph nodes ──────────────────────────────────────────────────────

def assess_state(state: AutonomousState) -> dict:
    """Analyze the current KG: size, coverage, quality."""
    stats = get_kg_stats()
    quality = get_quality_score()
    coverage = get_coverage_score()

    stats["coverage_score"] = coverage
    stats["quality_score"] = quality

    return {
        "kg_stats": stats,
        "quality_score": quality,
        "previous_quality": state.get("quality_score", 0.0),
    }


def detect_gaps(state: AutonomousState) -> dict:
    """Find coverage gaps in the KG."""
    gaps = detect_coverage_gaps()
    return {"coverage_gaps": gaps}


def plan_actions(state: AutonomousState) -> dict:
    """Decide what to do next based on gaps and available sources."""
    pending = list_pending_sources()
    action = plan_next_action(
        gaps=state["coverage_gaps"],
        kg_stats=state["kg_stats"],
        pending_sources=pending,
        action_history=state["action_history"],
    )
    return {
        "current_action": action,
        "pending_sources": pending,
    }


def execute_action(state: AutonomousState) -> dict:
    """Carry out the planned action."""
    action = state["current_action"]
    if not action:
        return {"action_result": "No action planned."}

    action_type = action.get("action_type", "none")
    now = datetime.now(timezone.utc).isoformat()
    result = ""

    if action_type == "ingest_source":
        # Evaluate and ingest a candidate source
        source = action.get("source", {})
        relevance = score_relevance(source, state["kg_stats"])

        if relevance["relevance_score"] < 0.5:
            result = f"Skipped source '{source.get('title', '?')}': low relevance ({relevance['relevance_score']:.2f}). Reason: {relevance['reason']}"
            mark_processed(source.get("id", ""), status="skipped")
        else:
            # Extract entities and relationships
            extraction = _extract_from_text(source.get("text", ""), source.get("title", ""))
            written = _write_extraction_to_neo4j(extraction, source.get("id", ""))
            result = (
                f"Ingested '{source.get('title', '?')}' (relevance: {relevance['relevance_score']:.2f}). "
                f"Added {written['entities']} entities, {written['relationships']} relationships."
            )
            mark_processed(source.get("id", ""), status="ingested")

            # Check for conflicts
            conflicts = resolve_conflicts(extraction)
            if conflicts:
                result += f" Resolved {len(conflicts)} conflict(s)."

    elif action_type == "enrich_entity":
        # Search the web for more information about a sparse entity
        entity_name = action.get("entity_name", "")
        result = _enrich_entity_from_web(entity_name)

    elif action_type == "validate_and_dedup":
        # Run deduplication
        duplicates = find_duplicate_candidates()
        merged = _merge_duplicates(duplicates)
        result = f"Validation: found {len(duplicates)} duplicate candidates, merged {merged}."

    elif action_type == "web_search":
        # Search the web for a specific topic
        query = action.get("query", "")
        result = _web_search_and_extract(query)

    else:
        result = f"Unknown action type: {action_type}"

    # Record in action history
    history = list(state["action_history"])
    history.append({
        "iteration": state["iteration"],
        "action_type": action_type,
        "result": result,
        "timestamp": now,
    })

    return {
        "action_result": result,
        "action_history": history,
    }


def evaluate_quality(state: AutonomousState) -> dict:
    """Check if the action improved the KG."""
    new_quality = get_quality_score()
    prev_quality = state.get("previous_quality", 0.0)

    improvement = new_quality - prev_quality
    stats = get_kg_stats()

    # Update action history with quality delta
    history = list(state["action_history"])
    if history:
        history[-1]["quality_before"] = prev_quality
        history[-1]["quality_after"] = new_quality
        history[-1]["quality_delta"] = improvement

    return {
        "quality_score": new_quality,
        "kg_stats": stats,
        "action_history": history,
    }


def should_continue(state: AutonomousState) -> str:
    """Decide if more work is needed."""
    iteration = state.get("iteration", 0) + 1
    max_iter = state.get("max_iterations", 5)
    quality = state.get("quality_score", 0.0)
    threshold = state.get("quality_threshold", 0.8)

    if iteration >= max_iter:
        return "done"
    if quality >= threshold:
        return "done"

    return "continue"


def increment_iteration(state: AutonomousState) -> dict:
    """Increment the iteration counter."""
    return {"iteration": state.get("iteration", 0) + 1}


# ── Internal helpers ─────────────────────────────────────────────────

def _extract_from_text(text: str, title: str) -> ExtractionResult:
    """Extract entities and relationships from text using LLM."""
    return chat_completion_structured(
        prompt=f"""Extract entities and relationships from the following text.

Title: {title}
Text: {text}

Extract all important entities (people, organizations, technologies, concepts)
and their relationships. Use UPPER_SNAKE_CASE for relationship types.""",
        output_schema=ExtractionResult,
        system="You are a knowledge graph extraction expert.",
    )


def _write_extraction_to_neo4j(extraction: ExtractionResult, source_id: str) -> dict:
    """Write extracted entities/relationships to Neo4j."""
    driver = _get_driver()
    now = datetime.now(timezone.utc).isoformat()
    entities_written = 0
    rels_written = 0

    with driver.session() as session:
        for entity in extraction.entities:
            session.run(
                """
                MERGE (e:Entity {name: $name})
                SET e.entity_type = $entity_type,
                    e.description = $description,
                    e.source = $source,
                    e.extracted_by = 'autonomous_agent',
                    e.updated_at = $now
                """,
                name=entity.name,
                entity_type=entity.entity_type,
                description=entity.description,
                source=source_id,
                now=now,
            )
            entities_written += 1

        for rel in extraction.relationships:
            session.run(
                f"""
                MATCH (s:Entity {{name: $source}})
                MATCH (t:Entity {{name: $target}})
                MERGE (s)-[r:{rel.relationship_type}]->(t)
                SET r.description = $description,
                    r.source = $source_id,
                    r.extracted_by = 'autonomous_agent',
                    r.created_at = $now
                """,
                source=rel.source,
                target=rel.target,
                description=rel.description,
                source_id=source_id,
                now=now,
            )
            rels_written += 1

    driver.close()
    return {"entities": entities_written, "relationships": rels_written}


def _enrich_entity_from_web(entity_name: str) -> str:
    """Search the web for information about an entity and add to KG."""
    from tavily import TavilyClient

    try:
        client = TavilyClient(api_key=get_tavily_api_key())
        result = client.search(query=f"{entity_name} artificial intelligence", max_results=3)

        context = "\n".join(
            r.get("content", "") for r in result.get("results", [])
        )
        if not context.strip():
            return f"No web results found for '{entity_name}'."

        extraction = _extract_from_text(context, f"Web search: {entity_name}")
        written = _write_extraction_to_neo4j(extraction, f"web_search_{entity_name}")
        return (
            f"Enriched '{entity_name}' from web: "
            f"added {written['entities']} entities, {written['relationships']} relationships."
        )
    except Exception as e:
        return f"Web enrichment error for '{entity_name}': {e}"


def _web_search_and_extract(query: str) -> str:
    """Generic web search and extraction."""
    from tavily import TavilyClient

    try:
        client = TavilyClient(api_key=get_tavily_api_key())
        result = client.search(query=query, max_results=3)

        context = "\n".join(r.get("content", "") for r in result.get("results", []))
        if not context.strip():
            return f"No results for query: {query}"

        extraction = _extract_from_text(context, f"Web: {query}")
        written = _write_extraction_to_neo4j(extraction, f"web_{query[:30]}")
        return (
            f"Web search '{query}': added {written['entities']} entities, "
            f"{written['relationships']} relationships."
        )
    except Exception as e:
        return f"Web search error: {e}"


def _merge_duplicates(duplicates: list[dict]) -> int:
    """Merge duplicate entities."""
    driver = _get_driver()
    merged = 0

    with driver.session() as session:
        for dup in duplicates[:10]:  # Limit to avoid excessive merging
            try:
                session.run(
                    """
                    MATCH (a:Entity {name: $keep}), (b:Entity {name: $merge})
                    WHERE a <> b
                    OPTIONAL MATCH (b)-[r]->()
                    WITH a, b, collect(r) AS rels
                    DETACH DELETE b
                    """,
                    keep=dup.get("canonical", dup.get("name_a", "")),
                    merge=dup.get("duplicate", dup.get("name_b", "")),
                )
                merged += 1
            except Exception:
                continue

    driver.close()
    return merged


# ── Build the graph ──────────────────────────────────────────────────

def build_autonomous_graph() -> StateGraph:
    """Build and compile the Autonomous Agent graph."""
    graph = StateGraph(AutonomousState)

    graph.add_node("assess", assess_state)
    graph.add_node("detect", detect_gaps)
    graph.add_node("plan", plan_actions)
    graph.add_node("execute", execute_action)
    graph.add_node("evaluate", evaluate_quality)
    graph.add_node("increment", increment_iteration)

    graph.set_entry_point("assess")
    graph.add_edge("assess", "detect")
    graph.add_edge("detect", "plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "evaluate")
    graph.add_edge("evaluate", "increment")

    graph.add_conditional_edges("increment", should_continue, {
        "continue": "assess",
        "done": END,
    })

    return graph.compile()


def run_autonomous(
    max_iterations: int = 5,
    quality_threshold: float = 0.8,
) -> AutonomousState:
    """Run the autonomous agent.

    Args:
        max_iterations: Maximum number of improvement iterations
        quality_threshold: Quality score (0-1) at which the agent stops

    Returns:
        Final state dict with all history and metrics
    """
    app = build_autonomous_graph()
    result = app.invoke({
        "kg_stats": {},
        "coverage_gaps": [],
        "pending_sources": [],
        "action_history": [],
        "quality_score": 0.0,
        "iteration": 0,
        "max_iterations": max_iterations,
        "quality_threshold": quality_threshold,
        "current_action": None,
        "action_result": "",
        "previous_quality": 0.0,
        "should_stop": False,
    })
    return result


if __name__ == "__main__":
    final_state = run_autonomous(max_iterations=3)
    print(f"\nFinal quality score: {final_state['quality_score']:.2f}")
    print(f"Iterations completed: {final_state['iteration']}")
    print(f"Actions taken: {len(final_state['action_history'])}")
    for action in final_state["action_history"]:
        print(f"  [{action['iteration']}] {action['action_type']}: {action['result'][:100]}")
