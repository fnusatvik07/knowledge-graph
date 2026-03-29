"""Enricher Agent -- analyzes the KG for coverage gaps, generates questions
about missing information, uses Tavily to search for answers, extracts new
facts, and adds them to the KG.

Uses LangGraph StateGraph for the enrichment pipeline.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import TypedDict

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
from shared.llm_clients import chat_completion, chat_completion_structured
from shared.config import get_neo4j_config, get_tavily_api_key


# ── Schemas ──────────────────────────────────────────────────────────

class GapInfo(BaseModel):
    entity_name: str = Field(description="Entity with a gap")
    gap_type: str = Field(description="One of: orphan, sparse, missing_type, disconnected")
    description: str = Field(description="What information is missing")
    search_query: str = Field(description="Web search query to fill this gap")


class GapAnalysis(BaseModel):
    gaps: list[GapInfo] = Field(default_factory=list)


class ExtractedFact(BaseModel):
    subject: str = Field(description="Source entity name")
    predicate: str = Field(description="Relationship type in UPPER_SNAKE_CASE")
    object: str = Field(description="Target entity name")
    object_type: str = Field(description="Entity type of the target")
    confidence: float = Field(description="Confidence 0-1")


class FactExtraction(BaseModel):
    facts: list[ExtractedFact] = Field(default_factory=list)


# ── LangGraph state ──────────────────────────────────────────────────

class EnricherState(TypedDict):
    gaps: list[dict]
    search_results: list[dict]
    new_facts_added: int
    report: str


# ── Helper ───────────────────────────────────────────────────────────

def _get_driver():
    config = get_neo4j_config()
    return GraphDatabase.driver(config["uri"], auth=(config["user"], config["password"]))


# ── Graph nodes ──────────────────────────────────────────────────────

def detect_gaps(state: EnricherState) -> dict:
    """Analyze the KG for coverage gaps."""
    driver = _get_driver()
    gaps = []

    with driver.session() as session:
        # Find orphan nodes (no relationships)
        orphans = session.run(
            """
            MATCH (e:Entity)
            WHERE NOT (e)--()
            RETURN e.name AS name, e.entity_type AS entity_type
            """
        )
        for record in orphans:
            gaps.append({
                "entity_name": record["name"],
                "gap_type": "orphan",
                "description": f"Entity '{record['name']}' ({record['entity_type']}) has no relationships",
                "search_query": f"{record['name']} {record['entity_type']} connections relationships",
            })

        # Find sparse entities (only 1 relationship)
        sparse = session.run(
            """
            MATCH (e:Entity)
            WITH e, size([(e)--() | 1]) AS degree
            WHERE degree = 1
            RETURN e.name AS name, e.entity_type AS entity_type, degree
            """
        )
        for record in sparse:
            gaps.append({
                "entity_name": record["name"],
                "gap_type": "sparse",
                "description": f"Entity '{record['name']}' has only {record['degree']} connection(s)",
                "search_query": f"{record['name']} related technologies applications",
            })

        # Find entity types with very few instances
        type_counts = session.run(
            """
            MATCH (e:Entity)
            WITH e.entity_type AS entity_type, count(*) AS cnt
            WHERE cnt < 2
            RETURN entity_type, cnt
            """
        )
        for record in type_counts:
            gaps.append({
                "entity_name": record["entity_type"],
                "gap_type": "missing_type",
                "description": f"Entity type '{record['entity_type']}' has only {record['cnt']} instance(s)",
                "search_query": f"{record['entity_type']} examples in AI and technology",
            })

    driver.close()

    # Use LLM to prioritize and generate better search queries
    if gaps:
        gap_descriptions = "\n".join(f"- {g['description']}" for g in gaps[:10])
        analysis = chat_completion(
            prompt=f"""Given these knowledge graph gaps, generate targeted search queries:

{gap_descriptions}

For each gap, suggest a specific web search query that would help fill it.
Format: QUERY: [query text]""",
            system="You are a knowledge graph enrichment specialist.",
        )
        # Update search queries with LLM suggestions
        query_lines = [l.strip() for l in analysis.split("\n") if l.strip().startswith("QUERY:")]
        for i, query_line in enumerate(query_lines):
            if i < len(gaps):
                gaps[i]["search_query"] = query_line.replace("QUERY:", "").strip()

    return {"gaps": gaps}


def search_for_information(state: EnricherState) -> dict:
    """Use Tavily to search for information to fill gaps."""
    from tavily import TavilyClient

    tavily = TavilyClient(api_key=get_tavily_api_key())
    search_results = []

    # Search for top gaps (limit to avoid too many API calls)
    for gap in state["gaps"][:5]:
        try:
            result = tavily.search(
                query=gap["search_query"],
                max_results=3,
                search_depth="basic",
            )
            search_results.append({
                "gap": gap,
                "results": [
                    {"title": r.get("title", ""), "content": r.get("content", "")}
                    for r in result.get("results", [])
                ],
            })
        except Exception as e:
            search_results.append({
                "gap": gap,
                "results": [],
                "error": str(e),
            })

    return {"search_results": search_results}


def extract_and_write_facts(state: EnricherState) -> dict:
    """Extract facts from search results and write to Neo4j."""
    driver = _get_driver()
    new_facts_added = 0
    now = datetime.now(timezone.utc).isoformat()

    for search_item in state["search_results"]:
        if not search_item.get("results"):
            continue

        # Combine search results into context
        context = "\n\n".join(
            f"Title: {r['title']}\nContent: {r['content']}"
            for r in search_item["results"]
        )

        gap = search_item["gap"]

        # Use LLM to extract structured facts
        try:
            extraction = chat_completion_structured(
                prompt=f"""From the following search results about '{gap['entity_name']}', extract
factual relationships that could be added to a knowledge graph.

Search context:
{context}

Focus on concrete, verifiable facts. Extract entities and relationships.""",
                output_schema=FactExtraction,
                system="You are a knowledge graph fact extractor. Only extract facts with high confidence.",
            )
        except Exception:
            continue

        with driver.session() as session:
            for fact in extraction.facts:
                if fact.confidence < 0.6:
                    continue

                # Create or merge entities and relationships
                session.run(
                    """
                    MERGE (s:Entity {name: $subject})
                    ON CREATE SET s.entity_type = 'Unknown', s.created_at = $now
                    MERGE (o:Entity {name: $object})
                    ON CREATE SET o.entity_type = $object_type, o.created_at = $now
                    """,
                    subject=fact.subject,
                    object=fact.object,
                    object_type=fact.object_type,
                    now=now,
                )
                session.run(
                    f"""
                    MATCH (s:Entity {{name: $subject}})
                    MATCH (o:Entity {{name: $object}})
                    MERGE (s)-[r:{fact.predicate}]->(o)
                    SET r.confidence = $confidence,
                        r.source = 'web_search',
                        r.extracted_by = 'enricher_agent',
                        r.created_at = $now
                    """,
                    subject=fact.subject,
                    object=fact.object,
                    confidence=fact.confidence,
                    now=now,
                )
                new_facts_added += 1

    driver.close()
    return {"new_facts_added": new_facts_added}


def generate_enrichment_report(state: EnricherState) -> dict:
    """Generate a report of the enrichment process."""
    gaps = state["gaps"]
    searches = state["search_results"]
    new_facts = state["new_facts_added"]

    report_lines = [
        "=== Enrichment Report ===",
        f"Gaps detected: {len(gaps)}",
        f"Searches performed: {len(searches)}",
        f"New facts added: {new_facts}",
        "",
        "Gaps analyzed:",
    ]
    for gap in gaps[:10]:
        report_lines.append(f"  [{gap['gap_type']}] {gap['description']}")

    report = "\n".join(report_lines)
    return {"report": report}


# ── Build the graph ──────────────────────────────────────────────────

def build_enricher_graph() -> StateGraph:
    """Build and compile the Enricher Agent graph."""
    graph = StateGraph(EnricherState)

    graph.add_node("detect_gaps", detect_gaps)
    graph.add_node("search", search_for_information)
    graph.add_node("extract_and_write", extract_and_write_facts)
    graph.add_node("report", generate_enrichment_report)

    graph.set_entry_point("detect_gaps")
    graph.add_edge("detect_gaps", "search")
    graph.add_edge("search", "extract_and_write")
    graph.add_edge("extract_and_write", "report")
    graph.add_edge("report", END)

    return graph.compile()


def run_enricher() -> str:
    """Run the enricher and return the report."""
    app = build_enricher_graph()
    result = app.invoke({
        "gaps": [],
        "search_results": [],
        "new_facts_added": 0,
        "report": "",
    })
    return result["report"]


if __name__ == "__main__":
    report = run_enricher()
    print(report)
