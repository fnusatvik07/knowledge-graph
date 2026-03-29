"""Web Search Agent: enriches the knowledge graph with live web data.

Uses Tavily search to find additional context about entities in the
knowledge graph and returns structured results for KG enrichment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from langgraph.graph import StateGraph, END

from shared.config import get_tavily_api_key
from shared.llm_clients import chat_completion
from src.graph.neo4j_store import Neo4jStore
from src.graph.schema import (
    ExtractionResult,
    ExtractedEntity,
    ExtractedRelationship,
    get_entity_types_list,
    get_relationship_types_list,
)
from src.prompts.extraction import (
    ENTITY_EXTRACTION_SYSTEM,
    ENTITY_EXTRACTION_USER,
)


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class WebSearchState(TypedDict):
    """State for the web search agent."""
    entity_names: list[str]              # Entities to search for
    search_results: list[dict[str, Any]]  # Raw search results
    extractions: list[ExtractionResult]   # Extracted entities/relationships
    status: str


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def search_entities(state: WebSearchState) -> dict[str, Any]:
    """Search the web for information about specified entities."""
    from tavily import TavilyClient

    client = TavilyClient(api_key=get_tavily_api_key())
    all_results: list[dict[str, Any]] = []

    for entity_name in state["entity_names"]:
        try:
            response = client.search(
                query=f"{entity_name} AI technology latest information",
                max_results=3,
                search_depth="basic",
            )
            for result in response.get("results", []):
                all_results.append({
                    "entity": entity_name,
                    "title": result.get("title", ""),
                    "content": result.get("content", ""),
                    "url": result.get("url", ""),
                })
        except Exception as e:
            all_results.append({
                "entity": entity_name,
                "error": str(e),
            })

    return {"search_results": all_results, "status": "searched"}


def extract_from_web_results(state: WebSearchState) -> dict[str, Any]:
    """Extract entities and relationships from web search results."""
    extractions: list[ExtractionResult] = []
    entity_types = ", ".join(get_entity_types_list())
    relationship_types = ", ".join(get_relationship_types_list())

    # Group results by entity and combine content
    grouped: dict[str, list[str]] = {}
    for result in state["search_results"]:
        if "error" in result:
            continue
        entity = result["entity"]
        grouped.setdefault(entity, []).append(result.get("content", ""))

    for entity, contents in grouped.items():
        combined = "\n\n".join(contents)[:3000]  # Limit to 3000 chars
        if not combined.strip():
            continue

        try:
            prompt = ENTITY_EXTRACTION_USER.format(
                text=combined,
                entity_types=entity_types,
                relationship_types=relationship_types,
            )
            response = chat_completion(
                prompt=prompt,
                system=ENTITY_EXTRACTION_SYSTEM,
                model="gpt-4o-mini",
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            data = json.loads(response)
            extraction = ExtractionResult(
                entities=[ExtractedEntity(**e) for e in data.get("entities", [])],
                relationships=[ExtractedRelationship(**r) for r in data.get("relationships", [])],
            )
            extractions.append(extraction)
        except Exception:
            pass

    return {"extractions": extractions, "status": "extracted"}


# ---------------------------------------------------------------------------
# Workflow builder
# ---------------------------------------------------------------------------

def build_web_search_workflow() -> StateGraph:
    """Build the LangGraph workflow for web search enrichment."""
    workflow = StateGraph(WebSearchState)
    workflow.add_node("search", search_entities)
    workflow.add_node("extract", extract_from_web_results)

    workflow.set_entry_point("search")
    workflow.add_edge("search", "extract")
    workflow.add_edge("extract", END)

    return workflow


class WebSearchAgent:
    """High-level interface for web search KG enrichment."""

    def __init__(self) -> None:
        self.workflow = build_web_search_workflow()
        self.app = self.workflow.compile()

    def search_and_extract(
        self, entity_names: list[str]
    ) -> dict[str, Any]:
        """Search for entities and extract structured information.

        Args:
            entity_names: List of entity names to search for.

        Returns:
            Dict with 'extractions' and 'search_results'.
        """
        initial_state: WebSearchState = {
            "entity_names": entity_names,
            "search_results": [],
            "extractions": [],
            "status": "started",
        }
        result = self.app.invoke(initial_state)
        return {
            "extractions": result.get("extractions", []),
            "search_results": result.get("search_results", []),
            "status": result.get("status", "unknown"),
        }
