"""LangChain-compatible tool for Cypher queries against Neo4j.

Wraps the Neo4jStore to provide a tool interface that LangGraph agents
can invoke for knowledge graph queries.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from langchain_core.tools import BaseTool
from pydantic import Field

from src.graph.neo4j_store import Neo4jStore


class KGQueryTool(BaseTool):
    """Tool for running Cypher queries against the Neo4j knowledge graph."""

    name: str = "kg_query"
    description: str = (
        "Run a Cypher query against the knowledge graph database. "
        "Use this to look up entities, find relationships, traverse paths, "
        "or answer questions about the knowledge graph. "
        "Input should be a valid Cypher query string."
    )
    store: Any = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, store: Neo4jStore, **kwargs: Any) -> None:
        super().__init__(store=store, **kwargs)

    def _run(self, query: str) -> str:
        """Execute a Cypher query and return JSON results."""
        try:
            results = self.store.run_cypher(query)
            # Convert Neo4j Node/Relationship objects to plain dicts
            serializable = _make_serializable(results)
            return json.dumps(serializable, indent=2, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def _arun(self, query: str) -> str:
        """Async not implemented; falls back to sync."""
        return self._run(query)


class KGEntityLookupTool(BaseTool):
    """Tool for looking up a specific entity by name."""

    name: str = "kg_entity_lookup"
    description: str = (
        "Look up a specific entity in the knowledge graph by name. "
        "Returns the entity properties and its immediate relationships. "
        "Input should be the entity name as a string."
    )
    store: Any = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, store: Neo4jStore, **kwargs: Any) -> None:
        super().__init__(store=store, **kwargs)

    def _run(self, entity_name: str) -> str:
        """Look up an entity and its neighbors."""
        try:
            # Find the entity
            query = (
                "MATCH (n {name: $name}) "
                "OPTIONAL MATCH (n)-[r]-(m) "
                "RETURN n, collect(DISTINCT {rel_type: type(r), neighbor: m.name, "
                "neighbor_type: m.entity_type}) AS relationships"
            )
            results = self.store.run_cypher(query, {"name": entity_name})
            if not results:
                return json.dumps({"error": f"Entity '{entity_name}' not found"})

            record = results[0]
            node_props = dict(record["n"])
            node_props.pop("embedding", None)

            return json.dumps({
                "entity": node_props,
                "relationships": record["relationships"],
            }, indent=2, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def _arun(self, entity_name: str) -> str:
        return self._run(entity_name)


class KGNeighborsTool(BaseTool):
    """Tool for finding neighbors of an entity within N hops."""

    name: str = "kg_neighbors"
    description: str = (
        "Find entities connected to a given entity within a specified "
        "number of hops. Input should be a JSON string with 'name' and "
        "optional 'max_hops' (default 2)."
    )
    store: Any = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, store: Neo4jStore, **kwargs: Any) -> None:
        super().__init__(store=store, **kwargs)

    def _run(self, input_str: str) -> str:
        """Find neighbors of an entity."""
        try:
            params = json.loads(input_str)
            name = params["name"]
            max_hops = params.get("max_hops", 2)
        except (json.JSONDecodeError, KeyError):
            # Treat input as plain entity name
            name = input_str.strip()
            max_hops = 2

        try:
            neighbors = self.store.get_neighbors(name=name, max_hops=max_hops)
            clean = []
            for n in neighbors:
                node = {k: v for k, v in n["node"].items() if k != "embedding"}
                clean.append({"node": node, "depth": n["depth"]})
            return json.dumps(clean, indent=2, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def _arun(self, input_str: str) -> str:
        return self._run(input_str)


def _make_serializable(obj: Any) -> Any:
    """Recursively convert Neo4j types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    if hasattr(obj, "items"):
        return {k: _make_serializable(v) for k, v in obj.items()}
    return obj
