"""Graph-based retrieval via Cypher traversal.

Given seed entities, traverses the knowledge graph N hops outward
and returns structured subgraph context.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.graph.neo4j_store import Neo4jStore


@dataclass
class SubgraphNode:
    """A node in a retrieved subgraph."""
    name: str
    entity_type: str
    description: str = ""
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubgraphEdge:
    """An edge in a retrieved subgraph."""
    source: str
    target: str
    relationship_type: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubgraphContext:
    """A subgraph context retrieved from the knowledge graph."""
    seed_entities: list[str]
    nodes: list[SubgraphNode]
    edges: list[SubgraphEdge]
    max_hops: int

    def to_text(self) -> str:
        """Format the subgraph as a readable text block for LLM consumption."""
        lines = [f"Subgraph context (seeds: {', '.join(self.seed_entities)}, hops: {self.max_hops})"]
        lines.append("")
        lines.append("Entities:")
        for node in self.nodes:
            desc = f" - {node.description}" if node.description else ""
            lines.append(f"  [{node.entity_type}] {node.name}{desc}")
        lines.append("")
        lines.append("Relationships:")
        for edge in self.edges:
            lines.append(f"  {edge.source} --[{edge.relationship_type}]--> {edge.target}")
        return "\n".join(lines)


class GraphRetriever:
    """Retrieves subgraph context from Neo4j via Cypher traversal."""

    def __init__(
        self,
        store: Neo4jStore,
        default_max_hops: int = 2,
    ) -> None:
        self.store = store
        self.default_max_hops = default_max_hops

    def retrieve(
        self,
        seed_entities: list[str],
        max_hops: int | None = None,
        rel_types: list[str] | None = None,
    ) -> SubgraphContext:
        """Traverse the graph from seed entities and return a SubgraphContext.

        Args:
            seed_entities: List of entity names to start traversal from.
            max_hops: Maximum traversal depth (default: 2).
            rel_types: Optional filter on relationship types.

        Returns:
            A SubgraphContext with unique nodes and edges.
        """
        hops = max_hops or self.default_max_hops
        rel_filter = ":" + "|".join(rel_types) if rel_types else ""

        nodes_map: dict[str, SubgraphNode] = {}
        edges_set: set[tuple[str, str, str]] = set()
        edges_list: list[SubgraphEdge] = []

        for seed in seed_entities:
            # Fetch the seed node itself
            seed_query = "MATCH (n {name: $name}) RETURN n LIMIT 1"
            seed_results = self.store.run_cypher(seed_query, {"name": seed})
            for record in seed_results:
                node = record["n"]
                name = node.get("name", seed)
                if name not in nodes_map:
                    props = {k: v for k, v in dict(node).items() if k != "embedding"}
                    nodes_map[name] = SubgraphNode(
                        name=name,
                        entity_type=node.get("entity_type", ""),
                        description=node.get("description", ""),
                        properties=props,
                    )

            # Traverse outward
            traverse_query = (
                f"MATCH path = (start {{name: $name}})"
                f"-[r{rel_filter}*1..{hops}]-(end) "
                "UNWIND relationships(path) AS rel "
                "WITH startNode(rel) AS s, endNode(rel) AS t, rel "
                "RETURN s, t, type(rel) AS rel_type, properties(rel) AS rel_props"
            )
            results = self.store.run_cypher(traverse_query, {"name": seed})

            for record in results:
                s = record["s"]
                t = record["t"]
                s_name = s.get("name", "")
                t_name = t.get("name", "")

                # Collect nodes
                if s_name and s_name not in nodes_map:
                    s_props = {k: v for k, v in dict(s).items() if k != "embedding"}
                    nodes_map[s_name] = SubgraphNode(
                        name=s_name,
                        entity_type=s.get("entity_type", ""),
                        description=s.get("description", ""),
                        properties=s_props,
                    )
                if t_name and t_name not in nodes_map:
                    t_props = {k: v for k, v in dict(t).items() if k != "embedding"}
                    nodes_map[t_name] = SubgraphNode(
                        name=t_name,
                        entity_type=t.get("entity_type", ""),
                        description=t.get("description", ""),
                        properties=t_props,
                    )

                # Collect edges (deduplicated)
                rel_type = record["rel_type"]
                edge_key = (s_name, t_name, rel_type)
                if edge_key not in edges_set:
                    edges_set.add(edge_key)
                    rel_props = record.get("rel_props", {}) or {}
                    edges_list.append(SubgraphEdge(
                        source=s_name,
                        target=t_name,
                        relationship_type=rel_type,
                        properties=rel_props,
                    ))

        return SubgraphContext(
            seed_entities=seed_entities,
            nodes=list(nodes_map.values()),
            edges=edges_list,
            max_hops=hops,
        )

    def find_paths(
        self,
        source_name: str,
        target_name: str,
        max_hops: int = 4,
    ) -> list[list[dict[str, str]]]:
        """Find all paths between two entities up to max_hops.

        Returns a list of paths, where each path is a list of steps
        (dicts with source, relationship, target).
        """
        query = (
            f"MATCH path = shortestPath("
            f"  (a {{name: $source}})-[*..{max_hops}]-(b {{name: $target}})"
            ") "
            "UNWIND relationships(path) AS rel "
            "RETURN startNode(rel).name AS from_node, "
            "       type(rel) AS rel_type, "
            "       endNode(rel).name AS to_node "
            "ORDER BY elementId(rel)"
        )
        results = self.store.run_cypher(
            query, {"source": source_name, "target": target_name}
        )
        if not results:
            return []

        path = [
            {
                "source": r["from_node"],
                "relationship": r["rel_type"],
                "target": r["to_node"],
            }
            for r in results
        ]
        return [path]
