"""Temporal metadata management for knowledge graph facts.

Provides functions for adding temporal validity windows to nodes and edges,
invalidating outdated facts, and running point-in-time queries. Inspired
by Graphiti's episodic approach but simplified for this project.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.graph.neo4j_store import Neo4jStore


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


class TemporalLayer:
    """Manages temporal metadata on a Neo4j-backed knowledge graph."""

    def __init__(self, store: Neo4jStore) -> None:
        self.store = store

    # ------------------------------------------------------------------
    # Adding temporal metadata
    # ------------------------------------------------------------------

    def add_node_temporal_metadata(
        self,
        label: str,
        name: str,
        *,
        valid_from: str | None = None,
        valid_to: str | None = None,
        source: str | None = None,
        confidence: float = 1.0,
    ) -> dict[str, Any]:
        """Attach temporal metadata to an existing node."""
        props: dict[str, Any] = {
            "valid_from": valid_from or _now_iso(),
            "confidence": confidence,
            "last_updated": _now_iso(),
        }
        if valid_to:
            props["valid_to"] = valid_to
        if source:
            props["source"] = source

        set_parts = ", ".join(f"n.{k} = ${k}" for k in props)
        query = (
            f"MATCH (n:{label} {{name: $name}}) "
            f"SET {set_parts} "
            "RETURN n"
        )
        results = self.store.run_cypher(query, {"name": name, **props})
        return dict(results[0]["n"]) if results else {}

    def add_relationship_temporal_metadata(
        self,
        source_name: str,
        target_name: str,
        rel_type: str,
        *,
        valid_from: str | None = None,
        valid_to: str | None = None,
        source: str | None = None,
        confidence: float = 1.0,
    ) -> dict[str, Any] | None:
        """Attach temporal metadata to an existing relationship."""
        props: dict[str, Any] = {
            "valid_from": valid_from or _now_iso(),
            "confidence": confidence,
            "last_updated": _now_iso(),
        }
        if valid_to:
            props["valid_to"] = valid_to
        if source:
            props["source"] = source

        set_parts = ", ".join(f"r.{k} = ${k}" for k in props)
        query = (
            f"MATCH (a {{name: $source_name}})-[r:{rel_type}]->(b {{name: $target_name}}) "
            f"SET {set_parts} "
            "RETURN type(r) AS rel_type, properties(r) AS props"
        )
        params = {"source_name": source_name, "target_name": target_name, **props}
        results = self.store.run_cypher(query, params)
        if results:
            return {"type": results[0]["rel_type"], "properties": results[0]["props"]}
        return None

    # ------------------------------------------------------------------
    # Invalidation
    # ------------------------------------------------------------------

    def invalidate_node(
        self,
        label: str,
        name: str,
        reason: str = "",
    ) -> None:
        """Mark a node as no longer valid by setting valid_to to now."""
        query = (
            f"MATCH (n:{label} {{name: $name}}) "
            "SET n.valid_to = $valid_to, n.invalidation_reason = $reason, "
            "    n.last_updated = $last_updated"
        )
        self.store.run_cypher(query, {
            "name": name,
            "valid_to": _now_iso(),
            "reason": reason,
            "last_updated": _now_iso(),
        })

    def invalidate_relationship(
        self,
        source_name: str,
        target_name: str,
        rel_type: str,
        reason: str = "",
    ) -> None:
        """Mark a relationship as no longer valid."""
        query = (
            f"MATCH (a {{name: $source_name}})-[r:{rel_type}]->(b {{name: $target_name}}) "
            "SET r.valid_to = $valid_to, r.invalidation_reason = $reason, "
            "    r.last_updated = $last_updated"
        )
        self.store.run_cypher(query, {
            "source_name": source_name,
            "target_name": target_name,
            "valid_to": _now_iso(),
            "reason": reason,
            "last_updated": _now_iso(),
        })

    # ------------------------------------------------------------------
    # Point-in-time queries
    # ------------------------------------------------------------------

    def get_valid_nodes_at(
        self,
        label: str,
        point_in_time: str,
    ) -> list[dict[str, Any]]:
        """Return all nodes of a label that were valid at a specific point in time.

        Args:
            label: The node label to filter on.
            point_in_time: ISO-8601 datetime string.
        """
        query = (
            f"MATCH (n:{label}) "
            "WHERE n.valid_from <= $pit "
            "  AND (n.valid_to IS NULL OR n.valid_to > $pit) "
            "RETURN n"
        )
        results = self.store.run_cypher(query, {"pit": point_in_time})
        return [dict(r["n"]) for r in results]

    def get_valid_relationships_at(
        self,
        rel_type: str,
        point_in_time: str,
    ) -> list[dict[str, Any]]:
        """Return all relationships of a type valid at a specific point in time."""
        query = (
            f"MATCH (a)-[r:{rel_type}]->(b) "
            "WHERE r.valid_from <= $pit "
            "  AND (r.valid_to IS NULL OR r.valid_to > $pit) "
            "RETURN a.name AS source, b.name AS target, "
            "       type(r) AS rel_type, properties(r) AS props"
        )
        results = self.store.run_cypher(query, {"pit": point_in_time})
        return [
            {
                "source": r["source"],
                "target": r["target"],
                "type": r["rel_type"],
                "properties": r["props"],
            }
            for r in results
        ]

    # ------------------------------------------------------------------
    # Current-state helpers
    # ------------------------------------------------------------------

    def get_current_entity_state(
        self,
        name: str,
        label: str | None = None,
    ) -> dict[str, Any] | None:
        """Get the currently valid state of an entity and its relationships.

        Returns a dict with 'node' properties and 'relationships' list.
        """
        label_filter = f":{label}" if label else ""
        node_query = (
            f"MATCH (n{label_filter} {{name: $name}}) "
            "WHERE n.valid_to IS NULL "
            "RETURN n"
        )
        results = self.store.run_cypher(node_query, {"name": name})
        if not results:
            return None

        node_data = dict(results[0]["n"])

        rel_query = (
            f"MATCH (n{label_filter} {{name: $name}})-[r]-(m) "
            "WHERE n.valid_to IS NULL "
            "  AND (r.valid_to IS NULL) "
            "RETURN type(r) AS rel_type, m.name AS connected_to, "
            "       labels(m) AS connected_labels, properties(r) AS rel_props"
        )
        rel_results = self.store.run_cypher(rel_query, {"name": name})
        relationships = [
            {
                "type": r["rel_type"],
                "connected_to": r["connected_to"],
                "connected_labels": r["connected_labels"],
                "properties": r["rel_props"],
            }
            for r in rel_results
        ]

        return {"node": node_data, "relationships": relationships}

    def get_entity_history(
        self,
        name: str,
    ) -> list[dict[str, Any]]:
        """Get all temporal versions of an entity (valid and invalidated)."""
        query = (
            "MATCH (n {name: $name}) "
            "RETURN n "
            "ORDER BY n.valid_from"
        )
        results = self.store.run_cypher(query, {"name": name})
        return [dict(r["n"]) for r in results]
