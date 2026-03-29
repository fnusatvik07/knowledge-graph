"""Neo4j connection manager and graph operations.

Provides CRUD operations for nodes and relationships, Cypher execution,
vector index creation, and vector similarity search.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

from neo4j import GraphDatabase, Driver, Session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from shared.config import get_neo4j_config
from shared.llm_clients import get_embedding


class Neo4jStore:
    """Manages a Neo4j connection and provides graph operations."""

    def __init__(self, config: dict[str, str] | None = None) -> None:
        self._config = config or get_neo4j_config()
        self._driver: Driver | None = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Establish a connection to Neo4j."""
        self._driver = GraphDatabase.driver(
            self._config["uri"],
            auth=(self._config["user"], self._config["password"]),
        )
        # Verify connectivity
        self._driver.verify_connectivity()

    def close(self) -> None:
        """Close the Neo4j driver."""
        if self._driver:
            self._driver.close()
            self._driver = None

    @property
    def driver(self) -> Driver:
        if self._driver is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._driver

    def _session(self) -> Session:
        return self.driver.session()

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def create_node(
        self,
        label: str,
        properties: dict[str, Any],
        merge: bool = True,
    ) -> dict[str, Any]:
        """Create or merge a node with the given label and properties.

        Returns the created/matched node properties.
        """
        prop_string = ", ".join(f"n.{k} = ${k}" for k in properties if k != "name")
        if merge:
            query = (
                f"MERGE (n:{label} {{name: $name}}) "
                f"SET {prop_string} "
                "RETURN n"
            )
        else:
            query = (
                f"CREATE (n:{label} {{name: $name}}) "
                f"SET {prop_string} "
                "RETURN n"
            )
        with self._session() as session:
            result = session.run(query, **properties)
            record = result.single()
            return dict(record["n"]) if record else {}

    def create_relationship(
        self,
        source_name: str,
        source_label: str,
        target_name: str,
        target_label: str,
        rel_type: str,
        properties: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Create or merge a relationship between two nodes.

        Nodes are matched by name and label. The relationship is merged.
        """
        props = properties or {}
        set_clause = ""
        if props:
            set_parts = ", ".join(f"r.{k} = ${k}" for k in props)
            set_clause = f"SET {set_parts}"

        query = (
            f"MATCH (a:{source_label} {{name: $source_name}}) "
            f"MATCH (b:{target_label} {{name: $target_name}}) "
            f"MERGE (a)-[r:{rel_type}]->(b) "
            f"{set_clause} "
            "RETURN type(r) AS rel_type, properties(r) AS props"
        )
        params = {"source_name": source_name, "target_name": target_name, **props}
        with self._session() as session:
            result = session.run(query, **params)
            record = result.single()
            if record:
                return {"type": record["rel_type"], "properties": record["props"]}
            return None

    def get_node(self, label: str, name: str) -> dict[str, Any] | None:
        """Retrieve a single node by label and name."""
        query = f"MATCH (n:{label} {{name: $name}}) RETURN n"
        with self._session() as session:
            result = session.run(query, name=name)
            record = result.single()
            return dict(record["n"]) if record else None

    def get_neighbors(
        self,
        name: str,
        label: str | None = None,
        max_hops: int = 1,
        rel_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Get neighbor nodes up to max_hops away.

        Returns a list of dicts with keys: node, relationship, depth.
        """
        label_filter = f":{label}" if label else ""
        rel_filter = ":" + "|".join(rel_types) if rel_types else ""

        query = (
            f"MATCH (start {{name: $name}})"
            f"-[r{rel_filter}*1..{max_hops}]-"
            f"(neighbor{label_filter}) "
            "RETURN neighbor, r, length(r) AS depth "
            "LIMIT 50"
        )
        neighbors: list[dict[str, Any]] = []
        with self._session() as session:
            result = session.run(query, name=name)
            for record in result:
                neighbors.append({
                    "node": dict(record["neighbor"]),
                    "depth": record["depth"],
                })
        return neighbors

    # ------------------------------------------------------------------
    # Raw Cypher
    # ------------------------------------------------------------------

    def run_cypher(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute an arbitrary Cypher query and return results as dicts."""
        with self._session() as session:
            result = session.run(query, **(parameters or {}))
            return [dict(record) for record in result]

    # ------------------------------------------------------------------
    # Vector index
    # ------------------------------------------------------------------

    def create_vector_index(
        self,
        index_name: str = "entity_embeddings",
        label: str = "Entity",
        property_name: str = "embedding",
        dimensions: int = 1536,
        similarity: str = "cosine",
    ) -> None:
        """Create a vector index on the given label and property."""
        query = (
            f"CREATE VECTOR INDEX {index_name} IF NOT EXISTS "
            f"FOR (n:{label}) ON (n.{property_name}) "
            f"OPTIONS {{indexConfig: {{"
            f"  `vector.dimensions`: {dimensions},"
            f"  `vector.similarity_function`: '{similarity}'"
            f"}}}}"
        )
        with self._session() as session:
            session.run(query)

    def vector_search(
        self,
        query_text: str,
        index_name: str = "entity_embeddings",
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Embed a query and search the vector index.

        Returns a list of dicts with keys: node, score.
        """
        query_embedding = get_embedding(query_text)
        cypher = (
            f"CALL db.index.vector.queryNodes('{index_name}', $top_k, $embedding) "
            "YIELD node, score "
            "RETURN node, score"
        )
        with self._session() as session:
            result = session.run(cypher, top_k=top_k, embedding=query_embedding)
            return [
                {"node": dict(record["node"]), "score": record["score"]}
                for record in result
            ]

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def clear_database(self) -> None:
        """Delete all nodes and relationships. Use with caution."""
        with self._session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def __enter__(self) -> "Neo4jStore":
        self.connect()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
