"""LangChain tools for dynamic KG interaction.

All tools are BaseTool subclasses with descriptions optimized for LLM
tool selection via function calling.
"""

import sys
import json
from pathlib import Path
from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
from shared.config import get_neo4j_config


def _get_driver():
    config = get_neo4j_config()
    return GraphDatabase.driver(config["uri"], auth=(config["user"], config["password"]))


# ── Tool Input Schemas ───────────────────────────────────────────────

class CypherQueryInput(BaseModel):
    query: str = Field(description="A valid Cypher query to execute against the Neo4j knowledge graph")


class EntitySearchInput(BaseModel):
    name: str = Field(description="Entity name to search for (supports partial matching)")
    entity_type: str | None = Field(default=None, description="Optional entity type filter (e.g., 'Technology', 'Person')")


class NeighborhoodInput(BaseModel):
    entity_name: str = Field(description="The name of the entity to explore")
    hops: int = Field(default=2, description="Number of hops to traverse (1-3)")


class GapDetectionInput(BaseModel):
    min_connections: int = Field(default=2, description="Minimum connections an entity should have to not be flagged")


# ── Tools ────────────────────────────────────────────────────────────

class CypherQueryTool(BaseTool):
    """Execute a Cypher query against the Neo4j knowledge graph."""

    name: str = "cypher_query"
    description: str = (
        "Execute a Cypher query against the Neo4j knowledge graph. "
        "Use this for precise graph queries like finding specific relationships, "
        "counting entities, or complex pattern matching. "
        "Returns the query results as a JSON list of records."
    )
    args_schema: Type[BaseModel] = CypherQueryInput

    def _run(self, query: str) -> str:
        driver = _get_driver()
        try:
            with driver.session() as session:
                result = session.run(query)
                records = [dict(record) for record in result]
                # Convert Neo4j nodes/relationships to dicts
                serializable = []
                for record in records:
                    row = {}
                    for key, value in record.items():
                        if hasattr(value, "items"):
                            row[key] = dict(value)
                        elif hasattr(value, "__iter__") and not isinstance(value, str):
                            row[key] = list(value)
                        else:
                            row[key] = value
                    serializable.append(row)
                return json.dumps(serializable, indent=2, default=str)
        except Exception as e:
            return f"Cypher query error: {e}"
        finally:
            driver.close()


class EntitySearchTool(BaseTool):
    """Search for entities in the knowledge graph by name or type."""

    name: str = "entity_search"
    description: str = (
        "Search for entities in the knowledge graph by name (partial match) "
        "and optionally by type. Use this to find specific entities or explore "
        "what entities exist. Returns matching entities with their properties."
    )
    args_schema: Type[BaseModel] = EntitySearchInput

    def _run(self, name: str, entity_type: str | None = None) -> str:
        driver = _get_driver()
        try:
            with driver.session() as session:
                if entity_type:
                    result = session.run(
                        """
                        MATCH (e:Entity)
                        WHERE toLower(e.name) CONTAINS toLower($name)
                          AND e.entity_type = $entity_type
                        RETURN e.name AS name, e.entity_type AS type,
                               e.description AS description
                        LIMIT 20
                        """,
                        name=name,
                        entity_type=entity_type,
                    )
                else:
                    result = session.run(
                        """
                        MATCH (e:Entity)
                        WHERE toLower(e.name) CONTAINS toLower($name)
                        RETURN e.name AS name, e.entity_type AS type,
                               e.description AS description
                        LIMIT 20
                        """,
                        name=name,
                    )
                records = [dict(r) for r in result]
                if not records:
                    return f"No entities found matching '{name}'"
                return json.dumps(records, indent=2, default=str)
        except Exception as e:
            return f"Entity search error: {e}"
        finally:
            driver.close()


class NeighborhoodTool(BaseTool):
    """Get the N-hop neighborhood of an entity in the knowledge graph."""

    name: str = "neighborhood_tool"
    description: str = (
        "Get the N-hop neighborhood of an entity -- all connected entities "
        "and relationships within N hops. Use this to understand an entity's "
        "context, find related concepts, or trace relationship paths. "
        "Returns nodes and edges in the neighborhood."
    )
    args_schema: Type[BaseModel] = NeighborhoodInput

    def _run(self, entity_name: str, hops: int = 2) -> str:
        hops = min(max(hops, 1), 3)  # Clamp to 1-3
        driver = _get_driver()
        try:
            with driver.session() as session:
                result = session.run(
                    f"""
                    MATCH (start:Entity {{name: $name}})
                    CALL apoc.path.subgraphAll(start, {{maxLevel: $hops}})
                    YIELD nodes, relationships
                    UNWIND nodes AS node
                    WITH collect(DISTINCT {{
                        name: node.name,
                        type: node.entity_type,
                        description: node.description
                    }}) AS nodeList, relationships
                    UNWIND relationships AS rel
                    RETURN nodeList,
                           collect(DISTINCT {{
                               source: startNode(rel).name,
                               target: endNode(rel).name,
                               type: type(rel)
                           }}) AS edgeList
                    """,
                    name=entity_name,
                    hops=hops,
                )
                record = result.single()
                if record is None:
                    # Fallback: try simple pattern match
                    result2 = session.run(
                        """
                        MATCH (start:Entity {name: $name})-[r]-(neighbor:Entity)
                        RETURN start.name AS source, type(r) AS relationship,
                               neighbor.name AS target, neighbor.entity_type AS target_type
                        LIMIT 50
                        """,
                        name=entity_name,
                    )
                    records = [dict(r) for r in result2]
                    if not records:
                        return f"Entity '{entity_name}' not found or has no neighbors"
                    return json.dumps(records, indent=2, default=str)

                return json.dumps({
                    "nodes": record["nodeList"],
                    "edges": record["edgeList"],
                }, indent=2, default=str)
        except Exception as e:
            return f"Neighborhood query error: {e}"
        finally:
            driver.close()


class GapDetectionTool(BaseTool):
    """Find sparse areas in the knowledge graph that need enrichment."""

    name: str = "gap_detection"
    description: str = (
        "Analyze the knowledge graph for coverage gaps: orphan nodes (no connections), "
        "sparse entities (few connections), and underrepresented entity types. "
        "Use this to identify areas that need more information."
    )
    args_schema: Type[BaseModel] = GapDetectionInput

    def _run(self, min_connections: int = 2) -> str:
        driver = _get_driver()
        try:
            gaps = {"orphan_nodes": [], "sparse_nodes": [], "type_distribution": {}}

            with driver.session() as session:
                # Orphan nodes
                orphans = session.run(
                    """
                    MATCH (e:Entity) WHERE NOT (e)--()
                    RETURN e.name AS name, e.entity_type AS type
                    LIMIT 20
                    """
                )
                gaps["orphan_nodes"] = [dict(r) for r in orphans]

                # Sparse nodes
                sparse = session.run(
                    """
                    MATCH (e:Entity)
                    WITH e, size([(e)--() | 1]) AS degree
                    WHERE degree < $min_connections AND degree > 0
                    RETURN e.name AS name, e.entity_type AS type, degree
                    ORDER BY degree ASC
                    LIMIT 20
                    """,
                    min_connections=min_connections,
                )
                gaps["sparse_nodes"] = [dict(r) for r in sparse]

                # Type distribution
                types = session.run(
                    "MATCH (e:Entity) RETURN e.entity_type AS type, count(*) AS count ORDER BY count DESC"
                )
                gaps["type_distribution"] = {r["type"]: r["count"] for r in types}

            return json.dumps(gaps, indent=2, default=str)
        except Exception as e:
            return f"Gap detection error: {e}"
        finally:
            driver.close()


class KGStatsTool(BaseTool):
    """Get overall statistics about the knowledge graph."""

    name: str = "kg_stats"
    description: str = (
        "Get high-level statistics about the knowledge graph: total nodes, "
        "total relationships, entity type counts, relationship type counts, "
        "and graph density. Use this for an overview of the graph."
    )

    def _run(self, **kwargs) -> str:
        driver = _get_driver()
        try:
            stats = {}
            with driver.session() as session:
                # Node count
                result = session.run("MATCH (e:Entity) RETURN count(e) AS count")
                stats["total_entities"] = result.single()["count"]

                # Relationship count
                result = session.run("MATCH ()-[r]->() RETURN count(r) AS count")
                stats["total_relationships"] = result.single()["count"]

                # Entity types
                result = session.run(
                    "MATCH (e:Entity) RETURN e.entity_type AS type, count(*) AS count ORDER BY count DESC"
                )
                stats["entity_types"] = {r["type"]: r["count"] for r in result}

                # Relationship types
                result = session.run(
                    "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY count DESC"
                )
                stats["relationship_types"] = {r["type"]: r["count"] for r in result}

                # Average degree
                result = session.run(
                    """
                    MATCH (e:Entity)
                    WITH e, size([(e)--() | 1]) AS degree
                    RETURN avg(degree) AS avg_degree, max(degree) AS max_degree,
                           min(degree) AS min_degree
                    """
                )
                record = result.single()
                if record:
                    stats["avg_degree"] = round(record["avg_degree"] or 0, 2)
                    stats["max_degree"] = record["max_degree"] or 0
                    stats["min_degree"] = record["min_degree"] or 0

            return json.dumps(stats, indent=2, default=str)
        except Exception as e:
            return f"KG stats error: {e}"
        finally:
            driver.close()
