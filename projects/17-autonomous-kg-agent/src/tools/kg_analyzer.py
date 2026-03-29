"""KG Analyzer -- tools for knowledge graph introspection.

Provides functions for getting statistics, coverage scores, quality
scores, and finding structural issues like orphans and duplicates.
"""

import sys
from pathlib import Path

from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
from shared.config import get_neo4j_config


def _get_driver():
    config = get_neo4j_config()
    return GraphDatabase.driver(config["uri"], auth=(config["user"], config["password"]))


def get_kg_stats() -> dict:
    """Get overall KG statistics: node/edge counts, type distributions.

    Returns:
        dict with total_entities, total_relationships, entity_types,
        relationship_types, avg_degree
    """
    driver = _get_driver()
    stats = {}

    with driver.session() as session:
        # Entity count
        result = session.run("MATCH (e:Entity) RETURN count(e) AS count")
        stats["total_entities"] = result.single()["count"]

        # Relationship count
        result = session.run("MATCH ()-[r]->() RETURN count(r) AS count")
        stats["total_relationships"] = result.single()["count"]

        # Entity type distribution
        result = session.run(
            "MATCH (e:Entity) RETURN e.entity_type AS type, count(*) AS count ORDER BY count DESC"
        )
        stats["entity_types"] = {r["type"]: r["count"] for r in result}

        # Relationship type distribution
        result = session.run(
            "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY count DESC"
        )
        stats["relationship_types"] = {r["type"]: r["count"] for r in result}

        # Average degree
        result = session.run(
            """
            MATCH (e:Entity)
            WITH e, size([(e)--() | 1]) AS degree
            RETURN avg(degree) AS avg_degree,
                   max(degree) AS max_degree,
                   min(degree) AS min_degree
            """
        )
        record = result.single()
        if record and record["avg_degree"] is not None:
            stats["avg_degree"] = round(record["avg_degree"], 2)
            stats["max_degree"] = record["max_degree"]
            stats["min_degree"] = record["min_degree"]
        else:
            stats["avg_degree"] = 0.0
            stats["max_degree"] = 0
            stats["min_degree"] = 0

    driver.close()
    return stats


def get_coverage_score() -> float:
    """Calculate how well the KG covers its domain (0-1).

    Based on:
    - Number of entity types represented (diversity)
    - Total entities (breadth)
    - Relationship density (depth)

    Returns:
        Coverage score from 0.0 to 1.0
    """
    driver = _get_driver()
    score = 0.0

    with driver.session() as session:
        # Entity count component (up to 50 entities = full score on this axis)
        result = session.run("MATCH (e:Entity) RETURN count(e) AS count")
        entity_count = result.single()["count"]
        entity_score = min(entity_count / 50.0, 1.0) * 0.3

        # Type diversity component (target: at least 5 distinct types)
        result = session.run("MATCH (e:Entity) RETURN count(DISTINCT e.entity_type) AS types")
        type_count = result.single()["types"]
        type_score = min(type_count / 5.0, 1.0) * 0.3

        # Relationship density component (target: avg degree >= 3)
        result = session.run(
            """
            MATCH (e:Entity)
            WITH e, size([(e)--() | 1]) AS degree
            RETURN avg(degree) AS avg_degree
            """
        )
        avg_degree = result.single()["avg_degree"] or 0
        density_score = min(avg_degree / 3.0, 1.0) * 0.2

        # Connectivity component (percentage of entities with at least 1 relationship)
        result = session.run(
            """
            MATCH (e:Entity)
            WITH count(e) AS total,
                 sum(CASE WHEN size([(e)--() | 1]) > 0 THEN 1 ELSE 0 END) AS connected
            RETURN CASE WHEN total > 0 THEN toFloat(connected) / total ELSE 0.0 END AS ratio
            """
        )
        connectivity = result.single()["ratio"]
        connectivity_score = connectivity * 0.2

        score = entity_score + type_score + density_score + connectivity_score

    driver.close()
    return round(min(score, 1.0), 3)


def get_quality_score() -> float:
    """Calculate KG quality based on density, connectivity, and consistency.

    Factors:
    - Relationship density (entities should have multiple connections)
    - No orphan nodes (all entities connected)
    - Type consistency (all entities have types)
    - Provenance (entities have source information)

    Returns:
        Quality score from 0.0 to 1.0
    """
    driver = _get_driver()

    with driver.session() as session:
        result = session.run("MATCH (e:Entity) RETURN count(e) AS count")
        total = result.single()["count"]

        if total == 0:
            driver.close()
            return 0.0

        # Factor 1: Connectivity (no orphans)
        result = session.run(
            """
            MATCH (e:Entity)
            WITH count(e) AS total,
                 sum(CASE WHEN size([(e)--() | 1]) > 0 THEN 1 ELSE 0 END) AS connected
            RETURN CASE WHEN total > 0 THEN toFloat(connected) / total ELSE 0.0 END AS ratio
            """
        )
        connectivity = result.single()["ratio"]

        # Factor 2: Density (average degree, target 3+)
        result = session.run(
            """
            MATCH (e:Entity)
            WITH size([(e)--() | 1]) AS degree
            RETURN avg(degree) AS avg_deg
            """
        )
        avg_deg = result.single()["avg_deg"] or 0
        density = min(avg_deg / 3.0, 1.0)

        # Factor 3: Type completeness (all entities should have a type)
        result = session.run(
            """
            MATCH (e:Entity)
            WHERE e.entity_type IS NOT NULL AND e.entity_type <> 'Unknown'
            RETURN count(e) AS typed
            """
        )
        typed = result.single()["typed"]
        type_completeness = typed / total if total > 0 else 0.0

        # Factor 4: Provenance (entities should have source info)
        result = session.run(
            """
            MATCH (e:Entity)
            WHERE e.source IS NOT NULL OR e.source_doc IS NOT NULL
            RETURN count(e) AS sourced
            """
        )
        sourced = result.single()["sourced"]
        provenance = sourced / total if total > 0 else 0.0

        quality = (
            connectivity * 0.3
            + density * 0.3
            + type_completeness * 0.2
            + provenance * 0.2
        )

    driver.close()
    return round(min(quality, 1.0), 3)


def find_orphan_entities() -> list[dict]:
    """Find entities with no relationships.

    Returns:
        List of dicts with name and entity_type
    """
    driver = _get_driver()

    with driver.session() as session:
        result = session.run(
            """
            MATCH (e:Entity) WHERE NOT (e)--()
            RETURN e.name AS name, e.entity_type AS entity_type
            ORDER BY e.name
            """
        )
        orphans = [{"name": r["name"], "entity_type": r["entity_type"]} for r in result]

    driver.close()
    return orphans


def find_duplicate_candidates() -> list[dict]:
    """Find entities with similar names that may be duplicates.

    Uses case-insensitive comparison and Jaro-Winkler similarity.

    Returns:
        List of dicts with name_a, name_b, similarity info
    """
    driver = _get_driver()
    duplicates = []

    with driver.session() as session:
        # Exact case-insensitive duplicates
        result = session.run(
            """
            MATCH (a:Entity), (b:Entity)
            WHERE id(a) < id(b)
              AND toLower(a.name) = toLower(b.name)
            RETURN a.name AS name_a, b.name AS name_b,
                   a.entity_type AS type_a, b.entity_type AS type_b
            """
        )
        for r in result:
            duplicates.append({
                "name_a": r["name_a"],
                "name_b": r["name_b"],
                "type_a": r["type_a"],
                "type_b": r["type_b"],
                "similarity": "exact_case_insensitive",
                "canonical": r["name_a"],
                "duplicate": r["name_b"],
            })

        # Fuzzy duplicates using Jaro-Winkler (if APOC available)
        try:
            result = session.run(
                """
                MATCH (a:Entity), (b:Entity)
                WHERE id(a) < id(b)
                  AND toLower(a.name) <> toLower(b.name)
                  AND apoc.text.jaroWinklerDistance(toLower(a.name), toLower(b.name)) > 0.9
                RETURN a.name AS name_a, b.name AS name_b,
                       a.entity_type AS type_a, b.entity_type AS type_b,
                       apoc.text.jaroWinklerDistance(toLower(a.name), toLower(b.name)) AS similarity_score
                LIMIT 20
                """
            )
            for r in result:
                duplicates.append({
                    "name_a": r["name_a"],
                    "name_b": r["name_b"],
                    "type_a": r["type_a"],
                    "type_b": r["type_b"],
                    "similarity": f"jaro_winkler:{r['similarity_score']:.3f}",
                    "canonical": r["name_a"],
                    "duplicate": r["name_b"],
                })
        except Exception:
            pass  # APOC may not be available

    driver.close()
    return duplicates


if __name__ == "__main__":
    stats = get_kg_stats()
    print("KG Stats:", stats)
    print(f"Coverage: {get_coverage_score():.3f}")
    print(f"Quality: {get_quality_score():.3f}")
    print(f"Orphans: {find_orphan_entities()}")
    print(f"Duplicates: {find_duplicate_candidates()}")
