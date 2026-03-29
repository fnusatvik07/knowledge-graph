"""Temporal queries for the Real-Time Streaming Knowledge Graph.

Query the knowledge graph with temporal awareness:
- What was true at a specific point in time T
- What changed between time T1 and T2
- Which facts were invalidated and why
- Timeline of an entity's evolution

Uses temporal metadata (valid_from, valid_until) stored on nodes and edges.

Neo4j Cypher temporal docs: https://neo4j.com/docs/cypher-manual/current/values-and-types/temporal/

Usage:
    python src/05_temporal_queries.py
    python src/05_temporal_queries.py --entity "TechCorp"
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timezone

# Shared imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from neo4j import GraphDatabase

# ── Neo4j connection ─────────────────────────────────────────────────────
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# ── Temporal query functions ─────────────────────────────────────────────

def query_state_at_time(driver, timestamp: str):
    """Show the state of the knowledge graph at a specific point in time.

    Returns all entities and relationships that were valid at the given timestamp.
    """
    print(f"\n{'='*60}")
    print(f"Graph State at: {timestamp}")
    print(f"{'='*60}")

    with driver.session() as session:
        # Entities valid at time T
        result = session.run("""
            MATCH (e:Entity)
            WHERE e.valid_from <= $timestamp
              AND (e.valid_until IS NULL OR e.valid_until > $timestamp)
            RETURN e.name AS name, e.entity_type AS type, e.description AS desc,
                   e.valid_from AS since
            ORDER BY e.valid_from
        """, {"timestamp": timestamp})

        entities = list(result)
        print(f"\n  Entities valid at this time: {len(entities)}")
        for r in entities:
            print(f"    [{r['type']}] {r['name']}")
            print(f"      {r['desc'][:80]}")
            print(f"      Valid since: {r['since']}")

        # Relationships valid at time T
        result = session.run("""
            MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
            WHERE r.valid_from <= $timestamp
              AND (r.valid_until IS NULL OR r.valid_until > $timestamp)
              AND r.invalidated = false
            RETURN s.name AS source, r.rel_type AS rel, t.name AS target,
                   r.description AS desc, r.valid_from AS since
            ORDER BY r.valid_from
        """, {"timestamp": timestamp})

        rels = list(result)
        print(f"\n  Active relationships at this time: {len(rels)}")
        for r in rels:
            print(f"    {r['source']} --[{r['rel']}]--> {r['target']}")
            print(f"      {r['desc'][:80]}")


def query_changes_between(driver, t1: str, t2: str):
    """Show all changes that occurred between two timestamps.

    Includes new entities, new relationships, and invalidated facts.
    """
    print(f"\n{'='*60}")
    print(f"Changes Between: {t1}")
    print(f"            And: {t2}")
    print(f"{'='*60}")

    with driver.session() as session:
        # New entities added in time range
        result = session.run("""
            MATCH (e:Entity)
            WHERE e.valid_from >= $t1 AND e.valid_from <= $t2
            RETURN e.name AS name, e.entity_type AS type, e.valid_from AS added_at
            ORDER BY e.valid_from
        """, {"t1": t1, "t2": t2})

        new_entities = list(result)
        print(f"\n  New entities added: {len(new_entities)}")
        for r in new_entities:
            print(f"    + [{r['type']}] {r['name']} (at {r['added_at']})")

        # New relationships in time range
        result = session.run("""
            MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
            WHERE r.valid_from >= $t1 AND r.valid_from <= $t2
            RETURN s.name AS source, r.rel_type AS rel, t.name AS target,
                   r.valid_from AS added_at
            ORDER BY r.valid_from
        """, {"t1": t1, "t2": t2})

        new_rels = list(result)
        print(f"\n  New relationships added: {len(new_rels)}")
        for r in new_rels:
            print(f"    + {r['source']} --[{r['rel']}]--> {r['target']} (at {r['added_at']})")

        # Facts invalidated in time range
        result = session.run("""
            MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
            WHERE r.invalidated = true
              AND r.valid_until >= $t1 AND r.valid_until <= $t2
            RETURN s.name AS source, r.rel_type AS rel, t.name AS target,
                   r.description AS desc, r.invalidation_reason AS reason,
                   r.valid_until AS invalidated_at
            ORDER BY r.valid_until
        """, {"t1": t1, "t2": t2})

        invalidated = list(result)
        print(f"\n  Facts invalidated: {len(invalidated)}")
        for r in invalidated:
            print(f"    X {r['source']} --[{r['rel']}]--> {r['target']}")
            print(f"      Reason: {r['reason'][:80] if r['reason'] else 'N/A'}")
            print(f"      Invalidated at: {r['invalidated_at']}")


def query_invalidated_facts(driver):
    """Show all facts that have been invalidated, with reasons."""
    print(f"\n{'='*60}")
    print("All Invalidated Facts")
    print(f"{'='*60}")

    with driver.session() as session:
        result = session.run("""
            MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
            WHERE r.invalidated = true
            RETURN s.name AS source, r.rel_type AS rel, t.name AS target,
                   r.description AS desc, r.invalidation_reason AS reason,
                   r.valid_from AS valid_from, r.valid_until AS valid_until
            ORDER BY r.valid_until DESC
        """)

        records = list(result)
        if not records:
            print("\n  No invalidated facts found.")
            return

        print(f"\n  Total invalidated facts: {len(records)}\n")
        for r in records:
            print(f"  Fact: {r['source']} --[{r['rel']}]--> {r['target']}")
            print(f"    Description: {r['desc'][:80] if r['desc'] else 'N/A'}")
            print(f"    Valid from:  {r['valid_from']}")
            print(f"    Valid until: {r['valid_until']}")
            print(f"    Reason:      {r['reason'] if r['reason'] else 'N/A'}")
            print()


def query_entity_timeline(driver, entity_name: str):
    """Show the complete timeline of an entity: when it appeared, how its
    relationships evolved, and what changed over time."""
    print(f"\n{'='*60}")
    print(f"Entity Timeline: {entity_name}")
    print(f"{'='*60}")

    with driver.session() as session:
        # Find the entity
        result = session.run("""
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower($name)
            RETURN e.name AS name, e.entity_type AS type, e.description AS desc,
                   e.valid_from AS created, e.last_updated AS updated,
                   e.mention_count AS mentions
        """, {"name": entity_name})

        entities = list(result)
        if not entities:
            print(f"\n  No entity found matching '{entity_name}'")
            return

        for e in entities:
            print(f"\n  Entity: {e['name']} ({e['type']})")
            print(f"  Description: {e['desc'][:80] if e['desc'] else 'N/A'}")
            print(f"  Created: {e['created']}")
            print(f"  Last Updated: {e['updated'] or 'N/A'}")
            print(f"  Mention Count: {e['mentions'] or 1}")

        # All relationships (active and invalidated)
        result = session.run("""
            MATCH (e:Entity)-[r:RELATES_TO]-(other:Entity)
            WHERE toLower(e.name) CONTAINS toLower($name)
            RETURN e.name AS entity, type(r) AS rel_type, r.rel_type AS rel_label,
                   other.name AS other_entity, r.description AS desc,
                   r.valid_from AS since, r.valid_until AS until,
                   r.invalidated AS invalidated, r.invalidation_reason AS reason,
                   startNode(r) = e AS is_outgoing
            ORDER BY r.valid_from
        """, {"name": entity_name})

        rels = list(result)
        print(f"\n  Relationship Timeline ({len(rels)} total):")
        for r in rels:
            direction = "-->" if r["is_outgoing"] else "<--"
            status = " [INVALIDATED]" if r["invalidated"] else " [ACTIVE]"
            print(f"\n    {r['entity']} {direction}[{r['rel_label']}]{direction} {r['other_entity']}{status}")
            print(f"      {r['desc'][:70] if r['desc'] else 'N/A'}")
            print(f"      Valid: {r['since']} to {r['until'] or 'present'}")
            if r["invalidated"] and r["reason"]:
                print(f"      Invalidation reason: {r['reason'][:70]}")


def show_full_graph_summary(driver):
    """Show a high-level summary of the entire graph with temporal info."""
    print(f"\n{'='*60}")
    print("Full Graph Summary")
    print(f"{'='*60}")

    with driver.session() as session:
        # Overall stats
        stats = session.run("""
            MATCH (e:Entity)
            WITH count(e) AS entities
            OPTIONAL MATCH ()-[r:RELATES_TO]->()
            WITH entities, count(r) AS total_rels
            OPTIONAL MATCH ()-[r2:RELATES_TO]->()
            WHERE r2.invalidated = true
            RETURN entities, total_rels, count(r2) AS invalidated
        """).single()

        print(f"\n  Entities:         {stats['entities']}")
        print(f"  Relationships:    {stats['total_rels']}")
        print(f"  Invalidated:      {stats['invalidated']}")

        # Entity type distribution
        result = session.run("""
            MATCH (e:Entity)
            RETURN e.entity_type AS type, count(e) AS count
            ORDER BY count DESC
        """)
        print("\n  Entity Type Distribution:")
        for r in result:
            print(f"    {r['type']:<20} {r['count']}")

        # Source documents processed
        result = session.run("""
            MATCH (s:Source)
            RETURN s.filename AS file, s.processed_at AS time
            ORDER BY s.processed_at
        """)
        print("\n  Sources Processed:")
        for r in result:
            print(f"    {r['file']} (at {r['time']})")

        # Time range
        result = session.run("""
            MATCH (e:Entity)
            RETURN min(e.valid_from) AS earliest, max(e.valid_from) AS latest
        """).single()
        if result["earliest"]:
            print(f"\n  Time Range: {result['earliest']} to {result['latest']}")


def main():
    parser = argparse.ArgumentParser(description="Temporal Queries for Streaming KG")
    parser.add_argument("--entity", type=str, help="Show timeline for a specific entity")
    parser.add_argument("--at", type=str, help="Show graph state at a specific ISO timestamp")
    parser.add_argument("--from", type=str, dest="from_time", help="Start of time range for change query")
    parser.add_argument("--to", type=str, dest="to_time", help="End of time range for change query")
    args = parser.parse_args()

    driver = get_driver()

    try:
        driver.verify_connectivity()
        print("Connected to Neo4j.\n")

        if args.entity:
            query_entity_timeline(driver, args.entity)

        elif args.at:
            query_state_at_time(driver, args.at)

        elif args.from_time and args.to_time:
            query_changes_between(driver, args.from_time, args.to_time)

        else:
            # Default: show everything
            show_full_graph_summary(driver)
            query_invalidated_facts(driver)

            # Show all entity timelines
            with driver.session() as session:
                result = session.run("""
                    MATCH (e:Entity)
                    WHERE e.mention_count > 1
                    RETURN DISTINCT e.name AS name
                    ORDER BY e.mention_count DESC
                    LIMIT 5
                """)
                top_entities = [r["name"] for r in result]

            if top_entities:
                print(f"\n\nTimelines for top {len(top_entities)} most-mentioned entities:")
                for name in top_entities:
                    query_entity_timeline(driver, name)

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Neo4j is running: docker compose up -d")
        sys.exit(1)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
