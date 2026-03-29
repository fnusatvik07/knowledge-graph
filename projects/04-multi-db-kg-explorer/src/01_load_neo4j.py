"""Load the shared KG dataset into Neo4j using the official neo4j Python driver.

Neo4j driver docs: https://neo4j.com/docs/python-manual/current/
Cypher reference: https://neo4j.com/docs/cypher-manual/current/

Prerequisites:
    docker compose up -d neo4j
    pip install neo4j
"""

import json
import sys
from pathlib import Path

# Shared imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from neo4j import GraphDatabase

# ── Configuration ────────────────────────────────────────────────────────
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password123"

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "entities_and_relations.json"


def load_data() -> dict:
    """Load the shared entities and relations JSON."""
    with open(DATA_FILE) as f:
        return json.load(f)


def clear_database(driver):
    """Remove all nodes and relationships."""
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("[neo4j] Cleared all existing data.")


def create_constraints(driver):
    """Create uniqueness constraints for each entity type.

    Cypher constraints docs:
    https://neo4j.com/docs/cypher-manual/current/constraints/
    """
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Company) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Model) REQUIRE m.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Architecture) REQUIRE a.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Technique) REQUIRE t.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Framework) REQUIRE f.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (h:Hardware) REQUIRE h.id IS UNIQUE",
    ]
    with driver.session() as session:
        for c in constraints:
            session.run(c)
    print(f"[neo4j] Created {len(constraints)} uniqueness constraints.")


def load_entities(driver, entities: list[dict]):
    """Load entities as labeled nodes using MERGE (idempotent).

    Cypher MERGE docs:
    https://neo4j.com/docs/cypher-manual/current/clauses/merge/
    """
    query = """
    MERGE (n:{label} {{id: $id}})
    SET n += $props
    """
    with driver.session() as session:
        for entity in entities:
            label = entity["type"]
            props = {k: v for k, v in entity.items() if k != "type"}
            # Cypher doesn't allow parameterized labels, so we format the label
            session.run(
                f"MERGE (n:{label} {{id: $id}}) SET n += $props",
                id=entity["id"],
                props=props,
            )
    print(f"[neo4j] Loaded {len(entities)} entities.")


def load_relationships(driver, relationships: list[dict]):
    """Load relationships between entities.

    We look up source/target by id across all labels using a generic match.
    """
    with driver.session() as session:
        for rel in relationships:
            rel_type = rel["type"]
            props = {k: v for k, v in rel.items() if k not in ("source", "target", "type")}
            # Match nodes by id regardless of label
            cypher = f"""
            MATCH (a {{id: $source}})
            MATCH (b {{id: $target}})
            MERGE (a)-[r:{rel_type}]->(b)
            SET r += $props
            """
            session.run(cypher, source=rel["source"], target=rel["target"], props=props)
    print(f"[neo4j] Loaded {len(relationships)} relationships.")


def run_example_queries(driver):
    """Run example Cypher queries to verify the data."""
    queries = {
        "Total nodes": "MATCH (n) RETURN count(n) AS count",
        "Total relationships": "MATCH ()-[r]->() RETURN count(r) AS count",
        "Companies": "MATCH (c:Company) RETURN c.name ORDER BY c.name",
        "Models and their creators": """
            MATCH (c:Company)-[:DEVELOPED]->(m:Model)
            RETURN c.name AS company, m.name AS model
            ORDER BY c.name
        """,
        "2-hop: Companies connected via shared architecture": """
            MATCH (c1:Company)-[:DEVELOPED]->(m1:Model)-[:USES_ARCHITECTURE]->(a:Architecture)
                  <-[:USES_ARCHITECTURE]-(m2:Model)<-[:DEVELOPED]-(c2:Company)
            WHERE c1.id < c2.id
            RETURN DISTINCT c1.name, c2.name, a.name
        """,
    }

    print("\n" + "=" * 60)
    print("NEO4J EXAMPLE QUERIES")
    print("=" * 60)

    with driver.session() as session:
        for title, query in queries.items():
            result = session.run(query)
            records = [dict(r) for r in result]
            print(f"\n--- {title} ---")
            for record in records:
                print(f"  {record}")


def main():
    print("=" * 60)
    print("Loading KG into Neo4j")
    print("=" * 60)

    data = load_data()
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        # Verify connectivity
        driver.verify_connectivity()
        print("[neo4j] Connected successfully.")

        clear_database(driver)
        create_constraints(driver)
        load_entities(driver, data["entities"])
        load_relationships(driver, data["relationships"])
        run_example_queries(driver)

    finally:
        driver.close()
        print("\n[neo4j] Connection closed.")


if __name__ == "__main__":
    main()
