"""Build a biomedical knowledge graph in Neo4j from extracted entities and relationships.

Creates properly labeled nodes (Gene, Disease, Drug, Protein, etc.) and typed
relationships with indexes and constraints for efficient querying.

Usage:
    python src/04_build_biomedical_kg.py
    python src/04_build_biomedical_kg.py --neo4j_uri bolt://localhost:7687 --clear

Prerequisites:
    docker run -d -p 7687:7687 -p 7474:7474 -e NEO4J_AUTH=neo4j/password neo4j:5

References:
    - Neo4j Python driver: https://neo4j.com/docs/python-manual/current/
    - LangChain Neo4j: https://python.langchain.com/docs/integrations/graphs/neo4j_cypher/
"""

import argparse
import json
import sys
from pathlib import Path

# ── Shared imports ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from neo4j import GraphDatabase

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

# ── Neo4j connection defaults ────────────────────────────────────────────
DEFAULT_URI = "bolt://localhost:7687"
DEFAULT_USER = "neo4j"
DEFAULT_PASSWORD = "password"


def create_constraints_and_indexes(tx):
    """Create uniqueness constraints and indexes for biomedical entity types.

    Constraints ensure no duplicate entities; indexes accelerate lookups.
    """
    entity_types = ["Gene", "Protein", "Disease", "Drug", "Pathway", "CellType", "Organism"]

    for entity_type in entity_types:
        # Uniqueness constraint on name (also creates an index)
        tx.run(f"""
            CREATE CONSTRAINT {entity_type.lower()}_name_unique IF NOT EXISTS
            FOR (n:{entity_type}) REQUIRE n.name IS UNIQUE
        """)

    # Additional indexes for common queries
    tx.run("CREATE INDEX pmid_index IF NOT EXISTS FOR (n:Entity) ON (n.pmid)")
    tx.run("CREATE INDEX entity_name_index IF NOT EXISTS FOR (n:Entity) ON (n.name)")

    print("Created constraints and indexes for all entity types.")


def clear_database(tx):
    """Remove all nodes and relationships from the database."""
    tx.run("MATCH (n) DETACH DELETE n")
    print("Cleared all data from the database.")


def create_entity_node(tx, entity: dict):
    """Create or merge a biomedical entity node with proper labels.

    Each node gets both its specific type label (e.g., Gene) and a generic
    Entity label for cross-type queries.
    """
    entity_type = entity["entity_type"]
    name = entity["name"]
    synonyms = entity.get("synonyms", [])
    description = entity.get("description", "")
    pmid = entity.get("pmid", "")

    query = f"""
        MERGE (n:{entity_type} {{name: $name}})
        SET n:Entity,
            n.synonyms = $synonyms,
            n.description = $description
        WITH n
        SET n.pmids = CASE
            WHEN n.pmids IS NULL THEN [$pmid]
            WHEN NOT $pmid IN n.pmids THEN n.pmids + $pmid
            ELSE n.pmids
        END
    """
    tx.run(query, name=name, synonyms=synonyms, description=description, pmid=pmid)


def create_relationship(tx, rel: dict):
    """Create a typed relationship between two entity nodes.

    Uses the source and target types to match the correct nodes,
    and the relationship type to create the correct edge.
    """
    source_type = rel["source_type"]
    target_type = rel["target_type"]
    rel_type = rel["relationship_type"]
    source_name = rel["source_name"]
    target_name = rel["target_name"]
    confidence = rel.get("confidence", 1.0)
    evidence = rel.get("evidence", "")
    pmid = rel.get("pmid", "")

    query = f"""
        MATCH (s:{source_type} {{name: $source_name}})
        MATCH (t:{target_type} {{name: $target_name}})
        MERGE (s)-[r:{rel_type}]->(t)
        SET r.confidence = $confidence,
            r.evidence = $evidence,
            r.pmid = $pmid
    """
    tx.run(
        query,
        source_name=source_name,
        target_name=target_name,
        confidence=confidence,
        evidence=evidence,
        pmid=pmid,
    )


def main():
    parser = argparse.ArgumentParser(description="Build biomedical KG in Neo4j.")
    parser.add_argument("--neo4j_uri", type=str, default=DEFAULT_URI)
    parser.add_argument("--neo4j_user", type=str, default=DEFAULT_USER)
    parser.add_argument("--neo4j_password", type=str, default=DEFAULT_PASSWORD)
    parser.add_argument("--clear", action="store_true", help="Clear existing data before loading")
    args = parser.parse_args()

    # Load extracted data
    entities_file = PROCESSED_DIR / "extracted_entities.json"
    relations_file = PROCESSED_DIR / "extracted_relationships.json"

    if not entities_file.exists():
        print(f"ERROR: Entities file not found: {entities_file}")
        print("Run 02_extract_biomedical_entities.py first.")
        sys.exit(1)

    if not relations_file.exists():
        print(f"ERROR: Relationships file not found: {relations_file}")
        print("Run 03_extract_relations.py first.")
        sys.exit(1)

    with open(entities_file) as f:
        entities = json.load(f)
    with open(relations_file) as f:
        relationships = json.load(f)

    print(f"Loaded {len(entities)} entities and {len(relationships)} relationships.")

    # Connect to Neo4j
    print(f"\nConnecting to Neo4j at {args.neo4j_uri}...")
    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))

    try:
        driver.verify_connectivity()
        print("Connected successfully.")

        with driver.session() as session:
            # Optionally clear existing data
            if args.clear:
                session.execute_write(clear_database)

            # Create constraints and indexes
            session.execute_write(create_constraints_and_indexes)

            # Create entity nodes
            print(f"\nCreating {len(entities)} entity nodes...")
            entity_counts = {}
            for entity in entities:
                session.execute_write(create_entity_node, entity)
                t = entity["entity_type"]
                entity_counts[t] = entity_counts.get(t, 0) + 1

            for t, c in sorted(entity_counts.items()):
                print(f"  {t}: {c} nodes")

            # Create relationships
            print(f"\nCreating {len(relationships)} relationships...")
            rel_counts = {}
            errors = 0
            for rel in relationships:
                try:
                    session.execute_write(create_relationship, rel)
                    t = rel["relationship_type"]
                    rel_counts[t] = rel_counts.get(t, 0) + 1
                except Exception as e:
                    errors += 1
                    print(f"  ERROR creating {rel['source_name']} -[{rel['relationship_type']}]-> "
                          f"{rel['target_name']}: {e}")

            for t, c in sorted(rel_counts.items()):
                print(f"  {t}: {c} relationships")

            if errors:
                print(f"\n  {errors} relationships failed (likely missing nodes)")

            # Verify graph statistics
            result = session.run("MATCH (n:Entity) RETURN count(n) AS node_count")
            node_count = result.single()["node_count"]
            result = session.run("MATCH ()-[r]->() RETURN count(r) AS rel_count")
            rel_count = result.single()["rel_count"]

            print(f"\n{'='*60}")
            print(f"Biomedical KG built successfully!")
            print(f"  Total nodes: {node_count}")
            print(f"  Total relationships: {rel_count}")
            print(f"\nView in Neo4j Browser: http://localhost:7474")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
