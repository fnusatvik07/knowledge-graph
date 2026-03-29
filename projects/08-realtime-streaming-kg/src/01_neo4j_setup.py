"""Initialize Neo4j schema for the Real-Time Streaming Knowledge Graph.

Sets up constraints, indexes (including vector index), and clears existing data if requested.

Neo4j Python Driver docs: https://neo4j.com/docs/python-manual/current/
LangChain Neo4j integration: https://python.langchain.com/docs/integrations/graphs/neo4j_cypher/

Usage:
    python src/01_neo4j_setup.py              # Setup schema (keep existing data)
    python src/01_neo4j_setup.py --clear      # Clear all data and recreate schema
"""

import sys
import argparse
from pathlib import Path

# Shared imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from neo4j import GraphDatabase

# ── Neo4j connection settings ────────────────────────────────────────────
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"


def get_driver():
    """Create a Neo4j driver instance."""
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def clear_database(driver):
    """Remove all nodes and relationships from the database."""
    print("Clearing all existing data...")
    with driver.session() as session:
        # Drop all constraints and indexes first
        result = session.run("SHOW CONSTRAINTS")
        for record in result:
            name = record["name"]
            session.run(f"DROP CONSTRAINT {name} IF EXISTS")
            print(f"  Dropped constraint: {name}")

        result = session.run("SHOW INDEXES")
        for record in result:
            name = record["name"]
            if record.get("type") != "LOOKUP":  # Keep default lookup indexes
                session.run(f"DROP INDEX {name} IF EXISTS")
                print(f"  Dropped index: {name}")

        # Delete all nodes and relationships in batches
        while True:
            result = session.run(
                "MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(n) AS deleted"
            )
            deleted = result.single()["deleted"]
            if deleted == 0:
                break
            print(f"  Deleted {deleted} nodes...")

    print("Database cleared.\n")


def create_constraints(driver):
    """Create uniqueness constraints for entity nodes."""
    constraints = [
        # Each entity has a unique ID (normalized name + type)
        ("entity_id_unique", "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE"),
        # Each document source is unique
        ("source_id_unique", "CREATE CONSTRAINT source_id_unique IF NOT EXISTS FOR (s:Source) REQUIRE s.source_id IS UNIQUE"),
    ]

    print("Creating constraints...")
    with driver.session() as session:
        for name, query in constraints:
            session.run(query)
            print(f"  Created constraint: {name}")
    print()


def create_indexes(driver):
    """Create indexes for efficient querying."""
    indexes = [
        # Index on entity name for fast lookups
        ("idx_entity_name", "CREATE INDEX idx_entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)"),
        # Index on entity type for type-filtered queries
        ("idx_entity_type", "CREATE INDEX idx_entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)"),
        # Index on timestamps for temporal queries
        ("idx_entity_valid_from", "CREATE INDEX idx_entity_valid_from IF NOT EXISTS FOR (e:Entity) ON (e.valid_from)"),
        ("idx_entity_valid_until", "CREATE INDEX idx_entity_valid_until IF NOT EXISTS FOR (e:Entity) ON (e.valid_until)"),
        # Index on source processing time
        ("idx_source_processed", "CREATE INDEX idx_source_processed IF NOT EXISTS FOR (s:Source) ON (s.processed_at)"),
        # Relationship property indexes for temporal edge queries
        ("idx_rel_valid_from", "CREATE INDEX idx_rel_valid_from IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.valid_from)"),
    ]

    print("Creating indexes...")
    with driver.session() as session:
        for name, query in indexes:
            session.run(query)
            print(f"  Created index: {name}")
    print()


def create_vector_index(driver):
    """Create a vector index for entity embeddings (semantic search).

    Neo4j vector index docs:
    https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/
    """
    print("Creating vector index...")
    with driver.session() as session:
        # Drop existing vector index if present
        try:
            session.run("DROP INDEX entity_embedding_index IF EXISTS")
        except Exception:
            pass

        session.run("""
            CREATE VECTOR INDEX entity_embedding_index IF NOT EXISTS
            FOR (e:Entity) ON (e.embedding)
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: 1536,
                    `vector.similarity_function`: 'cosine'
                }
            }
        """)
        print("  Created vector index: entity_embedding_index (1536 dims, cosine)")
    print()


def verify_setup(driver):
    """Verify the schema was created correctly."""
    print("Verifying setup...")
    with driver.session() as session:
        # Count constraints
        result = session.run("SHOW CONSTRAINTS")
        constraints = list(result)
        print(f"  Constraints: {len(constraints)}")

        # Count indexes
        result = session.run("SHOW INDEXES")
        indexes = list(result)
        print(f"  Indexes: {len(indexes)}")

        # Count nodes
        result = session.run("MATCH (n) RETURN count(n) AS count")
        count = result.single()["count"]
        print(f"  Existing nodes: {count}")

    print("\nNeo4j setup complete! Ready for streaming ingestion.")
    print("  Browser UI: http://localhost:7474")
    print("  Bolt URI:   bolt://localhost:7687")


def main():
    parser = argparse.ArgumentParser(description="Initialize Neo4j for Streaming KG")
    parser.add_argument("--clear", action="store_true", help="Clear all existing data before setup")
    args = parser.parse_args()

    print("=" * 60)
    print("Real-Time Streaming KG — Neo4j Setup")
    print("=" * 60 + "\n")

    driver = get_driver()

    try:
        # Test connection
        driver.verify_connectivity()
        print("Connected to Neo4j successfully.\n")

        if args.clear:
            clear_database(driver)

        create_constraints(driver)
        create_indexes(driver)
        create_vector_index(driver)
        verify_setup(driver)

    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure Neo4j is running: docker compose up -d")
        sys.exit(1)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
