"""End-to-end demo for the Real-Time Streaming Knowledge Graph.

Orchestrates the full pipeline:
1. Initialize Neo4j schema (clear + recreate)
2. Start the stream processor watching a demo directory
3. Sequentially copy sample documents into the watch directory with delays
4. Display graph statistics after each document is processed
5. Run temporal queries to show the graph's evolution

Usage:
    python src/06_run_demo.py
    python src/06_run_demo.py --provider openai --delay 5
"""

import sys
import shutil
import time
import argparse
from pathlib import Path

# Shared imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from neo4j import GraphDatabase

# Import local modules
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from importlib import import_module

# ── Neo4j connection ─────────────────────────────────────────────────────
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def print_banner(text: str):
    print(f"\n{'#'*70}")
    print(f"# {text}")
    print(f"{'#'*70}\n")


def print_graph_stats(driver, label: str = "Current"):
    """Print detailed graph statistics."""
    with driver.session() as session:
        entities = session.run("MATCH (e:Entity) RETURN count(e) AS c").single()["c"]
        rels = session.run("MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS c").single()["c"]
        active_rels = session.run(
            "MATCH ()-[r:RELATES_TO]->() WHERE r.invalidated = false RETURN count(r) AS c"
        ).single()["c"]
        invalidated = session.run(
            "MATCH ()-[r:RELATES_TO]->() WHERE r.invalidated = true RETURN count(r) AS c"
        ).single()["c"]
        sources = session.run("MATCH (s:Source) RETURN count(s) AS c").single()["c"]

        # Entity type breakdown
        type_result = session.run("""
            MATCH (e:Entity)
            RETURN e.entity_type AS type, count(e) AS count
            ORDER BY count DESC
        """)
        types = {r["type"]: r["count"] for r in type_result}

        print(f"\n  {label} Graph Statistics:")
        print(f"  {'─'*40}")
        print(f"    Entities:              {entities}")
        print(f"    Relationships (total): {rels}")
        print(f"    Relationships (active):{active_rels}")
        print(f"    Invalidated facts:     {invalidated}")
        print(f"    Documents processed:   {sources}")
        if types:
            print(f"    Entity types:")
            for t, c in types.items():
                print(f"      {t:<20} {c}")
        print()


def run_neo4j_setup(driver):
    """Initialize Neo4j with fresh schema."""
    print_banner("STEP 1: Neo4j Setup")

    # Import and run setup functions directly
    setup = import_module("01_neo4j_setup")
    setup.clear_database(driver)
    setup.create_constraints(driver)
    setup.create_indexes(driver)
    setup.create_vector_index(driver)
    setup.verify_setup(driver)


def process_document_directly(file_path: Path, driver, provider: str, model: str = None):
    """Process a single document using the stream processor module."""
    processor = import_module("02_stream_processor")
    processor.process_document(file_path, driver, provider=provider, model=model)


def run_temporal_queries(driver):
    """Run temporal queries to demonstrate the graph's evolution."""
    temporal = import_module("05_temporal_queries")
    temporal.show_full_graph_summary(driver)
    temporal.query_invalidated_facts(driver)

    # Show timeline for key entities
    for entity in ["TechCorp", "DataStream", "GlobalBank"]:
        temporal.query_entity_timeline(driver, entity)


def main():
    parser = argparse.ArgumentParser(description="End-to-end demo for Streaming KG")
    parser.add_argument("--provider", type=str, default="openai", help="LLM provider")
    parser.add_argument("--model", type=str, default=None, help="LLM model name")
    parser.add_argument("--delay", type=int, default=3, help="Delay between documents (seconds)")
    args = parser.parse_args()

    print("=" * 70)
    print("  REAL-TIME STREAMING KNOWLEDGE GRAPH — DEMO")
    print("=" * 70)
    print(f"\n  Provider: {args.provider}")
    print(f"  Delay between documents: {args.delay}s")

    # Check Neo4j connection
    driver = get_driver()
    try:
        driver.verify_connectivity()
        print("  Neo4j: Connected\n")
    except Exception as e:
        print(f"\n  Error: Cannot connect to Neo4j: {e}")
        print("  Run: docker compose up -d")
        sys.exit(1)

    # Ensure output directory exists
    output_dir = PROJECT_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear the change log
    change_log = output_dir / "change_log.jsonl"
    if change_log.exists():
        change_log.unlink()

    # Sample files
    data_dir = PROJECT_DIR / "data"
    sample_files = sorted(data_dir.glob("stream_sample_*.txt"))

    if not sample_files:
        print("Error: No sample files found in data/ directory.")
        sys.exit(1)

    print(f"  Found {len(sample_files)} sample documents to process.\n")

    try:
        # Step 1: Setup Neo4j
        run_neo4j_setup(driver)

        # Step 2: Process each document sequentially
        print_banner("STEP 2: Processing Documents")

        for i, sample_file in enumerate(sample_files, 1):
            print(f"\n{'~'*60}")
            print(f"  Document {i}/{len(sample_files)}: {sample_file.name}")
            print(f"{'~'*60}")

            process_document_directly(sample_file, driver, provider=args.provider, model=args.model)
            print_graph_stats(driver, f"After Document {i}")

            if i < len(sample_files):
                print(f"  Waiting {args.delay}s before next document...\n")
                time.sleep(args.delay)

        # Step 3: Temporal queries
        print_banner("STEP 3: Temporal Analysis")
        run_temporal_queries(driver)

        # Final summary
        print_banner("DEMO COMPLETE")
        print_graph_stats(driver, "Final")
        print("  The knowledge graph has been built incrementally from streaming data.")
        print("  Temporal metadata tracks when facts were added and invalidated.")
        print("\n  Next steps:")
        print("    - Explore the graph in Neo4j Browser: http://localhost:7474")
        print("    - Run temporal queries: python src/05_temporal_queries.py --entity 'TechCorp'")
        print("    - Start the WebSocket server: python src/03_websocket_server.py")
        print("    - Start the live dashboard:   python src/04_live_dashboard.py")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted.")
    except Exception as e:
        print(f"\nError during demo: {e}")
        raise
    finally:
        driver.close()


if __name__ == "__main__":
    main()
