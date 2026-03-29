"""Load the shared KG dataset into FalkorDB using the falkordb Python client.

FalkorDB docs: https://docs.falkordb.com/
FalkorDB Python client: https://github.com/FalkorDB/falkordb-py

FalkorDB is a Redis-based graph database that supports a subset of OpenCypher.
It is optimized for low-latency graph queries.

Prerequisites:
    docker compose up -d falkordb
    pip install falkordb
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from falkordb import FalkorDB

# ── Configuration ────────────────────────────────────────────────────────
FALKOR_HOST = "localhost"
FALKOR_PORT = 6379
GRAPH_NAME = "kg_explorer"

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "entities_and_relations.json"


def load_data() -> dict:
    with open(DATA_FILE) as f:
        return json.load(f)


def clear_graph(graph):
    """Delete all nodes and relationships from the graph."""
    try:
        graph.query("MATCH (n) DETACH DELETE n")
        print("[falkordb] Cleared existing graph data.")
    except Exception:
        print("[falkordb] Graph is empty or new.")


def create_indexes(graph):
    """Create indexes for faster lookups.

    FalkorDB index docs: https://docs.falkordb.com/commands/graph.query.html
    """
    labels = ["Company", "Model", "Architecture", "Technique", "Framework", "Hardware"]
    for label in labels:
        try:
            graph.query(f"CREATE INDEX FOR (n:{label}) ON (n.id)")
        except Exception:
            pass  # Index may already exist
    print(f"[falkordb] Created indexes for {len(labels)} labels.")


def load_entities(graph, entities: list[dict]):
    """Load entities as nodes using OpenCypher CREATE.

    FalkorDB supports parameterized queries via $param syntax.
    Docs: https://docs.falkordb.com/commands/graph.query.html
    """
    for entity in entities:
        label = entity["type"]
        props = {k: v for k, v in entity.items() if k != "type"}

        # Build property string for CREATE
        prop_parts = ", ".join(f"{k}: ${k}" for k in props)
        cypher = f"MERGE (n:{label} {{id: $id}}) SET " + ", ".join(
            f"n.{k} = ${k}" for k in props
        )

        graph.query(cypher, props)

    print(f"[falkordb] Loaded {len(entities)} entities.")


def load_relationships(graph, relationships: list[dict]):
    """Load relationships using OpenCypher MERGE."""
    for rel in relationships:
        rel_type = rel["type"]
        props = {k: v for k, v in rel.items() if k not in ("source", "target", "type")}

        if props:
            set_parts = ", ".join(f"r.{k} = ${k}" for k in props)
            set_clause = f"SET {set_parts}"
        else:
            set_clause = ""

        cypher = f"""
        MATCH (a {{id: $source}})
        MATCH (b {{id: $target}})
        MERGE (a)-[r:{rel_type}]->(b)
        {set_clause}
        """
        params = {"source": rel["source"], "target": rel["target"], **props}
        graph.query(cypher, params)

    print(f"[falkordb] Loaded {len(relationships)} relationships.")


def run_example_queries(graph):
    """Run OpenCypher queries against FalkorDB.

    FalkorDB supports a subset of OpenCypher. Some advanced features
    (APOC, full-text search, etc.) are not available.
    """
    queries = {
        "Total nodes": "MATCH (n) RETURN count(n) AS count",
        "Total relationships": "MATCH ()-[r]->() RETURN count(r) AS count",
        "Companies": "MATCH (c:Company) RETURN c.name ORDER BY c.name",
        "Models and their creators": """
            MATCH (c:Company)-[:DEVELOPED]->(m:Model)
            RETURN c.name AS company, m.name AS model
            ORDER BY c.name
        """,
        "2-hop: Companies sharing architectures": """
            MATCH (c1:Company)-[:DEVELOPED]->(m1:Model)-[:USES_ARCHITECTURE]->(a:Architecture)
                  <-[:USES_ARCHITECTURE]-(m2:Model)<-[:DEVELOPED]-(c2:Company)
            WHERE c1.id < c2.id
            RETURN DISTINCT c1.name AS company1, c2.name AS company2, a.name AS architecture
        """,
    }

    print("\n" + "=" * 60)
    print("FALKORDB EXAMPLE QUERIES (OpenCypher)")
    print("=" * 60)

    for title, query in queries.items():
        result = graph.query(query)
        print(f"\n--- {title} ---")
        for row in result.result_set:
            print(f"  {row}")


def main():
    print("=" * 60)
    print("Loading KG into FalkorDB")
    print("=" * 60)

    data = load_data()

    try:
        db = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
        graph = db.select_graph(GRAPH_NAME)
        print("[falkordb] Connected successfully.")

        clear_graph(graph)
        create_indexes(graph)
        load_entities(graph, data["entities"])
        load_relationships(graph, data["relationships"])
        run_example_queries(graph)

    except Exception as e:
        print(f"\n[falkordb] Error: {e}")
        print("Make sure FalkorDB is running: docker compose up -d falkordb")
        raise

    print("\n[falkordb] Done.")


if __name__ == "__main__":
    main()
