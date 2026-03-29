"""Load the shared KG dataset into Memgraph using gqlalchemy (Memgraph's Python OGM).

gqlalchemy docs: https://memgraph.com/docs/client-libraries/python
Memgraph Cypher: https://memgraph.com/docs/querying/clauses

Memgraph is Cypher-compatible, so queries look very similar to Neo4j.
Key differences: no APOC, different indexing syntax, streaming-optimized.

Prerequisites:
    docker compose up -d memgraph
    pip install gqlalchemy
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from gqlalchemy import Memgraph

# ── Configuration ────────────────────────────────────────────────────────
MEMGRAPH_HOST = "localhost"
MEMGRAPH_PORT = 7688  # Remapped to avoid Neo4j conflict

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "entities_and_relations.json"


def load_data() -> dict:
    with open(DATA_FILE) as f:
        return json.load(f)


def clear_database(mg: Memgraph):
    """Remove all data from Memgraph."""
    mg.execute("MATCH (n) DETACH DELETE n")
    print("[memgraph] Cleared all existing data.")


def create_indexes(mg: Memgraph):
    """Create label indexes for faster lookups.

    Memgraph index docs:
    https://memgraph.com/docs/fundamentals/indexes
    """
    labels = ["Company", "Model", "Architecture", "Technique", "Framework", "Hardware"]
    for label in labels:
        try:
            mg.execute(f"CREATE INDEX ON :{label}(id)")
        except Exception:
            pass  # Index may already exist
    print(f"[memgraph] Created indexes for {len(labels)} labels.")


def load_entities(mg: Memgraph, entities: list[dict]):
    """Load entities using Cypher MERGE.

    gqlalchemy supports raw Cypher execution via mg.execute().
    Docs: https://memgraph.com/docs/client-libraries/python#execute-cypher-queries
    """
    for entity in entities:
        label = entity["type"]
        props = {k: v for k, v in entity.items() if k != "type"}

        # Build SET clause from properties
        set_parts = ", ".join(f"n.{k} = ${k}" for k in props)
        cypher = f"MERGE (n:{label} {{id: $id}}) SET {set_parts}"

        mg.execute(cypher, props)

    print(f"[memgraph] Loaded {len(entities)} entities.")


def load_relationships(mg: Memgraph, relationships: list[dict]):
    """Load relationships using Cypher MERGE."""
    for rel in relationships:
        rel_type = rel["type"]
        props = {k: v for k, v in rel.items() if k not in ("source", "target", "type")}

        # Build property SET clause for the relationship
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
        mg.execute(cypher, params)

    print(f"[memgraph] Loaded {len(relationships)} relationships.")


def run_example_queries(mg: Memgraph):
    """Run Cypher queries against Memgraph.

    Memgraph supports standard Cypher with some extensions.
    Note: APOC procedures are NOT available in Memgraph.
    """
    queries = {
        "Total nodes": "MATCH (n) RETURN count(n) AS count",
        "Total relationships": "MATCH ()-[r]->() RETURN count(r) AS count",
        "Companies": "MATCH (c:Company) RETURN c.name AS name ORDER BY c.name",
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
    print("MEMGRAPH EXAMPLE QUERIES (Cypher)")
    print("=" * 60)

    for title, query in queries.items():
        results = list(mg.execute_and_fetch(query))
        print(f"\n--- {title} ---")
        for r in results:
            print(f"  {dict(r)}")


def main():
    print("=" * 60)
    print("Loading KG into Memgraph")
    print("=" * 60)

    data = load_data()
    mg = Memgraph(host=MEMGRAPH_HOST, port=MEMGRAPH_PORT)

    try:
        clear_database(mg)
        create_indexes(mg)
        load_entities(mg, data["entities"])
        load_relationships(mg, data["relationships"])
        run_example_queries(mg)
    except Exception as e:
        print(f"\n[memgraph] Error: {e}")
        print("Make sure Memgraph is running: docker compose up -d memgraph")
        raise

    print("\n[memgraph] Done.")


if __name__ == "__main__":
    main()
