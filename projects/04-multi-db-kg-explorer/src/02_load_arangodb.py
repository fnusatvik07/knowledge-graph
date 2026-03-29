"""Load the shared KG dataset into ArangoDB using python-arango.

python-arango docs: https://docs.python-arango.com/en/main/
AQL reference: https://docs.arangodb.com/stable/aql/

Prerequisites:
    docker compose up -d arangodb
    pip install python-arango
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from arango import ArangoClient

# ── Configuration ────────────────────────────────────────────────────────
ARANGO_HOST = "http://localhost:8529"
ARANGO_USER = "root"
ARANGO_PASSWORD = "password123"
DB_NAME = "kg_explorer"

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "entities_and_relations.json"


def load_data() -> dict:
    with open(DATA_FILE) as f:
        return json.load(f)


def setup_database(client) -> "StandardDatabase":
    """Create the database if it doesn't exist.

    ArangoDB requires creating databases explicitly before use.
    Docs: https://docs.python-arango.com/en/main/database.html
    """
    sys_db = client.db("_system", username=ARANGO_USER, password=ARANGO_PASSWORD)

    if not sys_db.has_database(DB_NAME):
        sys_db.create_database(DB_NAME)
        print(f"[arango] Created database '{DB_NAME}'.")
    else:
        print(f"[arango] Database '{DB_NAME}' already exists.")

    return client.db(DB_NAME, username=ARANGO_USER, password=ARANGO_PASSWORD)


def setup_collections(db):
    """Create vertex and edge collections.

    ArangoDB is multi-model: vertex collections store nodes,
    edge collections store relationships with _from/_to references.
    Docs: https://docs.python-arango.com/en/main/collection.html
    """
    # Vertex collections (one per entity type)
    vertex_types = ["Company", "Model", "Architecture", "Technique", "Framework", "Hardware"]
    for vtype in vertex_types:
        if not db.has_collection(vtype):
            db.create_collection(vtype)
            print(f"[arango] Created vertex collection: {vtype}")

    # Edge collection
    if not db.has_collection("relationships"):
        db.create_collection("relationships", edge=True)
        print("[arango] Created edge collection: relationships")


def clear_collections(db):
    """Truncate all collections to start fresh."""
    for name in db.collections():
        if not name["system"]:
            db.collection(name["name"]).truncate()
    print("[arango] Cleared all collections.")


def load_entities(db, entities: list[dict]):
    """Insert entities into their respective vertex collections.

    In ArangoDB, each document needs a _key for unique identification.
    Docs: https://docs.python-arango.com/en/main/document.html
    """
    for entity in entities:
        collection_name = entity["type"]
        doc = {k: v for k, v in entity.items() if k != "type"}
        doc["_key"] = doc.pop("id")  # ArangoDB uses _key for unique ID

        collection = db.collection(collection_name)
        if not collection.has(doc["_key"]):
            collection.insert(doc)

    print(f"[arango] Loaded {len(entities)} entities.")


def load_relationships(db, relationships: list[dict], entities: list[dict]):
    """Insert relationships into the edge collection.

    Edge documents require _from and _to in the format "CollectionName/_key".
    Docs: https://docs.python-arango.com/en/main/graph.html
    """
    # Build a lookup: entity id -> "CollectionName/id"
    id_to_collection = {e["id"]: e["type"] for e in entities}

    edge_col = db.collection("relationships")
    for rel in relationships:
        src_col = id_to_collection.get(rel["source"], "Company")
        tgt_col = id_to_collection.get(rel["target"], "Company")

        edge_doc = {
            "_from": f"{src_col}/{rel['source']}",
            "_to": f"{tgt_col}/{rel['target']}",
            "type": rel["type"],
        }
        # Copy extra properties
        for k, v in rel.items():
            if k not in ("source", "target", "type"):
                edge_doc[k] = v

        edge_col.insert(edge_doc)

    print(f"[arango] Loaded {len(relationships)} relationships.")


def run_example_queries(db):
    """Run AQL queries to verify the data.

    AQL (ArangoDB Query Language) is SQL-like but graph-aware.
    Docs: https://docs.arangodb.com/stable/aql/
    """
    queries = {
        "Total entities": "FOR doc IN Company COLLECT WITH COUNT INTO c RETURN c",
        "All companies": "FOR c IN Company SORT c.name RETURN c.name",
        "Models developed by companies": """
            FOR e IN relationships
                FILTER e.type == "DEVELOPED"
                LET company = DOCUMENT(e._from)
                LET model = DOCUMENT(e._to)
                SORT company.name
                RETURN {company: company.name, model: model.name}
        """,
        "2-hop: Traverse from OpenAI (depth 1-2)": """
            FOR v, e, p IN 1..2 OUTBOUND 'Company/openai' relationships
                RETURN {
                    path_length: LENGTH(p.edges),
                    entity: v.name,
                    via: e.type
                }
        """,
        "Aggregation: Models per company": """
            FOR e IN relationships
                FILTER e.type == "DEVELOPED"
                LET company = DOCUMENT(e._from)
                COLLECT comp = company.name WITH COUNT INTO model_count
                SORT model_count DESC
                RETURN {company: comp, models: model_count}
        """,
    }

    print("\n" + "=" * 60)
    print("ARANGODB EXAMPLE QUERIES (AQL)")
    print("=" * 60)

    for title, aql in queries.items():
        cursor = db.aql.execute(aql)
        results = list(cursor)
        print(f"\n--- {title} ---")
        for r in results:
            print(f"  {r}")


def main():
    print("=" * 60)
    print("Loading KG into ArangoDB")
    print("=" * 60)

    data = load_data()
    client = ArangoClient(hosts=ARANGO_HOST)

    db = setup_database(client)
    setup_collections(db)
    clear_collections(db)
    load_entities(db, data["entities"])
    load_relationships(db, data["relationships"], data["entities"])
    run_example_queries(db)

    print("\n[arango] Done.")


if __name__ == "__main__":
    main()
