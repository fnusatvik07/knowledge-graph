"""Hour 2 — Script 04: Load the Knowledge Graph into Neo4j"""
import json
from pathlib import Path
from neo4j import GraphDatabase

HERE = Path(__file__).parent

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "workshop2024"))
print("Connected to Neo4j\n")

# Clear old data
with driver.session() as s:
    s.run("MATCH (n) DETACH DELETE n")
print("Cleared existing data")

# Load extracted data
entities = json.loads((HERE / "output" / "entities.json").read_text())
relations = json.loads((HERE / "output" / "relations.json").read_text())

# Create nodes
print(f"\nCreating {len(entities)} nodes...")
with driver.session() as s:
    for e in entities:
        s.run(
            "CREATE (n:Entity {name: $name, type: $type, description: $desc})",
            name=e["name"], type=e["type"], desc=e["description"],
        )

# Create relationships
print(f"Creating {len(relations)} relationships...")
with driver.session() as s:
    for r in relations:
        s.run(
            """MATCH (a:Entity {name: $src}), (b:Entity {name: $tgt})
               CREATE (a)-[:RELATES_TO {type: $type, description: $desc}]->(b)""",
            src=r["source"], tgt=r["target"], type=r["type"], desc=r["description"],
        )

# Stats
with driver.session() as s:
    nodes = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    edges = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]

print(f"\nKnowledge Graph loaded: {nodes} nodes, {edges} edges")
print(f"\nOpen http://localhost:7474 and run:")
print(f"  MATCH (n)-[r]->(m) RETURN n, r, m")
driver.close()
