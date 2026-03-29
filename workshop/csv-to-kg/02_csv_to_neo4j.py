"""Hour 3 — Script 02: Transform CSV into a Knowledge Graph in Neo4j"""
from pathlib import Path
import pandas as pd
from neo4j import GraphDatabase

HERE = Path(__file__).parent

df = pd.read_csv(HERE / "data" / "ai_companies.csv")
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "workshop2024"))

# Clear old data
with driver.session() as s:
    s.run("MATCH (n) DETACH DELETE n")
print("Cleared Neo4j\n")

# One flat CSV row → multiple nodes + relationships
print(f"Loading {len(df)} companies into Neo4j...\n")
with driver.session() as s:
    for _, row in df.iterrows():
        # Company node
        s.run("MERGE (c:Company {name: $n}) SET c.founded=$y, c.funding=$f, c.employees=$e",
              n=row["company"], y=int(row["founded_year"]),
              f=float(row["funding_total_b"]), e=int(row["num_employees"]))

        # CEO → LEADS → Company
        s.run("MERGE (p:Person {name: $ceo}) MERGE (c:Company {name: $co}) MERGE (p)-[:LEADS]->(c)",
              ceo=row["ceo"], co=row["company"])

        # Company → HEADQUARTERED_IN → City
        s.run("MERGE (ci:City {name: $city}) MERGE (c:Company {name: $co}) MERGE (c)-[:HEADQUARTERED_IN]->(ci)",
              city=row["headquarters"], co=row["company"])

        # Company → PRODUCES → Product
        s.run("MERGE (pr:Product {name: $prod}) MERGE (c:Company {name: $co}) MERGE (c)-[:PRODUCES]->(pr)",
              prod=row["flagship_product"], co=row["company"])

        # Company → IN_CATEGORY → Category
        s.run("MERGE (cat:Category {name: $cat}) MERGE (c:Company {name: $co}) MERGE (c)-[:IN_CATEGORY]->(cat)",
              cat=row["category"], co=row["company"])

        # Company → USES_TECH → Technology
        s.run("MERGE (t:Technology {name: $tech}) MERGE (c:Company {name: $co}) MERGE (c)-[:USES_TECH]->(t)",
              tech=row["key_technology"], co=row["company"])

        # Investors (semicolon-separated) → INVESTED_IN → Company
        for inv in str(row["investors"]).split(";"):
            inv = inv.strip()
            if inv:
                s.run("MERGE (i:Investor {name: $inv}) MERGE (c:Company {name: $co}) MERGE (i)-[:INVESTED_IN]->(c)",
                      inv=inv, co=row["company"])

        print(f"  ✓ {row['company']}")

# Stats
with driver.session() as s:
    nodes = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    edges = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    labels = s.run("CALL db.labels() YIELD label RETURN collect(label) AS l").single()["l"]

print(f"\nGraph loaded: {nodes} nodes, {edges} relationships")
print(f"Node types: {', '.join(labels)}")
print(f"\nOpen http://localhost:7474 and run:")
print(f"  MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50")
driver.close()
