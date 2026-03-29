"""Hour 3 — Script 03: The Power of Cypher Queries"""
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "workshop2024"))


def run_query(title, cypher):
    print(f"\n{'='*60}")
    print(f"QUERY: {title}")
    print(f"CYPHER: {cypher}\n")
    with driver.session() as s:
        for record in s.run(cypher):
            print(f"  {dict(record)}")


# Query 1: Simple — companies in San Francisco
run_query(
    "Companies in San Francisco",
    "MATCH (c:Company)-[:HEADQUARTERED_IN]->(ci:City {name: 'San Francisco'}) RETURN c.name AS company"
)

# Query 2: Aggregation — investors who funded multiple companies
run_query(
    "Investors who funded 2+ companies",
    """MATCH (i:Investor)-[:INVESTED_IN]->(c:Company)
       WITH i.name AS investor, collect(c.name) AS companies, count(c) AS num
       WHERE num >= 2 RETURN investor, companies, num ORDER BY num DESC"""
)

# Query 3: Pattern matching — companies sharing the same technology
run_query(
    "Companies sharing the same technology",
    """MATCH (c1:Company)-[:USES_TECH]->(t:Technology)<-[:USES_TECH]-(c2:Company)
       WHERE c1.name < c2.name
       RETURN c1.name AS company1, c2.name AS company2, t.name AS shared_tech"""
)

# Query 4: Path finding — shortest path between two CEOs
run_query(
    "Shortest path: Sam Altman ↔ Demis Hassabis",
    """MATCH p = shortestPath(
         (a:Person {name: 'Sam Altman'})-[*]-(b:Person {name: 'Demis Hassabis'})
       ) RETURN [n IN nodes(p) | coalesce(n.name, '')] AS path, length(p) AS hops"""
)

# Query 5: Multi-hop — Andreessen Horowitz portfolio + technologies
run_query(
    "A16z portfolio and their technologies",
    """MATCH (i:Investor {name: 'Andreessen Horowitz'})-[:INVESTED_IN]->(c:Company)-[:USES_TECH]->(t:Technology)
       RETURN c.name AS company, t.name AS technology, c.funding AS funding_B"""
)

driver.close()
