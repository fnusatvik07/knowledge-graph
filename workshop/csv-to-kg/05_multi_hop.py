"""Hour 3 — Script 05: Multi-Hop Queries & Recommendations"""
import sys
from pathlib import Path
from neo4j import GraphDatabase

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent.parent / "projects"))

from shared.llm_clients import chat_completion

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "workshop2024"))


def run_and_explain(title, cypher, question):
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"Cypher: {cypher[:100]}...\n")
    with driver.session() as s:
        rows = s.run(cypher).values()
    context = "\n".join([str(r) for r in rows])
    print(f"Raw results: {len(rows)} rows")
    answer = chat_completion(
        prompt=f"Query results:\n{context}\n\nQuestion: {question}",
        system="Explain the results clearly. Use the data to support your answer.",
    )
    print(f"\n{answer[:400]}")


# 1. Recommend companies similar to OpenAI
run_and_explain(
    "RECOMMEND: Companies similar to OpenAI",
    """MATCH (openai:Company {name: 'OpenAI'})-[:USES_TECH]->(t)<-[:USES_TECH]-(other:Company)
       WHERE other.name <> 'OpenAI'
       WITH other, collect(t.name) AS shared_tech, count(t) AS score
       RETURN other.name AS company, shared_tech, score ORDER BY score DESC LIMIT 5""",
    "Which companies are most similar to OpenAI and why?"
)

# 2. Shortest path in the investment network
run_and_explain(
    "PATH: ChromaDB to Neo4j via investors",
    """MATCH p = shortestPath(
         (a:Company {name: 'ChromaDB'})-[*]-(b:Company {name: 'Neo4j'})
       ) RETURN [n IN nodes(p) | n.name] AS path, length(p) AS hops""",
    "How are ChromaDB and Neo4j connected in the investment network?"
)

# 3. AI capital city by total funding
run_and_explain(
    "AGGREGATE: AI capital by total funding",
    """MATCH (c:Company)-[:HEADQUARTERED_IN]->(ci:City)
       WITH ci.name AS city, sum(c.funding) AS total_funding, count(c) AS num_companies,
            collect(c.name) AS companies
       RETURN city, round(total_funding, 2) AS total_B, num_companies, companies
       ORDER BY total_funding DESC""",
    "Which city is the AI capital and why?"
)

driver.close()
print("\n\nDone! You've built Knowledge Graphs from both documents AND structured data.")
