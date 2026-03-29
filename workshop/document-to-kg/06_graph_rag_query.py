"""Hour 2 — Script 06: Graph RAG — Query the Knowledge Graph"""
import sys
from pathlib import Path
from neo4j import GraphDatabase

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent / "projects"))

from shared.llm_clients import chat_completion

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "workshop2024"))


def graph_rag(question: str) -> str:
    """Graph RAG: extract entities from question → Cypher → LLM answer."""
    # Step 1: LLM identifies key entities in the question
    entities_str = chat_completion(
        prompt=f"List the key entity names from this question, one per line:\n{question}",
        system="Return ONLY entity names, one per line.",
    )
    terms = [e.strip() for e in entities_str.strip().split("\n") if e.strip()]

    # Step 2: Fetch relevant subgraph from Neo4j
    triples = []
    with driver.session() as s:
        for term in terms[:5]:
            for rec in s.run(
                """MATCH (a:Entity)-[r]->(b:Entity)
                   WHERE toLower(a.name) CONTAINS toLower($t)
                      OR toLower(b.name) CONTAINS toLower($t)
                   RETURN a.name AS s, r.type AS r, b.name AS t, a.description AS d
                   LIMIT 15""", t=term,
            ):
                triples.append(f"{rec['s']} --[{rec['r']}]--> {rec['t']}")
                if rec["d"]:
                    triples.append(f"  ({rec['s']}: {rec['d']})")

    context = "\n".join(triples) if triples else "No graph data found."

    # Step 3: LLM answers using graph context
    return chat_completion(
        prompt=f"Knowledge Graph:\n{context}\n\nQuestion: {question}",
        system="Answer using the knowledge graph relationships.",
    )


# ── Same 3 questions from Script 01 ──────────────────────────────
questions = [
    ("EASY",      "What score did the LPG approach achieve out of 200?"),
    ("HARD",      "What is the relationship between BGE-m3 and the RAG2 pipeline?"),
    ("VERY HARD", "Compare all three approaches"),
]

print("GRAPH RAG — Querying the Knowledge Graph\n" + "=" * 50)
for level, q in questions:
    answer = graph_rag(q)
    print(f"\n[{level}] {q}")
    print(f"Answer: {answer[:400]}")

driver.close()
print("\n\nGraph RAG handles relationship questions that Classic RAG missed!")
