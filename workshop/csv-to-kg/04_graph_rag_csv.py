"""Hour 3 — Script 04: Graph RAG Over the CSV Knowledge Graph"""
import sys
from pathlib import Path
from neo4j import GraphDatabase

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent.parent / "projects"))

from shared.llm_clients import chat_completion

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "workshop2024"))


def graph_rag(question: str) -> str:
    """Query the CSV-derived knowledge graph and generate an answer."""
    # Fetch the full graph as context (small enough for our dataset)
    with driver.session() as s:
        rows = s.run("""
            MATCH (a)-[r]->(b)
            RETURN labels(a)[0] + ': ' + a.name + ' --[' + type(r) + ']--> ' +
                   labels(b)[0] + ': ' + b.name AS triple
            LIMIT 200
        """).values()
    context = "\n".join([r[0] for r in rows])

    return chat_completion(
        prompt=f"Knowledge Graph:\n{context}\n\nQuestion: {question}",
        system="Answer using the knowledge graph. Trace the relationships to explain your answer.",
    )


# Questions that a simple CSV lookup CAN'T answer easily
questions = [
    "Which investor has the most diverse portfolio across different AI categories?",
    "What technologies are common between companies funded by the same investor?",
    "If I want to invest in AI coding tools, which investors already have experience in that space?",
    "Which CEO leads a company that shares technology with the most other companies?",
]

print("GRAPH RAG OVER CSV DATA\n" + "=" * 60)
for q in questions:
    answer = graph_rag(q)
    print(f"\nQ: {q}")
    print(f"A: {answer[:300]}\n")

driver.close()
