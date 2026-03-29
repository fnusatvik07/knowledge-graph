"""Hour 2 — Script 07: Classic RAG vs Graph RAG — Side by Side"""
import sys
from pathlib import Path
from neo4j import GraphDatabase
import chromadb

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent / "projects"))

from shared.llm_clients import chat_completion, get_embedding_batch, get_embedding

# ── Setup Classic RAG ─────────────────────────────────────────────
paper = (HERE / "data" / "ai_agents_paper.txt").read_text()
chunks = [paper[i:i+500] for i in range(0, len(paper), 400)]
embs = get_embedding_batch(chunks)

chroma = chromadb.Client()
col = chroma.get_or_create_collection("cmp")
col.add(ids=[f"c{i}" for i in range(len(chunks))], embeddings=embs, documents=chunks)

# ── Setup Graph RAG ───────────────────────────────────────────────
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "workshop2024"))


def classic_rag(q):
    results = col.query(query_embeddings=[get_embedding(q)], n_results=3)
    ctx = "\n".join(results["documents"][0])
    return chat_completion(f"Context:\n{ctx}\n\nQuestion: {q}", system="Answer ONLY from context.")


def graph_rag(q):
    with driver.session() as s:
        rows = s.run("MATCH (a)-[r]->(b) RETURN a.name+' --['+r.type+']--> '+b.name AS t").values()
    ctx = "\n".join([r[0] for r in rows])
    return chat_completion(f"Graph:\n{ctx}\n\nQuestion: {q}", system="Answer using graph relationships.")


# ── Compare ───────────────────────────────────────────────────────
questions = [
    "What score did LPG achieve?",
    "What is the relationship between BGE-m3 and RAG2?",
    "Which technologies does the RDF pipeline use?",
    "Who are the authors and their organization?",
    "Compare LPG and RDF error sources",
]

print("CLASSIC RAG vs GRAPH RAG\n" + "=" * 70)
for q in questions:
    c = classic_rag(q)[:150]
    g = graph_rag(q)[:150]
    print(f"\nQ: {q}")
    print(f"  Classic: {c}...")
    print(f"  Graph:   {g}...")

driver.close()
