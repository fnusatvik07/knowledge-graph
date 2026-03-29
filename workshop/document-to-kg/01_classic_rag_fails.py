"""Hour 2 — Script 01: Why Classic RAG Fails"""
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent / "projects"))  # access shared/

from shared.llm_clients import chat_completion, get_embedding_batch, get_embedding
import chromadb

# ── Load the research paper ───────────────────────────────────────
paper = (HERE / "data" / "ai_agents_paper.txt").read_text()
print(f"Loaded paper: {len(paper)} chars\n")

# ── Chunk the text ────────────────────────────────────────────────
chunks = [paper[i:i+500] for i in range(0, len(paper), 400)]
print(f"Created {len(chunks)} chunks (500 chars, 100 overlap)")

# ── Embed and store in ChromaDB ───────────────────────────────────
print("Embedding chunks...")
embeddings = get_embedding_batch(chunks)

chroma = chromadb.Client()
collection = chroma.get_or_create_collection("paper")
collection.add(
    ids=[f"c{i}" for i in range(len(chunks))],
    embeddings=embeddings,
    documents=chunks,
)
print(f"Stored {collection.count()} chunks in ChromaDB\n")

# ── Ask 3 questions — watch RAG struggle ──────────────────────────
questions = [
    ("EASY",      "What score did the LPG approach achieve out of 200?"),
    ("HARD",      "What is the relationship between BGE-m3 and the RAG2 pipeline?"),
    ("VERY HARD", "Compare the performance of all three approaches across all four query intents"),
]

print("=" * 60)
for level, q in questions:
    # Retrieve top-3 chunks
    q_emb = get_embedding(q)
    results = collection.query(query_embeddings=[q_emb], n_results=3)
    context = "\n---\n".join(results["documents"][0])

    # Generate answer from retrieved context
    answer = chat_completion(
        prompt=f"Context:\n{context}\n\nQuestion: {q}",
        system="Answer based ONLY on the context. If info is missing, say so.",
    )

    print(f"\n[{level}] {q}")
    print(f"Answer: {answer[:300]}...")

print("\n" + "=" * 60)
print("Classic RAG struggles with relationships & comparisons.")
print("Let's fix this with a Knowledge Graph!")
