"""Step 7: Traditional vector RAG baseline using ChromaDB.

Builds a standard vector RAG pipeline for the same corpus:
1. Chunk documents
2. Generate embeddings via LangChain Embeddings
3. Store in ChromaDB
4. Retrieve relevant chunks for queries
5. Generate answers using retrieved context

This serves as the baseline for comparison against graph RAG approaches.
"""
# LangChain docs: https://python.langchain.com/docs/integrations/chat/

import argparse
import json
import sys
from pathlib import Path

# Add projects dir to path for shared imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared.llm_clients import chat_completion, get_embedding, get_embedding_batch
from shared.document_loader import load_text_files, chunk_text

PROJECT_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_DIR / "data" / "corpus"
OUTPUT_DIR = PROJECT_DIR / "output"
CHROMA_DIR = OUTPUT_DIR / "chromadb"

# Same questions used across all search methods for fair comparison
TEST_QUESTIONS = [
    "What are the main strategies for reducing global carbon emissions?",
    "How do battery technologies connect the renewable energy and electric vehicle sectors?",
    "What is the role of Tesla in the electric vehicle industry?",
    "How does the Inflation Reduction Act support clean energy?",
    "How do agriculture and energy systems interact in the context of climate solutions?",
]


def check_chromadb_installed():
    """Verify ChromaDB is available."""
    try:
        import chromadb  # noqa: F401
        return True
    except ImportError:
        print("=" * 60)
        print("ERROR: ChromaDB is not installed.")
        print()
        print("Install it with:")
        print("  pip install chromadb")
        print("=" * 60)
        return False


def build_vector_index(provider: str = "openai"):
    """Load corpus, chunk documents, embed, and store in ChromaDB.

    Uses get_embedding_batch() from the shared LLM layer for batch embeddings.
    LangChain embeddings docs: https://python.langchain.com/docs/integrations/text_embedding/
    """
    import chromadb

    # Load documents
    documents = load_text_files(CORPUS_DIR)
    if not documents:
        print(f"No documents found in {CORPUS_DIR}")
        sys.exit(1)

    print(f"Loaded {len(documents)} documents")

    # Chunk documents
    all_chunks = []
    all_metadatas = []
    all_ids = []

    for doc in documents:
        chunks = chunk_text(doc["content"], chunk_size=800, overlap=150)
        print(f"  {doc['title']}: {len(chunks)} chunks")

        for j, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append({
                "source": doc["title"],
                "source_file": doc["source"],
                "chunk_index": j,
            })
            all_ids.append(f"{doc['title']}_{j}")

    print(f"\nTotal chunks: {len(all_chunks)}")

    # Create ChromaDB collection
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Delete existing collection if it exists
    try:
        client.delete_collection("corpus")
    except Exception:
        pass

    collection = client.create_collection(
        name="corpus",
        metadata={"hnsw:space": "cosine"},
    )

    # Generate embeddings and add to collection in batches
    print("\nGenerating embeddings and storing in ChromaDB...")
    batch_size = 20

    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i : i + batch_size]
        batch_ids = all_ids[i : i + batch_size]
        batch_metadatas = all_metadatas[i : i + batch_size]

        # Get embeddings for the batch via shared LLM layer
        embeddings = get_embedding_batch(batch_chunks, provider=provider)

        collection.add(
            documents=batch_chunks,
            embeddings=embeddings,
            metadatas=batch_metadatas,
            ids=batch_ids,
        )
        print(f"  Added batch {i // batch_size + 1}/{(len(all_chunks) + batch_size - 1) // batch_size}")

    print(f"\nVector index built: {collection.count()} chunks stored")
    return client, collection


def retrieve_and_answer(collection, question: str, top_k: int = 5, provider: str = "openai") -> dict:
    """Retrieve relevant chunks and generate an answer.

    Returns a dict with the question, retrieved chunks, and answer.
    """
    # Get query embedding
    query_embedding = get_embedding(question, provider=provider)

    # Retrieve top-k chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved_chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Build context from retrieved chunks
    context_parts = []
    for chunk, meta, dist in zip(retrieved_chunks, metadatas, distances):
        similarity = 1 - dist  # cosine distance to similarity
        context_parts.append(
            f"[Source: {meta['source']}, Similarity: {similarity:.3f}]\n{chunk}"
        )

    context = "\n\n---\n\n".join(context_parts)

    # Generate answer
    answer = chat_completion(
        prompt=f"""Answer the following question based ONLY on the provided context.
If the context does not contain enough information, say so.

Context:
{context}

Question: {question}

Answer:""",
        system="You are a helpful research assistant. Answer questions accurately based "
               "on the provided context. Cite specific facts and be comprehensive.",
        provider=provider,
    )

    return {
        "question": question,
        "answer": answer,
        "num_chunks_retrieved": len(retrieved_chunks),
        "sources": [m["source"] for m in metadatas],
        "avg_similarity": sum(1 - d for d in distances) / len(distances),
        "context": context,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vector RAG baseline")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "ollama"],
        default="openai",
        help="LLM/embedding provider to use (default: openai)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    provider = args.provider

    print("=" * 60)
    print("Vector RAG Baseline (ChromaDB)")
    print("=" * 60)
    print()
    print(f"Using LLM provider: {provider}")
    print("Building a traditional vector RAG pipeline as baseline for")
    print("comparison against GraphRAG and LightRAG approaches.")
    print()

    if not check_chromadb_installed():
        sys.exit(1)

    # Build the vector index
    print("\n--- Building Vector Index ---")
    client, collection = build_vector_index(provider=provider)

    # Run test queries
    print("\n\n--- Running Test Queries ---")
    all_results = []

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n{'='*60}")
        print(f"Question {i}: {question}")
        print("-" * 60)

        result = retrieve_and_answer(collection, question, provider=provider)

        print(f"\nRetrieved {result['num_chunks_retrieved']} chunks from: {set(result['sources'])}")
        print(f"Average similarity: {result['avg_similarity']:.3f}")
        print(f"\nAnswer:\n{result['answer']}")

        all_results.append(result)

    # Save results for comparison script
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "vector_rag_results.json"
    save_results = [
        {
            "question": r["question"],
            "answer": r["answer"],
            "sources": r["sources"],
            "avg_similarity": r["avg_similarity"],
        }
        for r in all_results
    ]
    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\n\nResults saved to: {output_path}")

    # Summary
    print(f"\n{'='*60}")
    print("Vector RAG Baseline Summary")
    print("=" * 60)
    print(f"Total chunks in index: {collection.count()}")
    print(f"Questions answered: {len(all_results)}")
    avg_sim = sum(r["avg_similarity"] for r in all_results) / len(all_results)
    print(f"Average retrieval similarity: {avg_sim:.3f}")
    print(f"\nVector RAG characteristics:")
    print("  + Fast retrieval, simple architecture")
    print("  + Good for specific factual questions")
    print("  - Misses cross-document relationships")
    print("  - No understanding of entity connections")
    print("  - Limited on broad thematic questions")


if __name__ == "__main__":
    main()
