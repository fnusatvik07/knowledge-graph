"""Evaluate a traditional vector RAG system using ChromaDB.

Builds a vector RAG from the gold standard documents, runs evaluation questions,
and scores answers using RAGAS-inspired metrics (LLM-as-judge):
- Faithfulness: Is the answer grounded in the retrieved context?
- Answer Relevancy: Does the answer address the question?
- Context Precision: Is the retrieved context relevant to the question?
- Context Recall: Does the context contain the information needed?

ChromaDB docs: https://docs.trychroma.com/
LangChain ChromaDB: https://python.langchain.com/docs/integrations/vectorstores/chroma/
LangChain RAG: https://python.langchain.com/docs/tutorials/rag/

Usage:
    python src/03_vector_rag_eval.py
    python src/03_vector_rag_eval.py --provider openai --top-k 3
"""

import sys
import json
import argparse
from pathlib import Path

# Shared imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pydantic import BaseModel, Field
from shared.llm_clients import (
    chat_completion,
    chat_completion_structured,
    get_embedding,
    get_embedding_batch,
)

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
GOLD_STANDARD_PATH = PROJECT_DIR / "data" / "gold_standard.json"
EVAL_QUESTIONS_PATH = PROJECT_DIR / "data" / "eval_questions.json"
OUTPUT_DIR = PROJECT_DIR / "output"


# ── Pydantic schemas for evaluation scoring ──────────────────────────────

class RAGScore(BaseModel):
    """Scores for a single RAG evaluation metric."""
    score: float = Field(description="Score from 0.0 to 1.0")
    reasoning: str = Field(description="Brief explanation for the score")


class RAGEvaluation(BaseModel):
    """Complete evaluation of a single RAG answer."""
    faithfulness: RAGScore = Field(description="Is the answer grounded in the context?")
    answer_relevancy: RAGScore = Field(description="Does the answer address the question?")
    context_precision: RAGScore = Field(description="Is the retrieved context relevant?")
    context_recall: RAGScore = Field(description="Does the context have needed information?")


# ── Vector RAG system ────────────────────────────────────────────────────

class VectorRAG:
    """Simple vector RAG using ChromaDB for retrieval."""

    def __init__(self, provider: str = "openai", model: str = None):
        self.provider = provider
        self.model = model
        self.documents: list[dict] = []
        self.chunks: list[dict] = []

        import chromadb
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name="eval_docs",
            metadata={"hnsw:space": "cosine"},
        )

    def load_documents(self, gold_standard: dict):
        """Load gold standard documents and index them."""
        self.documents = gold_standard["documents"]

        # Chunk documents (simple paragraph-based splitting)
        chunks = []
        chunk_ids = []
        chunk_embeddings = []

        for doc in self.documents:
            text = doc["text"]
            # Split into sentences for finer-grained retrieval
            sentences = [s.strip() for s in text.split(". ") if s.strip()]

            # Group sentences into chunks of ~2-3 sentences
            for i in range(0, len(sentences), 2):
                chunk_text = ". ".join(sentences[i:i+2]) + "."
                chunk_id = f"{doc['doc_id']}_chunk_{i//2}"
                chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                })
                chunk_ids.append(chunk_id)

        # Get embeddings in batch
        chunk_texts = [c["text"] for c in chunks]
        embeddings = get_embedding_batch(chunk_texts, provider=self.provider)

        # Add to ChromaDB
        self.collection.add(
            ids=chunk_ids,
            documents=chunk_texts,
            embeddings=embeddings,
            metadatas=[{"doc_id": c["doc_id"], "title": c["title"]} for c in chunks],
        )

        self.chunks = chunks
        print(f"  Indexed {len(chunks)} chunks from {len(self.documents)} documents")

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Retrieve relevant chunks for a query."""
        query_embedding = get_embedding(query, provider=self.provider)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        retrieved = []
        for i in range(len(results["ids"][0])):
            retrieved.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else None,
                "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
            })
        return retrieved

    def answer(self, question: str, top_k: int = 3) -> dict:
        """Answer a question using vector RAG."""
        # Retrieve context
        retrieved = self.retrieve(question, top_k=top_k)
        context = "\n\n".join([r["text"] for r in retrieved])

        # Generate answer
        prompt = f"""Answer the following question based ONLY on the provided context.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""

        answer = chat_completion(
            prompt=prompt,
            system="You are a helpful assistant. Answer questions based only on the provided context. "
                   "Be thorough but concise.",
            provider=self.provider,
            model=self.model,
        )

        return {
            "question": question,
            "answer": answer,
            "context": context,
            "retrieved_chunks": retrieved,
        }


# ── Evaluation functions ─────────────────────────────────────────────────

def evaluate_answer(
    question: str,
    answer: str,
    context: str,
    reference_answer: str,
    provider: str = "openai",
    model: str = None,
) -> RAGEvaluation:
    """Evaluate a RAG answer using LLM-as-judge (RAGAS-inspired metrics).

    RAGAS framework: https://docs.ragas.io/en/latest/concepts/metrics/
    """
    prompt = f"""Evaluate the quality of this RAG (Retrieval-Augmented Generation) answer.

Question: {question}

Retrieved Context: {context}

Generated Answer: {answer}

Reference Answer: {reference_answer}

Score each metric from 0.0 to 1.0 with a brief reasoning:

1. Faithfulness: Is the generated answer factually grounded in the retrieved context?
   (1.0 = fully grounded, 0.0 = contains hallucinations)

2. Answer Relevancy: Does the answer directly address the question asked?
   (1.0 = perfectly relevant, 0.0 = completely off-topic)

3. Context Precision: Is the retrieved context relevant to answering the question?
   (1.0 = all context is relevant, 0.0 = none is relevant)

4. Context Recall: Does the retrieved context contain all the information needed
   to answer the question (compared to the reference answer)?
   (1.0 = all needed info present, 0.0 = none present)"""

    result = chat_completion_structured(
        prompt=prompt,
        output_schema=RAGEvaluation,
        system="You are an expert evaluator for RAG systems. Score precisely and explain your reasoning.",
        provider=provider,
        model=model,
    )
    return result


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Vector RAG Evaluation")
    parser.add_argument("--provider", type=str, default="openai", help="LLM provider")
    parser.add_argument("--model", type=str, default=None, help="LLM model name")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve")
    args = parser.parse_args()

    print("=" * 70)
    print("  KG Evaluation: Vector RAG (ChromaDB)")
    print("=" * 70)
    print(f"\n  Provider: {args.provider}")
    print(f"  Top-K: {args.top_k}\n")

    # Load data
    with open(GOLD_STANDARD_PATH) as f:
        gold_standard = json.load(f)
    with open(EVAL_QUESTIONS_PATH) as f:
        eval_data = json.load(f)

    questions = eval_data["questions"]

    # Build vector RAG
    print("  Building Vector RAG...")
    rag = VectorRAG(provider=args.provider, model=args.model)
    rag.load_documents(gold_standard)

    # Run evaluation
    results = []
    type_scores = {}

    for i, q in enumerate(questions, 1):
        print(f"\n  [{i}/{len(questions)}] {q['question'][:60]}...")

        # Get RAG answer
        rag_result = rag.answer(q["question"], top_k=args.top_k)

        # Evaluate
        evaluation = evaluate_answer(
            question=q["question"],
            answer=rag_result["answer"],
            context=rag_result["context"],
            reference_answer=q["reference_answer"],
            provider=args.provider,
            model=args.model,
        )

        scores = {
            "faithfulness": evaluation.faithfulness.score,
            "answer_relevancy": evaluation.answer_relevancy.score,
            "context_precision": evaluation.context_precision.score,
            "context_recall": evaluation.context_recall.score,
        }

        print(f"    Faith: {scores['faithfulness']:.2f}  "
              f"Relev: {scores['answer_relevancy']:.2f}  "
              f"Prec: {scores['context_precision']:.2f}  "
              f"Recall: {scores['context_recall']:.2f}")

        result_entry = {
            "question_id": q["id"],
            "question_type": q["type"],
            "question": q["question"],
            "answer": rag_result["answer"],
            "reference_answer": q["reference_answer"],
            "scores": scores,
            "evaluation_details": {
                "faithfulness_reasoning": evaluation.faithfulness.reasoning,
                "relevancy_reasoning": evaluation.answer_relevancy.reasoning,
                "precision_reasoning": evaluation.context_precision.reasoning,
                "recall_reasoning": evaluation.context_recall.reasoning,
            },
        }
        results.append(result_entry)

        # Aggregate by type
        q_type = q["type"]
        if q_type not in type_scores:
            type_scores[q_type] = []
        type_scores[q_type].append(scores)

    # Print summary
    print(f"\n\n{'='*70}")
    print("  VECTOR RAG EVALUATION SUMMARY")
    print(f"{'='*70}")

    # Overall averages
    all_scores = [r["scores"] for r in results]
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    print(f"\n  Overall Averages:")
    overall_avg = {}
    for m in metrics:
        avg = sum(s[m] for s in all_scores) / len(all_scores)
        overall_avg[m] = round(avg, 4)
        print(f"    {m:<22} {avg:.4f}")

    # Per-type averages
    print(f"\n  Per Question Type:")
    type_averages = {}
    for q_type in sorted(type_scores.keys()):
        scores_list = type_scores[q_type]
        type_avg = {}
        print(f"\n    {q_type} ({len(scores_list)} questions):")
        for m in metrics:
            avg = sum(s[m] for s in scores_list) / len(scores_list)
            type_avg[m] = round(avg, 4)
            print(f"      {m:<22} {avg:.4f}")
        type_averages[q_type] = type_avg

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "system": "vector_rag",
        "provider": args.provider,
        "top_k": args.top_k,
        "overall_averages": overall_avg,
        "per_type_averages": type_averages,
        "per_question_results": results,
    }
    output_path = OUTPUT_DIR / "vector_rag_eval.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
