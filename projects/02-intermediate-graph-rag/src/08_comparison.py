"""Step 8: Compare vector RAG vs. graph RAG using LLM-as-judge evaluation.

Runs a set of test questions through both vector RAG and graph RAG (DRIFT
search), then uses an LLM to evaluate answer quality on three dimensions:
- Comprehensiveness: How complete and thorough is the answer?
- Relevance: How well does the answer address the question?
- Faithfulness: Is the answer grounded in factual information?

Results are displayed as a comparison table.
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

# Test questions spanning local, global, and hybrid query types
TEST_QUESTIONS = [
    # Local-style (entity-focused)
    {
        "question": "What is the role of Tesla in the electric vehicle industry?",
        "type": "local",
    },
    # Global-style (thematic, broad)
    {
        "question": "What are the main strategies for reducing global carbon emissions?",
        "type": "global",
    },
    # Hybrid / DRIFT-style (cross-domain connections)
    {
        "question": "How do battery technologies connect the renewable energy and electric vehicle sectors, and what policy incentives accelerate their development?",
        "type": "hybrid",
    },
    # Relationship-focused
    {
        "question": "How do agriculture and energy systems interact in the context of climate solutions?",
        "type": "hybrid",
    },
    # Specific + broad
    {
        "question": "What role does the Inflation Reduction Act play in the clean energy transition, including its effects on carbon capture and renewable energy?",
        "type": "hybrid",
    },
]


# ---------------------------------------------------------------------------
# Vector RAG pipeline (inline version for comparison)
# ---------------------------------------------------------------------------

def build_vector_index(provider: str = "openai"):
    """Build a ChromaDB vector index from corpus.

    Uses get_embedding_batch() from the shared LLM layer for batch embeddings.
    LangChain embeddings docs: https://python.langchain.com/docs/integrations/text_embedding/
    """
    import chromadb

    documents = load_text_files(CORPUS_DIR)
    all_chunks, all_metadatas, all_ids = [], [], []

    for doc in documents:
        chunks = chunk_text(doc["content"], chunk_size=800, overlap=150)
        for j, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append({"source": doc["title"]})
            all_ids.append(f"{doc['title']}_{j}")

    chroma_dir = OUTPUT_DIR / "chromadb_comparison"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    try:
        client.delete_collection("comparison")
    except Exception:
        pass

    collection = client.create_collection("comparison", metadata={"hnsw:space": "cosine"})

    batch_size = 20
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        embeddings = get_embedding_batch(batch, provider=provider)
        collection.add(
            documents=batch,
            embeddings=embeddings,
            metadatas=all_metadatas[i : i + batch_size],
            ids=all_ids[i : i + batch_size],
        )

    return collection


def vector_rag_answer(collection, question: str, provider: str = "openai") -> str:
    """Get an answer using vector RAG."""
    query_embedding = get_embedding(question, provider=provider)
    results = collection.query(query_embeddings=[query_embedding], n_results=5, include=["documents"])
    context = "\n\n".join(results["documents"][0])

    return chat_completion(
        prompt=f"Answer based on the context.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:",
        system="You are a research assistant. Answer thoroughly based on the provided context.",
        provider=provider,
    )


# ---------------------------------------------------------------------------
# Graph RAG pipeline (DRIFT-style, inline for comparison)
# ---------------------------------------------------------------------------

def graph_rag_answer(question: str, documents: list[dict], provider: str = "openai") -> str:
    """Get an answer using DRIFT-style graph RAG search."""
    # Phase 1: Entity extraction
    entity_response = chat_completion(
        prompt=f'Extract key entities from this question as a JSON list: "{question}"',
        provider=provider,
        response_format={"type": "json_object"},
    )
    try:
        data = json.loads(entity_response)
        entities = []
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    entities = v
                    break
        elif isinstance(data, list):
            entities = data
    except json.JSONDecodeError:
        entities = []

    # Phase 2: Local search - find entity passages
    local_context = ""
    for entity in entities:
        entity_lower = entity.lower()
        for doc in documents:
            for sentence in doc["content"].replace("\n", " ").split(". "):
                if entity_lower in sentence.lower():
                    local_context += f"- {sentence.strip()}. [Source: {doc['title']}]\n"

    # Phase 3: Global search - community summaries
    global_context = ""
    for doc in documents:
        summary = chat_completion(
            prompt=f"Summarize this document in 2-3 sentences, focusing on relevance to: {question}\n\n{doc['content'][:1500]}",
            provider=provider,
        )
        global_context += f"[{doc['title']}]: {summary}\n\n"

    # Phase 4: Synthesize
    return chat_completion(
        prompt=f"""Answer using both entity-level detail and thematic context.

Question: {question}

Entity-level context:
{local_context[:2500]}

Thematic summaries:
{global_context[:2500]}

Provide a comprehensive answer showing cross-domain connections.""",
        system="You are an expert analyst with access to a knowledge graph. "
               "Synthesize entity-level facts with thematic understanding.",
        provider=provider,
    )


# ---------------------------------------------------------------------------
# LLM-as-Judge evaluation
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are an expert evaluator comparing two AI-generated answers.
Rate each answer on three dimensions using a 1-10 scale.

Question: {question}

=== ANSWER A (Vector RAG) ===
{answer_a}

=== ANSWER B (Graph RAG) ===
{answer_b}

Evaluate both answers on:

1. **Comprehensiveness** (1-10): How complete and thorough is the answer?
   Does it cover all relevant aspects of the question?

2. **Relevance** (1-10): How well does the answer address the specific question asked?
   Does it stay focused and avoid tangential information?

3. **Faithfulness** (1-10): Does the answer appear factually accurate and grounded?
   Are claims specific and verifiable rather than vague?

Respond in this exact JSON format:
{{
    "answer_a": {{
        "comprehensiveness": <score>,
        "relevance": <score>,
        "faithfulness": <score>,
        "reasoning": "<brief explanation>"
    }},
    "answer_b": {{
        "comprehensiveness": <score>,
        "relevance": <score>,
        "faithfulness": <score>,
        "reasoning": "<brief explanation>"
    }}
}}"""


def evaluate_answers(question: str, answer_a: str, answer_b: str, provider: str = "openai") -> dict:
    """Use LLM-as-judge to evaluate two answers."""
    response = chat_completion(
        prompt=JUDGE_PROMPT.format(
            question=question, answer_a=answer_a, answer_b=answer_b
        ),
        system="You are a fair and rigorous evaluator. Score objectively.",
        provider=provider,
        response_format={"type": "json_object"},
    )

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {
            "answer_a": {"comprehensiveness": 5, "relevance": 5, "faithfulness": 5, "reasoning": "Parse error"},
            "answer_b": {"comprehensiveness": 5, "relevance": 5, "faithfulness": 5, "reasoning": "Parse error"},
        }


def print_comparison_table(results: list[dict]):
    """Print a formatted comparison table."""
    try:
        from tabulate import tabulate
        use_tabulate = True
    except ImportError:
        use_tabulate = False

    # Build table data
    headers = [
        "Question (type)",
        "Vec-Comp", "Vec-Rel", "Vec-Faith", "Vec-Avg",
        "Graph-Comp", "Graph-Rel", "Graph-Faith", "Graph-Avg",
        "Winner",
    ]
    rows = []

    vec_totals = {"comp": 0, "rel": 0, "faith": 0}
    graph_totals = {"comp": 0, "rel": 0, "faith": 0}

    for r in results:
        va = r["eval"]["answer_a"]
        vb = r["eval"]["answer_b"]

        v_avg = (va["comprehensiveness"] + va["relevance"] + va["faithfulness"]) / 3
        g_avg = (vb["comprehensiveness"] + vb["relevance"] + vb["faithfulness"]) / 3

        winner = "Graph RAG" if g_avg > v_avg else ("Vector RAG" if v_avg > g_avg else "Tie")

        # Truncate question for display
        q_display = r["question"][:40] + "..." if len(r["question"]) > 40 else r["question"]
        q_display = f"{q_display} ({r['type']})"

        rows.append([
            q_display,
            va["comprehensiveness"], va["relevance"], va["faithfulness"], f"{v_avg:.1f}",
            vb["comprehensiveness"], vb["relevance"], vb["faithfulness"], f"{g_avg:.1f}",
            winner,
        ])

        vec_totals["comp"] += va["comprehensiveness"]
        vec_totals["rel"] += va["relevance"]
        vec_totals["faith"] += va["faithfulness"]
        graph_totals["comp"] += vb["comprehensiveness"]
        graph_totals["rel"] += vb["relevance"]
        graph_totals["faith"] += vb["faithfulness"]

    n = len(results)
    v_overall = sum(vec_totals.values()) / (3 * n)
    g_overall = sum(graph_totals.values()) / (3 * n)
    overall_winner = "Graph RAG" if g_overall > v_overall else ("Vector RAG" if v_overall > g_overall else "Tie")

    rows.append([
        "AVERAGE",
        f"{vec_totals['comp']/n:.1f}", f"{vec_totals['rel']/n:.1f}", f"{vec_totals['faith']/n:.1f}", f"{v_overall:.1f}",
        f"{graph_totals['comp']/n:.1f}", f"{graph_totals['rel']/n:.1f}", f"{graph_totals['faith']/n:.1f}", f"{g_overall:.1f}",
        overall_winner,
    ])

    if use_tabulate:
        print(tabulate(rows, headers=headers, tablefmt="grid"))
    else:
        # Manual table formatting
        print(f"\n{'Question':<45} | {'Vector RAG':>30} | {'Graph RAG':>30} | Winner")
        print(f"{'':45} | {'Comp  Rel  Faith  Avg':>30} | {'Comp  Rel  Faith  Avg':>30} |")
        print("-" * 120)
        for row in rows:
            print(
                f"{row[0]:<45} | "
                f"{row[1]:>4}  {row[2]:>4}  {row[3]:>5}  {row[4]:>4} | "
                f"{row[5]:>4}  {row[6]:>4}  {row[7]:>5}  {row[8]:>4} | {row[9]}"
            )

    return v_overall, g_overall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vector RAG vs Graph RAG comparison")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "ollama"],
        default="openai",
        help="LLM provider to use (default: openai)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    provider = args.provider

    print("=" * 60)
    print("Vector RAG vs. Graph RAG Comparison")
    print("=" * 60)
    print()
    print(f"Using LLM provider: {provider}")
    print("Running the same questions through both pipelines and")
    print("evaluating with LLM-as-judge scoring.")
    print()

    # Check ChromaDB
    try:
        import chromadb  # noqa: F401
    except ImportError:
        print("ChromaDB not installed. Run: pip install chromadb")
        sys.exit(1)

    # Build vector index
    print("--- Building Vector RAG Index ---")
    collection = build_vector_index(provider=provider)
    print(f"Vector index ready: {collection.count()} chunks\n")

    # Load documents for graph RAG
    documents = load_text_files(CORPUS_DIR)
    print(f"Loaded {len(documents)} documents for Graph RAG\n")

    # Run comparison
    results = []

    for i, q_info in enumerate(TEST_QUESTIONS, 1):
        question = q_info["question"]
        q_type = q_info["type"]

        print(f"\n{'='*60}")
        print(f"Question {i} [{q_type}]: {question}")
        print("-" * 60)

        # Vector RAG answer
        print("\n  [Vector RAG] Generating answer...")
        vec_answer = vector_rag_answer(collection, question, provider=provider)
        print(f"  Answer preview: {vec_answer[:150]}...")

        # Graph RAG answer
        print("\n  [Graph RAG] Generating answer...")
        graph_answer = graph_rag_answer(question, documents, provider=provider)
        print(f"  Answer preview: {graph_answer[:150]}...")

        # LLM-as-judge evaluation
        print("\n  [Judge] Evaluating both answers...")
        evaluation = evaluate_answers(question, vec_answer, graph_answer, provider=provider)

        results.append({
            "question": question,
            "type": q_type,
            "vector_answer": vec_answer,
            "graph_answer": graph_answer,
            "eval": evaluation,
        })

        # Print per-question scores
        va = evaluation["answer_a"]
        vb = evaluation["answer_b"]
        print(f"\n  Vector RAG:  Comp={va['comprehensiveness']}, Rel={va['relevance']}, Faith={va['faithfulness']}")
        print(f"  Graph RAG:   Comp={vb['comprehensiveness']}, Rel={vb['relevance']}, Faith={vb['faithfulness']}")
        print(f"  Vector reasoning: {va.get('reasoning', 'N/A')}")
        print(f"  Graph reasoning:  {vb.get('reasoning', 'N/A')}")

    # Print comparison table
    print(f"\n\n{'='*60}")
    print("COMPARISON TABLE")
    print("=" * 60)
    v_avg, g_avg = print_comparison_table(results)

    # Analysis
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print("=" * 60)

    # Count wins by question type
    type_wins = {"local": {"vector": 0, "graph": 0, "tie": 0},
                 "global": {"vector": 0, "graph": 0, "tie": 0},
                 "hybrid": {"vector": 0, "graph": 0, "tie": 0}}

    for r in results:
        va = r["eval"]["answer_a"]
        vb = r["eval"]["answer_b"]
        v_score = (va["comprehensiveness"] + va["relevance"] + va["faithfulness"]) / 3
        g_score = (vb["comprehensiveness"] + vb["relevance"] + vb["faithfulness"]) / 3

        if g_score > v_score:
            type_wins[r["type"]]["graph"] += 1
        elif v_score > g_score:
            type_wins[r["type"]]["vector"] += 1
        else:
            type_wins[r["type"]]["tie"] += 1

    print(f"\nOverall averages: Vector RAG = {v_avg:.1f}, Graph RAG = {g_avg:.1f}")
    print(f"\nWins by question type:")
    for qtype, wins in type_wins.items():
        total = sum(wins.values())
        if total > 0:
            print(f"  {qtype:>8}: Vector={wins['vector']}, Graph={wins['graph']}, Tie={wins['tie']}")

    print(f"\nKey insights:")
    print("  - Graph RAG typically excels on broad, thematic questions (global)")
    print("  - Graph RAG shows strength on cross-domain connections (hybrid)")
    print("  - Vector RAG can be competitive on focused, factual questions (local)")
    print("  - Graph RAG provides more comprehensive answers at the cost of latency")

    # Save full results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "comparison_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    main()
