"""Head-to-head comparison of Vector RAG vs Graph RAG.

Runs both systems on all 20 evaluation questions and scores with LLM-as-judge on:
- Comprehensiveness (1-10)
- Relevance (1-10)
- Faithfulness (1-10)
- Multi-hop ability (1-10)

Generates comparison charts using matplotlib.

LangChain structured output: https://python.langchain.com/docs/how_to/structured_output/

Usage:
    python src/05_head_to_head_comparison.py
    python src/05_head_to_head_comparison.py --provider openai
"""

import sys
import json
import argparse
from pathlib import Path

# Shared imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pydantic import BaseModel, Field
from shared.llm_clients import chat_completion_structured

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
VECTOR_RESULTS_PATH = PROJECT_DIR / "output" / "vector_rag_eval.json"
GRAPH_RESULTS_PATH = PROJECT_DIR / "output" / "graph_rag_eval.json"
EVAL_QUESTIONS_PATH = PROJECT_DIR / "data" / "eval_questions.json"
OUTPUT_DIR = PROJECT_DIR / "output"


# ── Pydantic schema for head-to-head judgment ────────────────────────────

class HeadToHeadScore(BaseModel):
    """LLM judge scores for a single question comparing two systems."""
    vector_comprehensiveness: int = Field(ge=1, le=10, description="Vector RAG comprehensiveness 1-10")
    vector_relevance: int = Field(ge=1, le=10, description="Vector RAG relevance 1-10")
    vector_faithfulness: int = Field(ge=1, le=10, description="Vector RAG faithfulness 1-10")
    vector_multihop: int = Field(ge=1, le=10, description="Vector RAG multi-hop ability 1-10")
    graph_comprehensiveness: int = Field(ge=1, le=10, description="Graph RAG comprehensiveness 1-10")
    graph_relevance: int = Field(ge=1, le=10, description="Graph RAG relevance 1-10")
    graph_faithfulness: int = Field(ge=1, le=10, description="Graph RAG faithfulness 1-10")
    graph_multihop: int = Field(ge=1, le=10, description="Graph RAG multi-hop ability 1-10")
    winner: str = Field(description="Which system produced a better answer: 'vector', 'graph', or 'tie'")
    reasoning: str = Field(description="Brief explanation of the comparison")


# ── Run comparison ───────────────────────────────────────────────────────

def run_head_to_head(
    question: str,
    reference_answer: str,
    vector_answer: str,
    graph_answer: str,
    question_type: str,
    provider: str = "openai",
    model: str = None,
) -> HeadToHeadScore:
    """Compare two RAG system answers using LLM-as-judge."""
    prompt = f"""Compare these two answers to the same question. One comes from a Vector RAG
system (traditional semantic search) and the other from a Graph RAG system (knowledge graph-based).

Question: {question}
Question Type: {question_type}
Reference Answer: {reference_answer}

Vector RAG Answer: {vector_answer}

Graph RAG Answer: {graph_answer}

Score each system from 1-10 on:
1. Comprehensiveness: How thorough and complete is the answer?
2. Relevance: How well does it address the specific question asked?
3. Faithfulness: How factually accurate is the answer?
4. Multi-hop Ability: How well does it connect information across multiple sources/facts?

Then determine the overall winner: 'vector', 'graph', or 'tie'."""

    return chat_completion_structured(
        prompt=prompt,
        output_schema=HeadToHeadScore,
        system="You are an impartial judge comparing two RAG system answers. "
               "Score each dimension independently and fairly. Consider the question type "
               "when evaluating (e.g., multi-hop questions should weight the multi-hop metric higher).",
        provider=provider,
        model=model,
    )


def generate_charts(comparison_results: list[dict], output_dir: Path):
    """Generate matplotlib bar charts comparing the two systems."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib not installed. Skipping chart generation.")
        return

    metrics = ["comprehensiveness", "relevance", "faithfulness", "multihop"]
    metric_labels = ["Comprehensive-\nness", "Relevance", "Faithfulness", "Multi-hop\nAbility"]

    # 1. Overall comparison bar chart
    vector_avgs = []
    graph_avgs = []
    for m in metrics:
        v_scores = [r[f"vector_{m}"] for r in comparison_results]
        g_scores = [r[f"graph_{m}"] for r in comparison_results]
        vector_avgs.append(sum(v_scores) / len(v_scores))
        graph_avgs.append(sum(g_scores) / len(g_scores))

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, vector_avgs, width, label="Vector RAG", color="#4C78A8")
    bars2 = ax.bar(x + width/2, graph_avgs, width, label="Graph RAG", color="#E45756")

    ax.set_ylabel("Average Score (1-10)")
    ax.set_title("Vector RAG vs Graph RAG: Overall Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.set_ylim(0, 10.5)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    chart_path = output_dir / "comparison_overall.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"  Chart saved: {chart_path}")

    # 2. Per question-type comparison
    question_types = sorted(set(r["question_type"] for r in comparison_results))

    fig, axes = plt.subplots(1, len(question_types), figsize=(5 * len(question_types), 6), sharey=True)
    if len(question_types) == 1:
        axes = [axes]

    for ax, q_type in zip(axes, question_types):
        type_results = [r for r in comparison_results if r["question_type"] == q_type]

        v_avgs = []
        g_avgs = []
        for m in metrics:
            v_avgs.append(sum(r[f"vector_{m}"] for r in type_results) / len(type_results))
            g_avgs.append(sum(r[f"graph_{m}"] for r in type_results) / len(type_results))

        x = np.arange(len(metrics))
        ax.bar(x - width/2, v_avgs, width, label="Vector RAG", color="#4C78A8")
        ax.bar(x + width/2, g_avgs, width, label="Graph RAG", color="#E45756")
        ax.set_title(f"{q_type}\n({len(type_results)} questions)")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=8)
        ax.set_ylim(0, 10.5)
        if ax == axes[0]:
            ax.set_ylabel("Average Score (1-10)")
        ax.legend(fontsize=8)

    plt.suptitle("Vector RAG vs Graph RAG: By Question Type", fontsize=14, y=1.02)
    plt.tight_layout()
    chart_path = output_dir / "comparison_by_type.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved: {chart_path}")

    # 3. Win rate pie chart
    winners = [r["winner"] for r in comparison_results]
    win_counts = {
        "Vector RAG": winners.count("vector"),
        "Graph RAG": winners.count("graph"),
        "Tie": winners.count("tie"),
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#4C78A8", "#E45756", "#72B7B2"]
    sizes = list(win_counts.values())
    labels = [f"{k}\n({v}/{len(winners)})" for k, v in win_counts.items()]

    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 12})
    ax.set_title("Win Rate: Vector RAG vs Graph RAG")

    plt.tight_layout()
    chart_path = output_dir / "comparison_winrate.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"  Chart saved: {chart_path}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Head-to-Head RAG Comparison")
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--use-saved", action="store_true",
                        help="Use saved eval results instead of re-running")
    args = parser.parse_args()

    print("=" * 70)
    print("  KG Evaluation: Head-to-Head Comparison")
    print("=" * 70)
    print(f"\n  Provider: {args.provider}\n")

    # Load eval questions
    with open(EVAL_QUESTIONS_PATH) as f:
        eval_data = json.load(f)
    questions = eval_data["questions"]

    # Load previously saved results if available
    vector_results = None
    graph_results = None

    if VECTOR_RESULTS_PATH.exists():
        with open(VECTOR_RESULTS_PATH) as f:
            vector_results = json.load(f)
        print(f"  Loaded vector RAG results from: {VECTOR_RESULTS_PATH}")

    if GRAPH_RESULTS_PATH.exists():
        with open(GRAPH_RESULTS_PATH) as f:
            graph_results = json.load(f)
        print(f"  Loaded graph RAG results from: {GRAPH_RESULTS_PATH}")

    if not vector_results or not graph_results:
        print("\n  WARNING: Previous evaluation results not found.")
        print("  Run src/03_vector_rag_eval.py and src/04_graph_rag_eval.py first.")
        print("  Proceeding with placeholder answers for demonstration.\n")

    # Run head-to-head comparison
    comparison_results = []

    for i, q in enumerate(questions, 1):
        print(f"\n  [{i}/{len(questions)}] {q['question'][:55]}...")

        # Get answers from saved results or use placeholders
        vector_answer = "No vector RAG answer available."
        graph_answer = "No graph RAG answer available."

        if vector_results:
            for r in vector_results.get("per_question_results", []):
                if r["question_id"] == q["id"]:
                    vector_answer = r["answer"]
                    break

        if graph_results:
            for r in graph_results.get("per_question_results", []):
                if r["question_id"] == q["id"]:
                    graph_answer = r["answer"]
                    break

        # Run head-to-head judgment
        judgment = run_head_to_head(
            question=q["question"],
            reference_answer=q["reference_answer"],
            vector_answer=vector_answer,
            graph_answer=graph_answer,
            question_type=q["type"],
            provider=args.provider,
            model=args.model,
        )

        result = {
            "question_id": q["id"],
            "question_type": q["type"],
            "question": q["question"],
            "vector_comprehensiveness": judgment.vector_comprehensiveness,
            "vector_relevance": judgment.vector_relevance,
            "vector_faithfulness": judgment.vector_faithfulness,
            "vector_multihop": judgment.vector_multihop,
            "graph_comprehensiveness": judgment.graph_comprehensiveness,
            "graph_relevance": judgment.graph_relevance,
            "graph_faithfulness": judgment.graph_faithfulness,
            "graph_multihop": judgment.graph_multihop,
            "winner": judgment.winner,
            "reasoning": judgment.reasoning,
        }
        comparison_results.append(result)

        v_avg = (judgment.vector_comprehensiveness + judgment.vector_relevance +
                 judgment.vector_faithfulness + judgment.vector_multihop) / 4
        g_avg = (judgment.graph_comprehensiveness + judgment.graph_relevance +
                 judgment.graph_faithfulness + judgment.graph_multihop) / 4
        print(f"    Vector avg: {v_avg:.1f}  Graph avg: {g_avg:.1f}  Winner: {judgment.winner}")

    # Print summary
    print(f"\n\n{'='*70}")
    print("  HEAD-TO-HEAD COMPARISON SUMMARY")
    print(f"{'='*70}")

    metrics = ["comprehensiveness", "relevance", "faithfulness", "multihop"]
    print(f"\n  {'Metric':<22} {'Vector RAG':>12} {'Graph RAG':>12} {'Delta':>10}")
    print(f"  {'─'*56}")

    for m in metrics:
        v_avg = sum(r[f"vector_{m}"] for r in comparison_results) / len(comparison_results)
        g_avg = sum(r[f"graph_{m}"] for r in comparison_results) / len(comparison_results)
        delta = g_avg - v_avg
        delta_str = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
        print(f"  {m:<22} {v_avg:>12.2f} {g_avg:>12.2f} {delta_str:>10}")

    # Win rate
    winners = [r["winner"] for r in comparison_results]
    print(f"\n  Win Rate:")
    print(f"    Vector RAG wins: {winners.count('vector')}")
    print(f"    Graph RAG wins:  {winners.count('graph')}")
    print(f"    Ties:            {winners.count('tie')}")

    # Per question type
    question_types = sorted(set(r["question_type"] for r in comparison_results))
    print(f"\n  Per Question Type Analysis:")
    for q_type in question_types:
        type_results = [r for r in comparison_results if r["question_type"] == q_type]
        type_winners = [r["winner"] for r in type_results]
        print(f"\n    {q_type}:")
        print(f"      Vector wins: {type_winners.count('vector')}  "
              f"Graph wins: {type_winners.count('graph')}  "
              f"Ties: {type_winners.count('tie')}")

        for m in metrics:
            v = sum(r[f"vector_{m}"] for r in type_results) / len(type_results)
            g = sum(r[f"graph_{m}"] for r in type_results) / len(type_results)
            print(f"      {m:<20} V:{v:.1f}  G:{g:.1f}")

    # Generate charts
    print(f"\n  Generating charts...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_charts(comparison_results, OUTPUT_DIR)

    # Save results
    output = {
        "comparison_results": comparison_results,
        "summary": {
            "vector_wins": winners.count("vector"),
            "graph_wins": winners.count("graph"),
            "ties": winners.count("tie"),
            "overall_averages": {
                m: {
                    "vector": round(sum(r[f"vector_{m}"] for r in comparison_results) / len(comparison_results), 4),
                    "graph": round(sum(r[f"graph_{m}"] for r in comparison_results) / len(comparison_results), 4),
                }
                for m in metrics
            },
        },
    }
    output_path = OUTPUT_DIR / "head_to_head_comparison.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
