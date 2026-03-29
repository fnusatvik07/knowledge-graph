"""Generate a comprehensive benchmark report aggregating all evaluation metrics.

Collects results from all evaluation scripts and produces a single markdown report:
- Extraction quality (precision, recall, F1)
- Graph quality metrics (coverage, density, components, conformance)
- Vector RAG vs Graph RAG RAGAS scores
- Head-to-head comparison with LLM-as-judge
- Per-question-type analysis
- Recommendations for when to use which approach

Usage:
    python src/06_benchmark_report.py
    python src/06_benchmark_report.py --format html
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Shared imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"

EXTRACTION_RESULTS = OUTPUT_DIR / "extraction_evaluation.json"
GRAPH_QUALITY_RESULTS = OUTPUT_DIR / "graph_quality_metrics.json"
VECTOR_RAG_RESULTS = OUTPUT_DIR / "vector_rag_eval.json"
GRAPH_RAG_RESULTS = OUTPUT_DIR / "graph_rag_eval.json"
COMPARISON_RESULTS = OUTPUT_DIR / "head_to_head_comparison.json"


def load_json_safe(path: Path) -> dict | None:
    """Load a JSON file, returning None if it doesn't exist."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def generate_extraction_section(data: dict) -> str:
    """Generate the extraction quality section."""
    lines = []
    lines.append("## 1. Entity & Relationship Extraction Quality\n")
    lines.append("Evaluation of LLM-based extraction against a gold standard dataset.\n")

    entity = data.get("entity_aggregate", {})
    rel = data.get("relationship_aggregate", {})

    lines.append("### Aggregate Metrics\n")
    lines.append("| Metric | Entity Extraction | Relationship Extraction |")
    lines.append("|--------|:-----------------:|:-----------------------:|")
    lines.append(f"| Precision | {entity.get('precision', 0):.2%} | {rel.get('precision', 0):.2%} |")
    lines.append(f"| Recall | {entity.get('recall', 0):.2%} | {rel.get('recall', 0):.2%} |")
    lines.append(f"| F1 Score | {entity.get('f1', 0):.2%} | {rel.get('f1', 0):.2%} |")
    lines.append("")

    # Per-document breakdown
    per_doc = data.get("per_document", [])
    if per_doc:
        lines.append("### Per-Document Breakdown\n")
        lines.append("| Document | Entity F1 | Rel F1 | Missed Entities | Extra Entities |")
        lines.append("|----------|:---------:|:------:|:---------------:|:--------------:|")
        for doc in per_doc:
            e = doc.get("entity_eval", {})
            r = doc.get("relationship_eval", {})
            missed = len(e.get("missed", []))
            extra = len(e.get("extra", []))
            lines.append(f"| {doc['doc_id']} | {e.get('f1', 0):.2%} | {r.get('f1', 0):.2%} | {missed} | {extra} |")
        lines.append("")

    return "\n".join(lines)


def generate_graph_quality_section(data: dict) -> str:
    """Generate the graph quality metrics section."""
    lines = []
    lines.append("## 2. Graph Structural Quality\n")
    lines.append("Evaluation of the knowledge graph's structural properties.\n")

    # Node coverage
    coverage = data.get("node_coverage", {})
    if coverage:
        lines.append("### Node Coverage\n")
        lines.append(f"- Expected entities: {coverage.get('expected_count', 'N/A')}")
        lines.append(f"- Graph entities: {coverage.get('graph_count', 'N/A')}")
        lines.append(f"- **Coverage: {coverage.get('coverage', 0):.2%}**")
        lines.append("")

    # Edge density
    density = data.get("edge_density", {})
    if density:
        lines.append("### Edge Density\n")
        lines.append(f"- Nodes: {density.get('node_count', 0)}")
        lines.append(f"- Edges: {density.get('edge_count', 0)}")
        lines.append(f"- Density: {density.get('density', 0):.6f}")
        lines.append(f"- Average degree: {density.get('avg_degree', 0)}")
        lines.append("")

    # Components
    components = data.get("connected_components", {})
    if components:
        lines.append("### Connected Components\n")
        lines.append(f"- Number of components: {components.get('num_components', 0)}")
        lines.append(f"- Largest component: {components.get('largest_component_size', 0)} nodes")
        lines.append(f"- Fully connected: {'Yes' if components.get('is_connected') else 'No'}")
        lines.append("")

    # Orphans
    orphans = data.get("orphan_nodes", {})
    if orphans:
        lines.append("### Orphan Nodes\n")
        lines.append(f"- Orphan count: {orphans.get('orphan_count', 0)} / {orphans.get('total_nodes', 0)}")
        lines.append(f"- Orphan percentage: {orphans.get('orphan_percentage', 0)}%")
        lines.append("")

    # Conformance
    conf = data.get("ontology_conformance", {})
    if conf:
        lines.append("### Ontology Conformance\n")
        lines.append(f"- Entity type conformance: {conf.get('entity_conformance', 0):.2%}")
        lines.append(f"- Relationship type conformance: {conf.get('relationship_conformance', 0):.2%}")
        lines.append("")

    return "\n".join(lines)


def generate_rag_comparison_section(vector_data: dict, graph_data: dict) -> str:
    """Generate the RAG comparison section."""
    lines = []
    lines.append("## 3. RAG System Evaluation (RAGAS Metrics)\n")
    lines.append("Comparison of Vector RAG (ChromaDB) vs Graph RAG (Neo4j) on the same evaluation questions.\n")

    v_avg = vector_data.get("overall_averages", {})
    g_avg = graph_data.get("overall_averages", {})

    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    metric_labels = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevancy",
        "context_precision": "Context Precision",
        "context_recall": "Context Recall",
    }

    lines.append("### Overall Averages\n")
    lines.append("| Metric | Vector RAG | Graph RAG | Delta |")
    lines.append("|--------|:----------:|:---------:|:-----:|")
    for m in metrics:
        v = v_avg.get(m, 0)
        g = g_avg.get(m, 0)
        delta = g - v
        delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
        lines.append(f"| {metric_labels[m]} | {v:.4f} | {g:.4f} | {delta_str} |")
    lines.append("")

    # Per question type
    v_types = vector_data.get("per_type_averages", {})
    g_types = graph_data.get("per_type_averages", {})

    if v_types and g_types:
        lines.append("### Per Question Type\n")
        for q_type in sorted(set(list(v_types.keys()) + list(g_types.keys()))):
            lines.append(f"\n#### {q_type}\n")
            lines.append("| Metric | Vector RAG | Graph RAG |")
            lines.append("|--------|:----------:|:---------:|")
            for m in metrics:
                v = v_types.get(q_type, {}).get(m, 0)
                g = g_types.get(q_type, {}).get(m, 0)
                lines.append(f"| {metric_labels[m]} | {v:.4f} | {g:.4f} |")
        lines.append("")

    return "\n".join(lines)


def generate_head_to_head_section(data: dict) -> str:
    """Generate the head-to-head comparison section."""
    lines = []
    lines.append("## 4. Head-to-Head LLM-as-Judge Comparison\n")
    lines.append("Direct comparison scored by LLM on comprehensiveness, relevance, faithfulness, and multi-hop ability (1-10 scale).\n")

    summary = data.get("summary", {})
    lines.append("### Win Rate\n")
    lines.append(f"- **Vector RAG wins:** {summary.get('vector_wins', 0)}")
    lines.append(f"- **Graph RAG wins:** {summary.get('graph_wins', 0)}")
    lines.append(f"- **Ties:** {summary.get('ties', 0)}")
    lines.append("")

    overall = summary.get("overall_averages", {})
    if overall:
        lines.append("### Average Scores (1-10)\n")
        lines.append("| Metric | Vector RAG | Graph RAG |")
        lines.append("|--------|:----------:|:---------:|")
        for m in ["comprehensiveness", "relevance", "faithfulness", "multihop"]:
            m_data = overall.get(m, {})
            lines.append(f"| {m.title()} | {m_data.get('vector', 0):.2f} | {m_data.get('graph', 0):.2f} |")
        lines.append("")

    # Charts reference
    lines.append("### Visualizations\n")
    lines.append("- `output/comparison_overall.png` - Overall metric comparison")
    lines.append("- `output/comparison_by_type.png` - Per question type breakdown")
    lines.append("- `output/comparison_winrate.png` - Win rate distribution")
    lines.append("")

    return "\n".join(lines)


def generate_recommendations(
    extraction_data: dict | None,
    vector_data: dict | None,
    graph_data: dict | None,
    comparison_data: dict | None,
) -> str:
    """Generate recommendations based on the evaluation results."""
    lines = []
    lines.append("## 5. Recommendations\n")
    lines.append("Based on the evaluation results, here are recommendations for when to use each approach:\n")

    lines.append("### When to Use Vector RAG\n")
    lines.append("- **Factual lookup questions**: When users need direct answers from specific documents")
    lines.append("- **Speed-critical applications**: Vector search is typically faster than graph traversal")
    lines.append("- **Simple retrieval**: When the answer exists in a single document or passage")
    lines.append("- **Low infrastructure requirements**: ChromaDB requires no external database server")
    lines.append("")

    lines.append("### When to Use Graph RAG\n")
    lines.append("- **Multi-hop reasoning**: When answers require connecting facts across multiple documents")
    lines.append("- **Relationship-heavy queries**: When the question asks about connections between entities")
    lines.append("- **Global/comparative questions**: When understanding the full corpus structure matters")
    lines.append("- **Entity-centric exploration**: When users want to explore an entity's full context")
    lines.append("")

    lines.append("### Hybrid Approach (Recommended)\n")
    lines.append("- Use **vector search** for initial retrieval to find relevant documents")
    lines.append("- Use **graph traversal** to expand context with related entities and relationships")
    lines.append("- Combine both contexts for the most comprehensive answers")
    lines.append("- Route questions to the appropriate system based on question type classification")
    lines.append("")

    lines.append("### Improving Extraction Quality\n")
    lines.append("- Use domain-specific few-shot examples in extraction prompts")
    lines.append("- Implement entity resolution with embedding-based similarity")
    lines.append("- Add a human-in-the-loop validation step for high-value knowledge bases")
    lines.append("- Consider using ontology constraints to guide extraction")
    lines.append("")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate Benchmark Report")
    parser.add_argument("--format", choices=["markdown", "html"], default="markdown")
    args = parser.parse_args()

    print("=" * 70)
    print("  KG Evaluation: Benchmark Report Generator")
    print("=" * 70)

    # Load all available results
    extraction_data = load_json_safe(EXTRACTION_RESULTS)
    graph_quality_data = load_json_safe(GRAPH_QUALITY_RESULTS)
    vector_data = load_json_safe(VECTOR_RAG_RESULTS)
    graph_data = load_json_safe(GRAPH_RAG_RESULTS)
    comparison_data = load_json_safe(COMPARISON_RESULTS)

    available = sum(1 for d in [extraction_data, graph_quality_data, vector_data, graph_data, comparison_data] if d)
    print(f"\n  Found {available}/5 evaluation result files.\n")

    # Build the report
    report_lines = []
    report_lines.append("# Knowledge Graph Evaluation & Benchmarking Report\n")
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append("---\n")

    # Table of Contents
    report_lines.append("## Table of Contents\n")
    report_lines.append("1. [Entity & Relationship Extraction Quality](#1-entity--relationship-extraction-quality)")
    report_lines.append("2. [Graph Structural Quality](#2-graph-structural-quality)")
    report_lines.append("3. [RAG System Evaluation](#3-rag-system-evaluation-ragas-metrics)")
    report_lines.append("4. [Head-to-Head Comparison](#4-head-to-head-llm-as-judge-comparison)")
    report_lines.append("5. [Recommendations](#5-recommendations)")
    report_lines.append("\n---\n")

    # Section 1: Extraction
    if extraction_data:
        report_lines.append(generate_extraction_section(extraction_data))
        print("  Added: Extraction quality section")
    else:
        report_lines.append("## 1. Entity & Relationship Extraction Quality\n")
        report_lines.append("*No extraction evaluation results found. Run `src/01_extract_and_evaluate.py` first.*\n")

    report_lines.append("---\n")

    # Section 2: Graph Quality
    if graph_quality_data:
        report_lines.append(generate_graph_quality_section(graph_quality_data))
        print("  Added: Graph quality section")
    else:
        report_lines.append("## 2. Graph Structural Quality\n")
        report_lines.append("*No graph quality results found. Run `src/02_graph_quality_metrics.py` first.*\n")

    report_lines.append("---\n")

    # Section 3: RAG Comparison
    if vector_data and graph_data:
        report_lines.append(generate_rag_comparison_section(vector_data, graph_data))
        print("  Added: RAG comparison section")
    else:
        report_lines.append("## 3. RAG System Evaluation\n")
        missing = []
        if not vector_data:
            missing.append("`src/03_vector_rag_eval.py`")
        if not graph_data:
            missing.append("`src/04_graph_rag_eval.py`")
        report_lines.append(f"*Missing results. Run {' and '.join(missing)} first.*\n")

    report_lines.append("---\n")

    # Section 4: Head-to-Head
    if comparison_data:
        report_lines.append(generate_head_to_head_section(comparison_data))
        print("  Added: Head-to-head comparison section")
    else:
        report_lines.append("## 4. Head-to-Head Comparison\n")
        report_lines.append("*No comparison results found. Run `src/05_head_to_head_comparison.py` first.*\n")

    report_lines.append("---\n")

    # Section 5: Recommendations
    report_lines.append(generate_recommendations(extraction_data, vector_data, graph_data, comparison_data))
    print("  Added: Recommendations section")

    # Write report
    report_content = "\n".join(report_lines)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.format == "html":
        try:
            import markdown
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>KG Evaluation Benchmark Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               max-width: 900px; margin: 40px auto; padding: 0 20px; line-height: 1.6; }}
        table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: center; }}
        th {{ background-color: #f4f4f4; font-weight: 600; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
        h1 {{ border-bottom: 2px solid #333; padding-bottom: 8px; }}
        h2 {{ color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 4px; }}
        hr {{ border: none; border-top: 1px solid #eee; margin: 32px 0; }}
    </style>
</head>
<body>
{markdown.markdown(report_content, extensions=['tables', 'fenced_code'])}
</body>
</html>"""
            output_path = OUTPUT_DIR / "benchmark_report.html"
            with open(output_path, "w") as f:
                f.write(html_content)
            print(f"\n  HTML report saved to: {output_path}")
        except ImportError:
            print("  `markdown` package not installed. Falling back to markdown format.")
            output_path = OUTPUT_DIR / "benchmark_report.md"
            with open(output_path, "w") as f:
                f.write(report_content)
            print(f"\n  Markdown report saved to: {output_path}")
    else:
        output_path = OUTPUT_DIR / "benchmark_report.md"
        with open(output_path, "w") as f:
            f.write(report_content)
        print(f"\n  Markdown report saved to: {output_path}")

    print("\n  Report generation complete!")
    print(f"  Sections included: {available + 1} (including recommendations)")


if __name__ == "__main__":
    main()
