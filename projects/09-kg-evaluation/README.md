# Project 9: KG Evaluation & Benchmarking

Comprehensive evaluation framework for Knowledge Graphs and Graph RAG systems. Compare vector RAG vs graph RAG quantitatively, measure extraction quality, and benchmark graph structural properties.

## What This Project Does

1. **Extraction Evaluation** — Measure entity/relationship extraction accuracy against a gold standard using precision, recall, and F1 (with fuzzy matching)
2. **Graph Quality Metrics** — Evaluate structural properties: node coverage, edge density, connected components, orphan detection, ontology conformance, centrality analysis
3. **Vector RAG Evaluation** — Build a traditional vector RAG with ChromaDB and evaluate with faithfulness, relevance, context precision, and context recall metrics
4. **Graph RAG Evaluation** — Build a graph RAG and evaluate with the same metrics for apples-to-apples comparison
5. **Head-to-Head Comparison** — LLM-as-judge scoring on comprehensiveness, relevance, faithfulness, and multi-hop ability with visualization
6. **Benchmark Report** — Generate a comprehensive report aggregating all metrics with recommendations

## Evaluation Metrics

### Extraction Quality
- **Precision**: What fraction of extracted entities/relations are correct
- **Recall**: What fraction of expected entities/relations were extracted
- **F1 Score**: Harmonic mean of precision and recall
- Fuzzy matching with configurable similarity threshold

### RAG Quality (RAGAS-inspired)
- **Faithfulness**: Is the answer grounded in the retrieved context?
- **Answer Relevancy**: Does the answer address the question?
- **Context Precision**: Is the retrieved context relevant?
- **Context Recall**: Does the context cover all needed information?

### LLM-as-Judge (Head-to-Head)
- **Comprehensiveness** (1-10): How thorough is the answer?
- **Relevance** (1-10): How well does it address the question?
- **Faithfulness** (1-10): Is it factually accurate?
- **Multi-hop Ability** (1-10): Can it connect information across sources?

## Prerequisites

- Python 3.11+
- Dependencies: `chromadb`, `neo4j`, `matplotlib`, `langchain`, `langchain-openai`
- Neo4j running (for graph quality and graph RAG evaluations)

## Quick Start

```bash
# Run extraction evaluation
python src/01_extract_and_evaluate.py

# Run graph quality metrics (requires Neo4j with data)
python src/02_graph_quality_metrics.py

# Run vector RAG evaluation
python src/03_vector_rag_eval.py

# Run graph RAG evaluation
python src/04_graph_rag_eval.py

# Head-to-head comparison with charts
python src/05_head_to_head_comparison.py

# Generate comprehensive benchmark report
python src/06_benchmark_report.py
```

## File Structure

```
09-kg-evaluation/
├── README.md
├── data/
│   ├── gold_standard.json
│   └── eval_questions.json
├── output/
└── src/
    ├── __init__.py
    ├── 01_extract_and_evaluate.py
    ├── 02_graph_quality_metrics.py
    ├── 03_vector_rag_eval.py
    ├── 04_graph_rag_eval.py
    ├── 05_head_to_head_comparison.py
    └── 06_benchmark_report.py
```
