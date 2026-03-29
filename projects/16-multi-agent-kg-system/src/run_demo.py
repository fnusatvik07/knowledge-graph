"""End-to-end demo of the Multi-Agent KG System.

Steps:
1. Supervisor ingests all documents (parallel extraction)
2. Validator checks quality
3. Enricher fills gaps
4. Researcher answers sample questions with dynamic tool use

This is the main entry point for Project 16.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.agents.supervisor import run_supervisor
from src.tools.kg_tools import KGStatsTool


# ── Configuration ────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

SAMPLE_QUESTIONS = [
    "What companies are involved in developing large language models?",
    "How are knowledge graphs and RAG related?",
    "What is the connection between Transformers and vector databases?",
]


def load_documents() -> list[dict]:
    """Load documents from data/documents.json."""
    docs_path = DATA_DIR / "documents.json"
    with open(docs_path) as f:
        docs = json.load(f)
    return [{"id": d["id"], "title": d["title"], "text": d["text"]} for d in docs]


def print_kg_stats():
    """Print current KG statistics."""
    stats_tool = KGStatsTool()
    stats = stats_tool.invoke({})
    print("\n--- KG Statistics ---")
    print(stats)
    print()


def main():
    print("=" * 70)
    print("  MULTI-AGENT KG SYSTEM -- END-TO-END DEMO")
    print("=" * 70)

    # Load documents
    documents = load_documents()
    print(f"\nLoaded {len(documents)} documents from {DATA_DIR / 'documents.json'}")
    for doc in documents:
        print(f"  - [{doc['id']}] {doc['title']}")

    # ── Step 1: Ingest all documents ─────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 1: Ingesting documents (parallel extraction)")
    print("=" * 70)

    ingest_report = run_supervisor(
        task="Ingest all provided documents into the knowledge graph. Use parallel extraction.",
        documents=documents,
    )
    print(ingest_report)
    print_kg_stats()

    # ── Step 2: Validate quality ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 2: Validating KG quality")
    print("=" * 70)

    validate_report = run_supervisor(
        task="Validate the knowledge graph quality. Check for duplicates, type inconsistencies, and data issues.",
    )
    print(validate_report)
    print_kg_stats()

    # ── Step 3: Enrich the KG ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 3: Enriching the KG (filling gaps)")
    print("=" * 70)

    enrich_report = run_supervisor(
        task="Analyze the knowledge graph for gaps and enrich it with additional information from the web.",
    )
    print(enrich_report)
    print_kg_stats()

    # ── Step 4: Answer questions ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 4: Answering questions with dynamic tool use")
    print("=" * 70)

    research_report = run_supervisor(
        task="Answer the provided research questions using the knowledge graph and web search as needed.",
        questions=SAMPLE_QUESTIONS,
    )
    print(research_report)

    # ── Final summary ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL KG STATE")
    print("=" * 70)
    print_kg_stats()

    print("\nDemo complete. Check Neo4j Browser at http://localhost:7474 to explore the graph.")


if __name__ == "__main__":
    main()
