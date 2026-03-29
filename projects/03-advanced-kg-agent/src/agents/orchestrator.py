"""Orchestrator: main LangGraph StateGraph for the Advanced KG Agent.

Defines two operational modes:
  - INGEST: document -> graph_builder_agent -> optional web_search enrichment
  - QUERY:  question -> research_agent -> answer

This is the primary entry point for the system.
"""
# LangChain docs: https://python.langchain.com/docs/integrations/chat/

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from langgraph.graph import StateGraph, END

from shared.config import get_neo4j_config
from shared.llm_clients import get_chat_model
from shared.document_loader import load_text_files
from src.graph.neo4j_store import Neo4jStore
from src.graph.temporal_layer import TemporalLayer
from src.agents.graph_builder_agent import GraphBuilderAgent
from src.agents.research_agent import ResearchAgent
from src.agents.web_search_agent import WebSearchAgent


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_DIR / "data" / "sources"
OUTPUT_DIR = PROJECT_DIR / "output"


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class OrchestratorState(TypedDict):
    """Top-level state for the orchestrator."""
    mode: str                                # "ingest" or "query"
    # Ingest fields
    documents: list[dict[str, Any]]
    ingest_result: dict[str, Any]
    enrich_entities: list[str]               # Entities to enrich via web
    enrichment_result: dict[str, Any]
    # Query fields
    question: str
    answer_result: dict[str, Any]
    # Shared
    status: str
    errors: list[str]


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def route_mode(state: OrchestratorState) -> str:
    """Route to the appropriate workflow based on mode."""
    mode = state.get("mode", "query")
    if mode == "ingest":
        return "load_documents"
    return "research"


def load_documents(state: OrchestratorState) -> dict[str, Any]:
    """Load documents from the data directory or use provided documents."""
    documents = state.get("documents", [])
    if not documents:
        documents = load_text_files(DATA_DIR)
    return {"documents": documents, "status": "documents_loaded"}


def build_graph(state: OrchestratorState) -> dict[str, Any]:
    """Run the graph builder agent on loaded documents."""
    store: Neo4jStore = state["_store"]  # type: ignore[typeddict-item]
    agent = GraphBuilderAgent(store)
    result = agent.ingest(state["documents"])
    return {"ingest_result": result, "status": "graph_built"}


def should_enrich(state: OrchestratorState) -> str:
    """Decide whether to run web enrichment."""
    entities = state.get("enrich_entities", [])
    if entities:
        return "web_enrich"
    return "done"


def web_enrich(state: OrchestratorState) -> dict[str, Any]:
    """Run web search enrichment for specified entities."""
    agent = WebSearchAgent()
    result = agent.search_and_extract(state["enrich_entities"])
    return {"enrichment_result": result, "status": "enriched"}


def research(state: OrchestratorState) -> dict[str, Any]:
    """Run the research agent to answer a question."""
    store: Neo4jStore = state["_store"]  # type: ignore[typeddict-item]
    agent = ResearchAgent(store)
    result = agent.answer(state["question"])
    return {"answer_result": result, "status": "answered"}


# ---------------------------------------------------------------------------
# Workflow builder
# ---------------------------------------------------------------------------

def build_orchestrator(store: Neo4jStore) -> StateGraph:
    """Build the main orchestrator workflow."""

    def _build_graph(state: OrchestratorState) -> dict[str, Any]:
        state["_store"] = store  # type: ignore[typeddict-item]
        return build_graph(state)

    def _research(state: OrchestratorState) -> dict[str, Any]:
        state["_store"] = store  # type: ignore[typeddict-item]
        return research(state)

    workflow = StateGraph(OrchestratorState)

    # Nodes
    workflow.add_node("load_documents", load_documents)
    workflow.add_node("build_graph", _build_graph)
    workflow.add_node("web_enrich", web_enrich)
    workflow.add_node("research", _research)

    # Entry routing
    workflow.set_conditional_entry_point(
        route_mode,
        {"load_documents": "load_documents", "research": "research"},
    )

    # Ingest path
    workflow.add_edge("load_documents", "build_graph")
    workflow.add_conditional_edges(
        "build_graph",
        should_enrich,
        {"web_enrich": "web_enrich", "done": END},
    )
    workflow.add_edge("web_enrich", END)

    # Query path
    workflow.add_edge("research", END)

    return workflow


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

class KGAgentOrchestrator:
    """High-level orchestrator for the Advanced KG Agent.

    Uses get_chat_model() to obtain a LangChain BaseChatModel, enabling
    provider-agnostic LLM access across all sub-agents.
    LangChain chat model docs: https://python.langchain.com/docs/integrations/chat/
    """

    def __init__(self, store: Neo4jStore | None = None, provider: str = "openai") -> None:
        if store is None:
            self.store = Neo4jStore()
            self.store.connect()
        else:
            self.store = store
        self.provider = provider
        # Pre-build a LangChain chat model for provider-agnostic access
        self.llm = get_chat_model(provider=provider)
        self.workflow = build_orchestrator(self.store)
        self.app = self.workflow.compile()

    def ingest(
        self,
        documents: list[dict[str, Any]] | None = None,
        enrich_entities: list[str] | None = None,
    ) -> dict[str, Any]:
        """Ingest documents into the knowledge graph.

        Args:
            documents: Optional list of documents. If None, loads from data/sources/.
            enrich_entities: Optional list of entity names to enrich via web search.
        """
        state: OrchestratorState = {
            "mode": "ingest",
            "documents": documents or [],
            "ingest_result": {},
            "enrich_entities": enrich_entities or [],
            "enrichment_result": {},
            "question": "",
            "answer_result": {},
            "status": "started",
            "errors": [],
        }
        result = self.app.invoke(state)
        return {
            "ingest_result": result.get("ingest_result", {}),
            "enrichment_result": result.get("enrichment_result", {}),
            "status": result.get("status", "unknown"),
        }

    def query(self, question: str) -> dict[str, Any]:
        """Answer a question using the knowledge graph.

        Args:
            question: Natural language question.
        """
        state: OrchestratorState = {
            "mode": "query",
            "documents": [],
            "ingest_result": {},
            "enrich_entities": [],
            "enrichment_result": {},
            "question": question,
            "answer_result": {},
            "status": "started",
            "errors": [],
        }
        result = self.app.invoke(state)
        return result.get("answer_result", {})

    def close(self) -> None:
        """Close the Neo4j connection."""
        self.store.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advanced KG Agent Orchestrator")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "ollama"],
        default="openai",
        help="LLM provider to use (default: openai)",
    )
    return parser.parse_args()


def main() -> None:
    """Demonstrate both INGEST and QUERY workflows."""
    args = parse_args()
    provider = args.provider

    print("=" * 70)
    print("Advanced Multi-Source Knowledge Graph Agent")
    print("=" * 70)
    print(f"Using LLM provider: {provider}")

    # Initialize
    store = Neo4jStore()
    store.connect()

    try:
        # Create vector index
        print("\n[1] Creating vector index...")
        store.create_vector_index(
            index_name="entity_embeddings",
            label="Entity",
            property_name="embedding",
            dimensions=1536,
        )

        orchestrator = KGAgentOrchestrator(store, provider=provider)

        # --- INGEST MODE ---
        print("\n[2] INGEST MODE: Loading and processing documents...")
        documents = load_text_files(DATA_DIR)
        print(f"    Found {len(documents)} documents:")
        for doc in documents:
            print(f"      - {doc['title']} ({len(doc['content'])} chars)")

        ingest_result = orchestrator.ingest(documents=documents)
        ir = ingest_result.get("ingest_result", {})
        print(f"\n    Ingestion complete:")
        print(f"      Nodes created:         {ir.get('nodes_created', 0)}")
        print(f"      Relationships created: {ir.get('relationships_created', 0)}")
        if ir.get("errors"):
            print(f"      Errors: {len(ir['errors'])}")

        # --- QUERY MODE ---
        print("\n" + "=" * 70)
        print("[3] QUERY MODE: Asking multi-hop questions...")

        questions = [
            "What is the relationship between Google and OpenAI in the AI landscape?",
            "How are transformer models connected to the major tech companies?",
            "Which organizations have invested in AI safety research?",
        ]

        for i, question in enumerate(questions, 1):
            print(f"\n  Question {i}: {question}")
            result = orchestrator.query(question)
            answer = result.get("answer", "No answer generated.")
            print(f"  Answer: {answer[:500]}...")
            sub_qs = result.get("sub_questions", [])
            if sub_qs:
                print(f"  Sub-questions decomposed: {len(sub_qs)}")

        print("\n" + "=" * 70)
        print("Done.")

    finally:
        store.close()


if __name__ == "__main__":
    main()
