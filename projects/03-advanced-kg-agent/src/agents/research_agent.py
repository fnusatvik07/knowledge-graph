"""Research Agent: multi-hop question answering over the knowledge graph.

Decomposes complex questions into sub-questions, queries the KG iteratively
using hybrid retrieval, and synthesizes a final answer.
"""
# LangChain docs: https://python.langchain.com/docs/integrations/chat/

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from langgraph.graph import StateGraph, END

from shared.llm_clients import get_chat_model, chat_completion
from src.graph.neo4j_store import Neo4jStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import rerank
from src.prompts.reasoning import (
    QUESTION_DECOMPOSITION_SYSTEM,
    QUESTION_DECOMPOSITION_USER,
    ANSWER_SYNTHESIS_SYSTEM,
    ANSWER_SYNTHESIS_USER,
    MULTI_HOP_REASONING_SYSTEM,
    MULTI_HOP_REASONING_USER,
)


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class ResearchState(TypedDict):
    """State for the research agent."""
    question: str                          # Original question
    sub_questions: list[str]               # Decomposed sub-questions
    current_sub_q_index: int               # Index of current sub-question
    sub_answers: list[dict[str, Any]]      # Answers to sub-questions
    context_collected: list[dict[str, Any]]  # All retrieved context
    final_answer: str                      # Synthesized answer
    status: str


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def decompose_question(state: ResearchState) -> dict[str, Any]:
    """Decompose a complex question into sub-questions."""
    prompt = QUESTION_DECOMPOSITION_USER.format(question=state["question"])
    response = chat_completion(
        prompt=prompt,
        system=QUESTION_DECOMPOSITION_SYSTEM,
        model="gpt-4o-mini",
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    try:
        data = json.loads(response)
        sub_questions = data.get("sub_questions", [state["question"]])
    except json.JSONDecodeError:
        sub_questions = [state["question"]]

    return {
        "sub_questions": sub_questions,
        "current_sub_q_index": 0,
        "status": "decomposed",
    }


def query_knowledge_graph(state: ResearchState) -> dict[str, Any]:
    """Query the KG for the current sub-question using hybrid retrieval."""
    store: Neo4jStore = state["_store"]  # type: ignore[typeddict-item]
    retriever = HybridRetriever(store)

    idx = state["current_sub_q_index"]
    sub_q = state["sub_questions"][idx]

    # Retrieve relevant context
    results = retriever.retrieve(sub_q, top_k=8)

    # Convert to dicts for reranking
    candidates = [
        {
            "name": r.name,
            "entity_type": r.entity_type,
            "description": r.description,
            "rrf_score": r.rrf_score,
            "source": r.source,
        }
        for r in results
    ]

    # Rerank
    reranked = rerank(sub_q, candidates, top_k=5)

    # Generate a sub-answer using the retrieved context
    context_text = "\n".join(
        f"- {c['name']} [{c.get('entity_type', '')}]: {c.get('description', '')}"
        for c in reranked
    )

    prompt = MULTI_HOP_REASONING_USER.format(
        sub_question=sub_q,
        context=context_text,
        original_question=state["question"],
    )
    sub_answer = chat_completion(
        prompt=prompt,
        system=MULTI_HOP_REASONING_SYSTEM,
        model="gpt-4o-mini",
        temperature=0.0,
    )

    sub_answers = state.get("sub_answers", []) + [
        {"question": sub_q, "answer": sub_answer, "sources": reranked}
    ]
    context_collected = state.get("context_collected", []) + reranked

    return {
        "sub_answers": sub_answers,
        "context_collected": context_collected,
        "current_sub_q_index": idx + 1,
        "status": "querying",
    }


def should_continue_querying(state: ResearchState) -> str:
    """Check if there are more sub-questions to answer."""
    if state["current_sub_q_index"] < len(state["sub_questions"]):
        return "query_kg"
    return "synthesize"


def synthesize_answer(state: ResearchState) -> dict[str, Any]:
    """Synthesize a final answer from all sub-answers and context."""
    sub_answers_text = "\n\n".join(
        f"Sub-question: {sa['question']}\nAnswer: {sa['answer']}"
        for sa in state["sub_answers"]
    )

    prompt = ANSWER_SYNTHESIS_USER.format(
        original_question=state["question"],
        sub_answers=sub_answers_text,
    )
    final_answer = chat_completion(
        prompt=prompt,
        system=ANSWER_SYNTHESIS_SYSTEM,
        model="gpt-4o-mini",
        temperature=0.0,
    )

    return {"final_answer": final_answer, "status": "completed"}


# ---------------------------------------------------------------------------
# Workflow builder
# ---------------------------------------------------------------------------

def build_research_workflow(store: Neo4jStore) -> StateGraph:
    """Build the LangGraph workflow for research/QA."""

    def _query_kg(state: ResearchState) -> dict[str, Any]:
        state["_store"] = store  # type: ignore[typeddict-item]
        return query_knowledge_graph(state)

    workflow = StateGraph(ResearchState)
    workflow.add_node("decompose", decompose_question)
    workflow.add_node("query_kg", _query_kg)
    workflow.add_node("synthesize", synthesize_answer)

    workflow.set_entry_point("decompose")
    workflow.add_edge("decompose", "query_kg")
    workflow.add_conditional_edges(
        "query_kg",
        should_continue_querying,
        {"query_kg": "query_kg", "synthesize": "synthesize"},
    )
    workflow.add_edge("synthesize", END)

    return workflow


class ResearchAgent:
    """High-level interface for multi-hop question answering.

    Uses get_chat_model() to obtain a LangChain BaseChatModel, enabling
    provider-agnostic LLM access for the research workflow.
    LangChain chat model docs: https://python.langchain.com/docs/integrations/chat/
    """

    def __init__(self, store: Neo4jStore, provider: str = "openai") -> None:
        self.store = store
        self.provider = provider
        # Pre-build a LangChain chat model for the agent's LLM calls
        self.llm = get_chat_model(provider=provider)
        self.workflow = build_research_workflow(store)
        self.app = self.workflow.compile()

    def answer(self, question: str) -> dict[str, Any]:
        """Answer a complex question using multi-hop KG reasoning.

        Args:
            question: Natural language question.

        Returns:
            Dict with 'answer', 'sub_questions', 'sub_answers', 'sources'.
        """
        initial_state: ResearchState = {
            "question": question,
            "sub_questions": [],
            "current_sub_q_index": 0,
            "sub_answers": [],
            "context_collected": [],
            "final_answer": "",
            "status": "started",
        }
        result = self.app.invoke(initial_state)
        return {
            "answer": result.get("final_answer", ""),
            "sub_questions": result.get("sub_questions", []),
            "sub_answers": result.get("sub_answers", []),
            "sources": result.get("context_collected", []),
        }
