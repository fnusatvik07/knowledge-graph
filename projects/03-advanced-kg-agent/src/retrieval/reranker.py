"""LLM-based reranker for retrieval results.

Takes candidate results and the original query, uses the LLM to score
relevance, and returns reranked results.
"""
# LangChain docs: https://python.langchain.com/docs/integrations/chat/

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from shared.llm_clients import chat_completion


RERANK_SYSTEM_PROMPT = """\
You are a relevance scoring engine. Given a query and a list of candidate \
results from a knowledge graph, score each candidate on a scale of 0.0 to 1.0 \
based on how relevant it is to answering the query.

Consider:
- Direct relevance to the query topic
- Potential for providing useful context for answering the query
- Entity type relevance (e.g., a PERSON is more relevant for "who" questions)

Return a JSON array of objects with "name" and "score" keys, sorted by \
descending score. Example:
[{"name": "Entity A", "score": 0.95}, {"name": "Entity B", "score": 0.7}]

Return ONLY the JSON array, no other text."""


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int = 5,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
) -> list[dict[str, Any]]:
    """Rerank candidate results using an LLM.

    Args:
        query: The original user query.
        candidates: List of candidate dicts, each with at least 'name',
            'entity_type', and 'description'.
        top_k: Number of results to return after reranking.
        model: LLM model to use for scoring.
        provider: LLM provider ("openai", "anthropic", "ollama").

    Returns:
        Reranked list of candidate dicts with added 'relevance_score'.
    """
    if not candidates:
        return []

    # Format candidates for the LLM
    candidates_text = "\n".join(
        f"- {c.get('name', 'unknown')} [{c.get('entity_type', '')}]: "
        f"{c.get('description', 'No description')}"
        for c in candidates
    )

    prompt = (
        f"Query: {query}\n\n"
        f"Candidates:\n{candidates_text}\n\n"
        "Score each candidate for relevance to the query."
    )

    try:
        response = chat_completion(
            prompt=prompt,
            system=RERANK_SYSTEM_PROMPT,
            model=model,
            provider=provider,
            temperature=0.0,
        )

        # Parse LLM scores
        scores_raw = json.loads(response.strip())
        score_map: dict[str, float] = {
            item["name"]: item["score"] for item in scores_raw
        }
    except (json.JSONDecodeError, KeyError, TypeError):
        # Fallback: return candidates in original order with default scores
        score_map = {c["name"]: 1.0 - (i * 0.01) for i, c in enumerate(candidates)}

    # Merge scores back into candidates
    scored = []
    for c in candidates:
        name = c.get("name", "")
        scored.append({
            **c,
            "relevance_score": score_map.get(name, 0.0),
        })

    # Sort by relevance descending
    scored.sort(key=lambda x: x["relevance_score"], reverse=True)
    return scored[:top_k]


def rerank_hybrid_results(
    query: str,
    hybrid_results: list[Any],
    top_k: int = 5,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
) -> list[dict[str, Any]]:
    """Convenience wrapper that reranks HybridResult objects.

    Converts HybridResult dataclass instances to dicts, reranks, and
    returns enriched dicts.
    """
    candidates = [
        {
            "name": r.name,
            "entity_type": r.entity_type,
            "description": r.description,
            "rrf_score": r.rrf_score,
            "source": r.source,
        }
        for r in hybrid_results
    ]
    return rerank(query, candidates, top_k=top_k, model=model, provider=provider)
