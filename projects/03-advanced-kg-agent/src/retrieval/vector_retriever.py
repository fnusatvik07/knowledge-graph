"""Vector-based retrieval against Neo4j vector index.

Embeds a query using a LangChain Embeddings instance and searches the
Neo4j vector index for the top-k most similar entity nodes.
"""
# LangChain docs: https://python.langchain.com/docs/integrations/chat/

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from shared.llm_clients import get_embeddings
from src.graph.neo4j_store import Neo4jStore


@dataclass
class VectorResult:
    """A single vector search result."""
    name: str
    entity_type: str
    description: str
    score: float
    properties: dict[str, Any] = field(default_factory=dict)


class VectorRetriever:
    """Retrieves entities from Neo4j via vector similarity search.

    Uses get_embeddings() to obtain a LangChain Embeddings instance for
    provider-agnostic embedding generation.
    LangChain embeddings docs: https://python.langchain.com/docs/integrations/text_embedding/
    """

    def __init__(
        self,
        store: Neo4jStore,
        index_name: str = "entity_embeddings",
        default_top_k: int = 5,
        provider: str = "openai",
    ) -> None:
        self.store = store
        self.index_name = index_name
        self.default_top_k = default_top_k
        # LangChain Embeddings instance for provider-agnostic embedding
        self.embeddings = get_embeddings(provider=provider)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[VectorResult]:
        """Embed the query and search the vector index.

        Args:
            query: Natural language query string.
            top_k: Number of results to return.

        Returns:
            List of VectorResult sorted by descending score.
        """
        k = top_k or self.default_top_k
        raw_results = self.store.vector_search(
            query_text=query,
            index_name=self.index_name,
            top_k=k,
        )

        results: list[VectorResult] = []
        for item in raw_results:
            node = item["node"]
            # Remove embedding from returned properties to keep output compact
            props = {k: v for k, v in node.items() if k != "embedding"}
            results.append(VectorResult(
                name=node.get("name", ""),
                entity_type=node.get("entity_type", ""),
                description=node.get("description", ""),
                score=item["score"],
                properties=props,
            ))

        return results

    def retrieve_with_context(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve results and include one-hop neighbor context.

        For each vector-matched node, also fetches immediate neighbors
        to provide richer context for downstream reasoning.
        """
        base_results = self.retrieve(query, top_k)
        enriched: list[dict[str, Any]] = []

        for result in base_results:
            neighbors = self.store.get_neighbors(
                name=result.name,
                max_hops=1,
            )
            enriched.append({
                "entity": {
                    "name": result.name,
                    "type": result.entity_type,
                    "description": result.description,
                    "score": result.score,
                },
                "neighbors": [
                    {
                        "name": n["node"].get("name", ""),
                        "type": n["node"].get("entity_type", ""),
                        "depth": n["depth"],
                    }
                    for n in neighbors
                ],
            })

        return enriched
