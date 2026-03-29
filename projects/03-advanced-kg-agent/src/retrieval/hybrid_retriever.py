"""Hybrid retrieval combining vector search and graph traversal.

Runs both retrievers in parallel and fuses results using
Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.graph.neo4j_store import Neo4jStore
from src.retrieval.vector_retriever import VectorRetriever, VectorResult
from src.retrieval.graph_retriever import GraphRetriever, SubgraphContext


@dataclass
class HybridResult:
    """A single result from hybrid retrieval."""
    name: str
    entity_type: str
    description: str
    rrf_score: float
    vector_rank: int | None = None
    graph_depth: int | None = None
    properties: dict[str, Any] = field(default_factory=dict)
    source: str = ""  # "vector", "graph", or "both"


class HybridRetriever:
    """Combines vector and graph retrieval with reciprocal rank fusion."""

    def __init__(
        self,
        store: Neo4jStore,
        vector_top_k: int = 10,
        graph_max_hops: int = 2,
        rrf_k: int = 60,
    ) -> None:
        self.store = store
        self.vector_retriever = VectorRetriever(store, default_top_k=vector_top_k)
        self.graph_retriever = GraphRetriever(store, default_max_hops=graph_max_hops)
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        seed_entities: list[str] | None = None,
        top_k: int = 10,
    ) -> list[HybridResult]:
        """Run vector and graph retrieval, then fuse with RRF.

        Args:
            query: Natural language query for vector search.
            seed_entities: Optional seed entities for graph traversal.
                If not provided, uses top vector results as seeds.
            top_k: Number of final results to return.

        Returns:
            Ranked list of HybridResult.
        """
        # Run vector search first (we may need results as graph seeds)
        vector_results = self.vector_retriever.retrieve(query)

        # Determine graph seeds
        if not seed_entities:
            seed_entities = [r.name for r in vector_results[:3]]

        # Run graph retrieval (can be parallelized in future)
        graph_context = SubgraphContext(
            seed_entities=[], nodes=[], edges=[], max_hops=0
        )
        if seed_entities:
            graph_context = self.graph_retriever.retrieve(seed_entities)

        # Fuse results using RRF
        return self._reciprocal_rank_fusion(
            vector_results, graph_context, top_k
        )

    def retrieve_parallel(
        self,
        query: str,
        seed_entities: list[str],
        top_k: int = 10,
    ) -> list[HybridResult]:
        """Run vector and graph retrieval in parallel, then fuse.

        Use this when seed_entities are known ahead of time.
        """
        with ThreadPoolExecutor(max_workers=2) as executor:
            vector_future = executor.submit(
                self.vector_retriever.retrieve, query
            )
            graph_future = executor.submit(
                self.graph_retriever.retrieve, seed_entities
            )
            vector_results = vector_future.result()
            graph_context = graph_future.result()

        return self._reciprocal_rank_fusion(
            vector_results, graph_context, top_k
        )

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[VectorResult],
        graph_context: SubgraphContext,
        top_k: int,
    ) -> list[HybridResult]:
        """Merge results using Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank)) across retrieval sources.
        """
        k = self.rrf_k
        scores: dict[str, float] = {}
        metadata: dict[str, dict[str, Any]] = {}

        # Score vector results
        for rank, vr in enumerate(vector_results, start=1):
            scores[vr.name] = scores.get(vr.name, 0) + 1.0 / (k + rank)
            metadata[vr.name] = {
                "entity_type": vr.entity_type,
                "description": vr.description,
                "vector_rank": rank,
                "properties": vr.properties,
                "source": "vector",
            }

        # Score graph nodes
        for rank, node in enumerate(graph_context.nodes, start=1):
            scores[node.name] = scores.get(node.name, 0) + 1.0 / (k + rank)
            if node.name in metadata:
                metadata[node.name]["graph_depth"] = rank
                metadata[node.name]["source"] = "both"
            else:
                metadata[node.name] = {
                    "entity_type": node.entity_type,
                    "description": node.description,
                    "graph_depth": rank,
                    "properties": node.properties,
                    "source": "graph",
                }

        # Sort by RRF score descending
        sorted_names = sorted(scores.keys(), key=lambda n: scores[n], reverse=True)

        results: list[HybridResult] = []
        for name in sorted_names[:top_k]:
            meta = metadata[name]
            results.append(HybridResult(
                name=name,
                entity_type=meta.get("entity_type", ""),
                description=meta.get("description", ""),
                rrf_score=scores[name],
                vector_rank=meta.get("vector_rank"),
                graph_depth=meta.get("graph_depth"),
                properties=meta.get("properties", {}),
                source=meta.get("source", ""),
            ))

        return results

    def retrieve_as_text(
        self,
        query: str,
        seed_entities: list[str] | None = None,
        top_k: int = 10,
    ) -> str:
        """Retrieve and format results as text for LLM consumption."""
        results = self.retrieve(query, seed_entities, top_k)
        lines = [f"Hybrid retrieval results for: {query}", ""]
        for i, r in enumerate(results, 1):
            lines.append(
                f"{i}. [{r.entity_type}] {r.name} "
                f"(score={r.rrf_score:.4f}, via={r.source})"
            )
            if r.description:
                lines.append(f"   {r.description}")
        return "\n".join(lines)
