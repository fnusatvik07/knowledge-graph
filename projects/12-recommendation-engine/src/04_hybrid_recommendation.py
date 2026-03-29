"""Hybrid recommendation engine combining three signals.

1. Collaborative filtering (graph traversal, from script 02)
2. Content-based filtering (shared attributes, from script 03)
3. Embedding similarity (LLM-generated movie description embeddings)

Weighted fusion of all three signals produces the final ranking.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import networkx as nx

# ── Path setup ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared.llm_clients import get_embedding_batch

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "movies_kg.json"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Import recommendation functions from other scripts
sys.path.insert(0, str(PROJECT_DIR / "src"))
from importlib import import_module

# We'll re-use the core functions rather than importing to keep things clear


def load_graph() -> nx.DiGraph:
    """Build graph from raw data."""
    with open(DATA_FILE) as f:
        data = json.load(f)

    G = nx.DiGraph()
    for entity in data["entities"]:
        attrs = {k: v for k, v in entity.items() if k != "id"}
        G.add_node(entity["id"], **attrs)
    for rel in data["relationships"]:
        attrs = {k: v for k, v in rel.items() if k not in ("source", "target")}
        G.add_edge(rel["source"], rel["target"], **attrs)
    return G


def get_user_ratings(G: nx.DiGraph, user_id: str) -> dict[str, float]:
    """Get all movies rated by a user."""
    ratings = {}
    for target in G.successors(user_id):
        edge = G.edges[user_id, target]
        if edge.get("type") == "RATED":
            ratings[target] = edge["score"]
    return ratings


# ── Signal 1: Collaborative Filtering ────────────────────────────────

def collaborative_scores(G: nx.DiGraph, user_id: str) -> dict[str, float]:
    """Get collaborative filtering scores for unseen movies.

    Returns dict mapping movie_id -> normalized score [0, 1].
    """
    user_ratings = get_user_ratings(G, user_id)
    all_users = [n for n, d in G.nodes(data=True) if d.get("type") == "User" and n != user_id]

    movie_scores: dict[str, float] = defaultdict(float)

    for other_user in all_users:
        other_ratings = get_user_ratings(G, other_user)
        shared = set(user_ratings) & set(other_ratings)
        if not shared:
            continue

        # Compute similarity
        union = set(user_ratings) | set(other_ratings)
        jaccard = len(shared) / len(union)
        avg_diff = sum(abs(user_ratings[m] - other_ratings[m]) for m in shared) / len(shared)
        agreement = 1.0 - (avg_diff / 5.0)
        similarity = 0.4 * jaccard + 0.6 * agreement

        for movie_id, rating in other_ratings.items():
            if movie_id not in user_ratings and rating >= 4.0:
                movie_scores[movie_id] += similarity * rating

    # Normalize to [0, 1]
    if movie_scores:
        max_score = max(movie_scores.values())
        if max_score > 0:
            movie_scores = {k: v / max_score for k, v in movie_scores.items()}

    return dict(movie_scores)


# ── Signal 2: Content-Based Filtering ────────────────────────────────

def content_based_scores(G: nx.DiGraph, user_id: str) -> dict[str, float]:
    """Get content-based scores for unseen movies.

    Returns dict mapping movie_id -> normalized score [0, 1].
    """
    user_ratings = get_user_ratings(G, user_id)
    liked_movies = {m: s for m, s in user_ratings.items() if s >= 4.0}
    all_rated = set(user_ratings)

    WEIGHTS = {"DIRECTED_BY": 3.0, "HAS_ACTOR": 2.0, "HAS_GENRE": 1.0}

    movie_scores: dict[str, float] = defaultdict(float)

    for movie_id, rating in liked_movies.items():
        rating_boost = rating / 5.0
        for target in G.successors(movie_id):
            edge_type = G.edges[movie_id, target].get("type")
            if edge_type not in WEIGHTS:
                continue
            weight = WEIGHTS[edge_type]
            # Find movies sharing this attribute
            for predecessor in G.predecessors(target):
                pred_edge = G.edges[predecessor, target]
                if pred_edge.get("type") == edge_type and predecessor not in all_rated:
                    if G.nodes[predecessor].get("type") == "Movie":
                        movie_scores[predecessor] += weight * rating_boost

    # Normalize to [0, 1]
    if movie_scores:
        max_score = max(movie_scores.values())
        if max_score > 0:
            movie_scores = {k: v / max_score for k, v in movie_scores.items()}

    return dict(movie_scores)


# ── Signal 3: Embedding Similarity ───────────────────────────────────

def compute_movie_embeddings(
    G: nx.DiGraph,
    provider: str = "openai",
    model: str | None = None,
) -> dict[str, np.ndarray]:
    """Compute embeddings for all movie descriptions.

    Returns dict mapping movie_id -> embedding vector.
    """
    movies = [
        (n, d) for n, d in G.nodes(data=True)
        if d.get("type") == "Movie" and d.get("description")
    ]

    movie_ids = [m[0] for m in movies]
    descriptions = [m[1]["description"] for m in movies]

    print(f"  Computing embeddings for {len(descriptions)} movies...")
    vectors = get_embedding_batch(descriptions, provider=provider, model=model)

    embeddings = {}
    for movie_id, vec in zip(movie_ids, vectors):
        embeddings[movie_id] = np.array(vec)

    return embeddings


def embedding_similarity_scores(
    G: nx.DiGraph,
    user_id: str,
    embeddings: dict[str, np.ndarray],
) -> dict[str, float]:
    """Score unseen movies by embedding similarity to liked movies.

    For each unseen movie, compute average cosine similarity to all
    movies the user liked.

    Returns dict mapping movie_id -> normalized score [0, 1].
    """
    user_ratings = get_user_ratings(G, user_id)
    liked_movies = {m: s for m, s in user_ratings.items() if s >= 4.0}
    all_rated = set(user_ratings)

    if not liked_movies:
        return {}

    # Get embeddings for liked movies
    liked_embeddings = {
        m: embeddings[m] for m in liked_movies if m in embeddings
    }
    if not liked_embeddings:
        return {}

    movie_scores: dict[str, float] = {}

    for movie_id, embedding in embeddings.items():
        if movie_id in all_rated:
            continue

        # Average cosine similarity to liked movies
        similarities = []
        for liked_id, liked_emb in liked_embeddings.items():
            cos_sim = np.dot(embedding, liked_emb) / (
                np.linalg.norm(embedding) * np.linalg.norm(liked_emb) + 1e-8
            )
            # Weight by user's rating
            rating_weight = liked_movies[liked_id] / 5.0
            similarities.append(cos_sim * rating_weight)

        movie_scores[movie_id] = float(np.mean(similarities))

    # Normalize to [0, 1]
    if movie_scores:
        min_score = min(movie_scores.values())
        max_score = max(movie_scores.values())
        score_range = max_score - min_score
        if score_range > 0:
            movie_scores = {k: (v - min_score) / score_range for k, v in movie_scores.items()}

    return movie_scores


# ── Hybrid Fusion ────────────────────────────────────────────────────

def hybrid_recommend(
    G: nx.DiGraph,
    user_id: str,
    embeddings: dict[str, np.ndarray],
    weights: dict[str, float] | None = None,
    top_k: int = 10,
) -> list[dict]:
    """Combine all three signals with weighted fusion.

    Args:
        G: The movie knowledge graph.
        user_id: Target user ID.
        embeddings: Pre-computed movie embeddings.
        weights: Signal weights. Defaults to
            {"collaborative": 0.35, "content": 0.35, "embedding": 0.30}
        top_k: Number of recommendations to return.

    Returns:
        Sorted list of recommendation dicts.
    """
    if weights is None:
        weights = {"collaborative": 0.35, "content": 0.35, "embedding": 0.30}

    print("  Computing collaborative filtering scores...")
    collab = collaborative_scores(G, user_id)
    print(f"    {len(collab)} candidates from collaborative filtering")

    print("  Computing content-based scores...")
    content = content_based_scores(G, user_id)
    print(f"    {len(content)} candidates from content-based filtering")

    print("  Computing embedding similarity scores...")
    emb_scores = embedding_similarity_scores(G, user_id, embeddings)
    print(f"    {len(emb_scores)} candidates from embedding similarity")

    # Gather all candidate movies
    all_candidates = set(collab) | set(content) | set(emb_scores)

    # Fuse scores
    fused = []
    for movie_id in all_candidates:
        c_score = collab.get(movie_id, 0.0) * weights["collaborative"]
        cb_score = content.get(movie_id, 0.0) * weights["content"]
        e_score = emb_scores.get(movie_id, 0.0) * weights["embedding"]
        total = c_score + cb_score + e_score

        fused.append(
            {
                "movie_id": movie_id,
                "title": G.nodes[movie_id].get("name", movie_id),
                "year": G.nodes[movie_id].get("year"),
                "total_score": round(total, 4),
                "collaborative_score": round(collab.get(movie_id, 0.0), 4),
                "content_score": round(content.get(movie_id, 0.0), 4),
                "embedding_score": round(emb_scores.get(movie_id, 0.0), 4),
                "signals_present": sum(
                    1 for s in [collab.get(movie_id), content.get(movie_id), emb_scores.get(movie_id)]
                    if s
                ),
            }
        )

    fused.sort(key=lambda x: x["total_score"], reverse=True)
    return fused[:top_k]


def main():
    print("Loading movie knowledge graph...")
    G = load_graph()

    # Compute embeddings for all movies
    print("\nComputing movie embeddings...")
    embeddings = compute_movie_embeddings(G)
    print(f"Computed embeddings for {len(embeddings)} movies\n")

    target_user = "user_1"
    target_name = G.nodes[target_user]["name"]

    print("=" * 70)
    print(f"HYBRID RECOMMENDATIONS FOR: {target_name}")
    print("=" * 70)

    # Show user's ratings
    ratings = get_user_ratings(G, target_user)
    print(f"\n{target_name}'s ratings:")
    for movie_id, score in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
        print(f"  {G.nodes[movie_id]['name']:45s} {score:.1f}/5.0")

    # Get hybrid recommendations
    print("\nGenerating hybrid recommendations...")
    recs = hybrid_recommend(G, target_user, embeddings, top_k=10)

    print(f"\n--- Top {len(recs)} Hybrid Recommendations ---")
    print(f"{'Rank':<5} {'Title':<40} {'Total':>7} {'Collab':>7} {'Content':>8} {'Embed':>7} {'Signals':>8}")
    print("-" * 85)
    for i, rec in enumerate(recs, 1):
        print(
            f"{i:<5} {rec['title'][:39]:<40} "
            f"{rec['total_score']:>7.4f} "
            f"{rec['collaborative_score']:>7.4f} "
            f"{rec['content_score']:>8.4f} "
            f"{rec['embedding_score']:>7.4f} "
            f"{rec['signals_present']:>8}"
        )

    # Save results
    output_path = OUTPUT_DIR / "hybrid_recommendations.json"
    with open(output_path, "w") as f:
        json.dump({"user": target_name, "recommendations": recs}, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
