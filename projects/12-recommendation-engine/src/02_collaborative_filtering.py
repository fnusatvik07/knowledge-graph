"""Graph-based collaborative filtering for movie recommendations.

For a target user, find similar users by traversing:
    User -> RATED -> Movie <- RATED <- OtherUser

Users who rated the same movies with similar scores are considered similar.
Recommend movies that similar users liked but the target hasn't seen.
Pure graph traversal -- no embeddings.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import networkx as nx

# ── Path setup ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "movies_kg.json"


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
    """Get all movies rated by a user with their scores.

    Returns:
        dict mapping movie_id -> rating score
    """
    ratings = {}
    for target in G.successors(user_id):
        edge = G.edges[user_id, target]
        if edge.get("type") == "RATED":
            ratings[target] = edge["score"]
    return ratings


def compute_user_similarity(
    ratings_a: dict[str, float],
    ratings_b: dict[str, float],
) -> float:
    """Compute similarity between two users based on shared movie ratings.

    Uses a combination of:
    - Jaccard overlap (how many movies in common)
    - Rating agreement (how similar the scores are for shared movies)

    Returns:
        Similarity score between 0 and 1.
    """
    shared_movies = set(ratings_a) & set(ratings_b)
    if not shared_movies:
        return 0.0

    # Jaccard component: fraction of overlap
    union = set(ratings_a) | set(ratings_b)
    jaccard = len(shared_movies) / len(union)

    # Rating agreement: 1 - normalized average absolute difference
    max_diff = 5.0  # ratings are 0-5
    total_diff = sum(abs(ratings_a[m] - ratings_b[m]) for m in shared_movies)
    avg_diff = total_diff / len(shared_movies)
    agreement = 1.0 - (avg_diff / max_diff)

    # Weighted combination
    similarity = 0.4 * jaccard + 0.6 * agreement
    return similarity


def find_similar_users(
    G: nx.DiGraph,
    target_user: str,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Find the most similar users to the target user.

    Traversal: target_user -> RATED -> Movie <- RATED <- other_user

    Returns:
        List of (user_id, similarity_score) tuples, sorted descending.
    """
    target_ratings = get_user_ratings(G, target_user)
    if not target_ratings:
        print(f"  User {target_user} has no ratings.")
        return []

    # Find all users who rated at least one movie in common
    candidate_users: set[str] = set()
    for movie_id in target_ratings:
        # Traverse backward: Movie <- RATED <- OtherUser
        for predecessor in G.predecessors(movie_id):
            edge = G.edges[predecessor, movie_id]
            if edge.get("type") == "RATED" and predecessor != target_user:
                candidate_users.add(predecessor)

    # Compute similarity with each candidate
    similarities = []
    for other_user in candidate_users:
        other_ratings = get_user_ratings(G, other_user)
        sim = compute_user_similarity(target_ratings, other_ratings)
        if sim > 0:
            similarities.append((other_user, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def recommend_collaborative(
    G: nx.DiGraph,
    target_user: str,
    top_k: int = 10,
    min_similar_users: int = 2,
) -> list[dict]:
    """Recommend movies using collaborative filtering.

    For each similar user, take movies they rated highly that the target
    hasn't seen. Weight by user similarity and their rating.

    Returns:
        List of recommendation dicts with movie info, score, and reasoning.
    """
    target_ratings = get_user_ratings(G, target_user)
    similar_users = find_similar_users(G, target_user, top_k=10)

    if not similar_users:
        print("  No similar users found.")
        return []

    # Aggregate recommendations from similar users
    movie_scores: dict[str, dict] = defaultdict(
        lambda: {"weighted_score": 0.0, "recommenders": []}
    )

    for other_user, similarity in similar_users:
        other_ratings = get_user_ratings(G, other_user)
        for movie_id, rating in other_ratings.items():
            # Skip movies the target has already seen
            if movie_id in target_ratings:
                continue
            # Only recommend movies rated >= 4.0
            if rating < 4.0:
                continue

            weighted = similarity * rating
            movie_scores[movie_id]["weighted_score"] += weighted
            movie_scores[movie_id]["recommenders"].append(
                {
                    "user": G.nodes[other_user]["name"],
                    "similarity": round(similarity, 3),
                    "rating": rating,
                }
            )

    # Build recommendation list
    recommendations = []
    for movie_id, info in movie_scores.items():
        num_recommenders = len(info["recommenders"])
        movie_attrs = G.nodes[movie_id]
        recommendations.append(
            {
                "movie_id": movie_id,
                "title": movie_attrs.get("name", movie_id),
                "year": movie_attrs.get("year"),
                "score": round(info["weighted_score"], 3),
                "num_recommenders": num_recommenders,
                "recommenders": info["recommenders"],
            }
        )

    # Sort by score, break ties by number of recommenders
    recommendations.sort(key=lambda x: (x["score"], x["num_recommenders"]), reverse=True)
    return recommendations[:top_k]


def main():
    print("Loading movie knowledge graph...")
    G = load_graph()

    # Get all users
    users = [n for n, d in G.nodes(data=True) if d.get("type") == "User"]
    print(f"Found {len(users)} users\n")

    # Run collaborative filtering for a target user
    target_user = "user_1"
    target_name = G.nodes[target_user]["name"]

    print("=" * 60)
    print(f"COLLABORATIVE FILTERING FOR: {target_name}")
    print("=" * 60)

    # Show the target user's ratings
    target_ratings = get_user_ratings(G, target_user)
    print(f"\n{target_name}'s ratings ({len(target_ratings)} movies):")
    for movie_id, score in sorted(target_ratings.items(), key=lambda x: x[1], reverse=True):
        title = G.nodes[movie_id].get("name", movie_id)
        print(f"  {title:45s}  {score:.1f}/5.0")

    # Find similar users
    print(f"\n--- Similar Users to {target_name} ---")
    similar = find_similar_users(G, target_user)
    for user_id, sim in similar:
        name = G.nodes[user_id]["name"]
        shared = set(target_ratings) & set(get_user_ratings(G, user_id))
        print(f"  {name:12s}  similarity={sim:.3f}  shared_movies={len(shared)}")

    # Generate recommendations
    print(f"\n--- Recommendations for {target_name} ---")
    recs = recommend_collaborative(G, target_user, top_k=10)
    if not recs:
        print("  No recommendations found.")
    else:
        for i, rec in enumerate(recs, 1):
            print(f"\n  {i}. {rec['title']} ({rec['year']})")
            print(f"     Score: {rec['score']:.3f}  (from {rec['num_recommenders']} similar users)")
            for r in rec["recommenders"]:
                print(f"       - {r['user']} (similarity={r['similarity']}, rated {r['rating']})")

    # Try another user for comparison
    print("\n")
    target_user = "user_5"
    target_name = G.nodes[target_user]["name"]
    print("=" * 60)
    print(f"COLLABORATIVE FILTERING FOR: {target_name}")
    print("=" * 60)

    target_ratings = get_user_ratings(G, target_user)
    print(f"\n{target_name}'s ratings ({len(target_ratings)} movies):")
    for movie_id, score in sorted(target_ratings.items(), key=lambda x: x[1], reverse=True):
        title = G.nodes[movie_id].get("name", movie_id)
        print(f"  {title:45s}  {score:.1f}/5.0")

    print(f"\n--- Recommendations for {target_name} ---")
    recs = recommend_collaborative(G, target_user, top_k=10)
    for i, rec in enumerate(recs, 1):
        print(f"\n  {i}. {rec['title']} ({rec['year']})")
        print(f"     Score: {rec['score']:.3f}  (from {rec['num_recommenders']} similar users)")


if __name__ == "__main__":
    main()
