"""Content-based recommendation via knowledge graph traversal.

For movies a user liked, find other movies that share genres, actors, or
directors by traversing the graph:

    LikedMovie --[HAS_GENRE]--> Genre    <--[HAS_GENRE]-- OtherMovie
    LikedMovie --[HAS_ACTOR]--> Actor    <--[HAS_ACTOR]-- OtherMovie
    LikedMovie --[DIRECTED_BY]--> Director <--[DIRECTED_BY]-- OtherMovie

Score by number of shared attributes and rank recommendations.
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


def get_user_liked_movies(
    G: nx.DiGraph,
    user_id: str,
    min_rating: float = 4.0,
) -> dict[str, float]:
    """Get movies the user rated at or above min_rating.

    Returns:
        dict mapping movie_id -> rating score
    """
    liked = {}
    for target in G.successors(user_id):
        edge = G.edges[user_id, target]
        if edge.get("type") == "RATED" and edge.get("score", 0) >= min_rating:
            liked[target] = edge["score"]
    return liked


def get_movie_attributes(G: nx.DiGraph, movie_id: str) -> dict[str, set[str]]:
    """Get all attributes (genres, actors, directors) of a movie.

    Returns:
        dict with keys 'genres', 'actors', 'directors', each mapping to
        a set of entity IDs.
    """
    attrs: dict[str, set[str]] = {"genres": set(), "actors": set(), "directors": set()}

    for target in G.successors(movie_id):
        edge = G.edges[movie_id, target]
        edge_type = edge.get("type")
        if edge_type == "HAS_GENRE":
            attrs["genres"].add(target)
        elif edge_type == "HAS_ACTOR":
            attrs["actors"].add(target)
        elif edge_type == "DIRECTED_BY":
            attrs["directors"].add(target)

    return attrs


def find_movies_sharing_attribute(
    G: nx.DiGraph,
    attribute_id: str,
    edge_type: str,
    exclude_movies: set[str],
) -> set[str]:
    """Find all movies that share a given attribute.

    Traversal: Attribute <-- [edge_type] -- OtherMovie

    Returns:
        Set of movie IDs sharing this attribute.
    """
    sharing = set()
    for predecessor in G.predecessors(attribute_id):
        edge = G.edges[predecessor, attribute_id]
        if edge.get("type") == edge_type and predecessor not in exclude_movies:
            sharing.add(predecessor)
    return sharing


def recommend_content_based(
    G: nx.DiGraph,
    user_id: str,
    top_k: int = 10,
    min_rating: float = 4.0,
) -> list[dict]:
    """Generate content-based recommendations.

    For each liked movie, find other movies sharing attributes (genres,
    actors, directors). Score by number of shared attributes, weighted
    by attribute type.

    Weights:
        - Shared director: 3 points (strong signal)
        - Shared actor: 2 points
        - Shared genre: 1 point

    Returns:
        Sorted list of recommendation dicts.
    """
    liked_movies = get_user_liked_movies(G, user_id, min_rating)
    all_rated = {
        t for t in G.successors(user_id)
        if G.edges[user_id, t].get("type") == "RATED"
    }

    if not liked_movies:
        return []

    # Attribute weights
    WEIGHTS = {"directors": 3.0, "actors": 2.0, "genres": 1.0}
    EDGE_TYPES = {"genres": "HAS_GENRE", "actors": "HAS_ACTOR", "directors": "DIRECTED_BY"}

    # Track scores and explanations for each candidate movie
    candidate_scores: dict[str, dict] = defaultdict(
        lambda: {"score": 0.0, "shared_attributes": [], "source_movies": set()}
    )

    for movie_id, user_rating in liked_movies.items():
        movie_attrs = get_movie_attributes(G, movie_id)
        movie_name = G.nodes[movie_id]["name"]
        rating_boost = user_rating / 5.0  # scale by how much user liked it

        for attr_category in ("genres", "actors", "directors"):
            weight = WEIGHTS[attr_category]
            edge_type = EDGE_TYPES[attr_category]

            for attr_id in movie_attrs[attr_category]:
                attr_name = G.nodes[attr_id]["name"]
                # Find other movies with this attribute
                sharing = find_movies_sharing_attribute(G, attr_id, edge_type, all_rated)

                for candidate_id in sharing:
                    if G.nodes[candidate_id].get("type") != "Movie":
                        continue
                    candidate_scores[candidate_id]["score"] += weight * rating_boost
                    candidate_scores[candidate_id]["shared_attributes"].append(
                        {
                            "type": attr_category.rstrip("s"),
                            "name": attr_name,
                            "via_movie": movie_name,
                        }
                    )
                    candidate_scores[candidate_id]["source_movies"].add(movie_name)

    # Build recommendation list
    recommendations = []
    for movie_id, info in candidate_scores.items():
        movie_attrs = G.nodes[movie_id]
        # Deduplicate shared attributes
        seen = set()
        unique_attrs = []
        for attr in info["shared_attributes"]:
            key = (attr["type"], attr["name"], attr["via_movie"])
            if key not in seen:
                seen.add(key)
                unique_attrs.append(attr)

        recommendations.append(
            {
                "movie_id": movie_id,
                "title": movie_attrs.get("name", movie_id),
                "year": movie_attrs.get("year"),
                "score": round(info["score"], 3),
                "num_shared_attributes": len(unique_attrs),
                "shared_attributes": unique_attrs,
                "source_movies": sorted(info["source_movies"]),
            }
        )

    recommendations.sort(key=lambda x: x["score"], reverse=True)
    return recommendations[:top_k]


def main():
    print("Loading movie knowledge graph...")
    G = load_graph()

    users = [n for n, d in G.nodes(data=True) if d.get("type") == "User"]
    print(f"Found {len(users)} users\n")

    target_user = "user_1"
    target_name = G.nodes[target_user]["name"]

    print("=" * 60)
    print(f"CONTENT-BASED RECOMMENDATIONS FOR: {target_name}")
    print("=" * 60)

    # Show liked movies and their attributes
    liked = get_user_liked_movies(G, target_user)
    print(f"\n{target_name}'s liked movies (rated >= 4.0):")
    for movie_id, score in sorted(liked.items(), key=lambda x: x[1], reverse=True):
        title = G.nodes[movie_id]["name"]
        attrs = get_movie_attributes(G, movie_id)
        genres = [G.nodes[g]["name"] for g in attrs["genres"]]
        actors = [G.nodes[a]["name"] for a in attrs["actors"]]
        directors = [G.nodes[d]["name"] for d in attrs["directors"]]
        print(f"\n  {title} ({score:.1f}/5.0)")
        print(f"    Genres: {', '.join(genres)}")
        print(f"    Actors: {', '.join(actors)}")
        print(f"    Directors: {', '.join(directors)}")

    # Generate recommendations
    print(f"\n--- Content-Based Recommendations ---")
    recs = recommend_content_based(G, target_user, top_k=10)
    if not recs:
        print("  No recommendations found.")
    else:
        for i, rec in enumerate(recs, 1):
            print(f"\n  {i}. {rec['title']} ({rec['year']})")
            print(f"     Score: {rec['score']:.3f}  ({rec['num_shared_attributes']} shared attributes)")
            print(f"     Connected via: {', '.join(rec['source_movies'])}")
            # Group shared attributes by type
            by_type: dict[str, list[str]] = defaultdict(list)
            for attr in rec["shared_attributes"]:
                by_type[attr["type"]].append(f"{attr['name']} (via {attr['via_movie']})")
            for attr_type, items in by_type.items():
                print(f"       Shared {attr_type}s: {'; '.join(items)}")


if __name__ == "__main__":
    main()
