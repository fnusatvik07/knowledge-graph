"""Generate explainable recommendations by tracing graph paths.

For each recommendation, trace the specific graph paths that led to it
and use an LLM to produce a natural-language explanation.

Example output:
  "Recommended because you liked 'Inception' which shares the genre
   'Sci-Fi' with 'Interstellar', and both are directed by Christopher Nolan."
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import networkx as nx

# ── Path setup ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared.llm_clients import chat_completion

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "movies_kg.json"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


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


def trace_recommendation_paths(
    G: nx.DiGraph,
    user_id: str,
    recommended_movie: str,
    min_rating: float = 4.0,
) -> list[dict]:
    """Trace all graph paths connecting the user to a recommended movie.

    Finds paths through shared attributes:
    - User -> liked Movie -> shared Genre <- recommended Movie
    - User -> liked Movie -> shared Actor <- recommended Movie
    - User -> liked Movie -> shared Director <- recommended Movie
    - User -> liked Movie -> SIMILAR_TO -> recommended Movie

    Returns:
        List of path descriptions with structured details.
    """
    user_ratings = get_user_ratings(G, user_id)
    liked_movies = {m: s for m, s in user_ratings.items() if s >= min_rating}
    user_name = G.nodes[user_id]["name"]
    rec_name = G.nodes[recommended_movie]["name"]

    paths = []

    # Get recommended movie's attributes
    rec_attrs: dict[str, set] = {"genres": set(), "actors": set(), "directors": set()}
    for target in G.successors(recommended_movie):
        edge_type = G.edges[recommended_movie, target].get("type")
        if edge_type == "HAS_GENRE":
            rec_attrs["genres"].add(target)
        elif edge_type == "HAS_ACTOR":
            rec_attrs["actors"].add(target)
        elif edge_type == "DIRECTED_BY":
            rec_attrs["directors"].add(target)

    for liked_movie_id, rating in liked_movies.items():
        liked_name = G.nodes[liked_movie_id]["name"]

        # Check shared genres
        for target in G.successors(liked_movie_id):
            edge_type = G.edges[liked_movie_id, target].get("type")

            if edge_type == "HAS_GENRE" and target in rec_attrs["genres"]:
                genre_name = G.nodes[target]["name"]
                paths.append({
                    "type": "shared_genre",
                    "liked_movie": liked_name,
                    "liked_rating": rating,
                    "shared_entity": genre_name,
                    "recommended_movie": rec_name,
                    "path": f"{user_name} --[rated {rating}]--> {liked_name} "
                            f"--[HAS_GENRE]--> {genre_name} "
                            f"<--[HAS_GENRE]-- {rec_name}",
                })

            elif edge_type == "HAS_ACTOR" and target in rec_attrs["actors"]:
                actor_name = G.nodes[target]["name"]
                paths.append({
                    "type": "shared_actor",
                    "liked_movie": liked_name,
                    "liked_rating": rating,
                    "shared_entity": actor_name,
                    "recommended_movie": rec_name,
                    "path": f"{user_name} --[rated {rating}]--> {liked_name} "
                            f"--[HAS_ACTOR]--> {actor_name} "
                            f"<--[HAS_ACTOR]-- {rec_name}",
                })

            elif edge_type == "DIRECTED_BY" and target in rec_attrs["directors"]:
                director_name = G.nodes[target]["name"]
                paths.append({
                    "type": "shared_director",
                    "liked_movie": liked_name,
                    "liked_rating": rating,
                    "shared_entity": director_name,
                    "recommended_movie": rec_name,
                    "path": f"{user_name} --[rated {rating}]--> {liked_name} "
                            f"--[DIRECTED_BY]--> {director_name} "
                            f"<--[DIRECTED_BY]-- {rec_name}",
                })

            elif edge_type == "SIMILAR_TO" and target == recommended_movie:
                paths.append({
                    "type": "similar_movie",
                    "liked_movie": liked_name,
                    "liked_rating": rating,
                    "shared_entity": None,
                    "recommended_movie": rec_name,
                    "path": f"{user_name} --[rated {rating}]--> {liked_name} "
                            f"--[SIMILAR_TO]--> {rec_name}",
                })

    return paths


def generate_natural_explanation(
    user_name: str,
    rec_title: str,
    paths: list[dict],
    provider: str = "openai",
    model: str | None = None,
) -> str:
    """Use LLM to generate a natural-language explanation from graph paths.

    Returns:
        A friendly, conversational explanation string.
    """
    if not paths:
        return f"We think you might enjoy '{rec_title}' based on your viewing history."

    # Build a structured summary of the paths
    path_summaries = []
    for p in paths:
        if p["type"] == "shared_genre":
            path_summaries.append(
                f"- You rated '{p['liked_movie']}' {p['liked_rating']}/5, and it shares "
                f"the genre '{p['shared_entity']}' with '{rec_title}'."
            )
        elif p["type"] == "shared_actor":
            path_summaries.append(
                f"- You rated '{p['liked_movie']}' {p['liked_rating']}/5, and "
                f"{p['shared_entity']} appears in both '{p['liked_movie']}' and '{rec_title}'."
            )
        elif p["type"] == "shared_director":
            path_summaries.append(
                f"- You rated '{p['liked_movie']}' {p['liked_rating']}/5, and both are "
                f"directed by {p['shared_entity']}."
            )
        elif p["type"] == "similar_movie":
            path_summaries.append(
                f"- You rated '{p['liked_movie']}' {p['liked_rating']}/5, and it is "
                f"considered similar to '{rec_title}'."
            )

    prompt = f"""You are a friendly movie recommendation assistant. Generate a concise,
natural-language explanation for why we are recommending the movie '{rec_title}'
to {user_name}.

Here are the specific connections we found in the knowledge graph:

{chr(10).join(path_summaries)}

Write a 2-3 sentence recommendation explanation that sounds natural and
conversational. Mention the most important connections (shared directors and
actors are stronger signals than shared genres). Do not use bullet points.
Start with "Recommended because" and keep it under 100 words."""

    response = chat_completion(
        prompt=prompt,
        system="You write concise, friendly movie recommendation explanations.",
        provider=provider,
        model=model,
    )
    return response.strip()


def get_content_based_recs(G: nx.DiGraph, user_id: str, top_k: int = 5) -> list[str]:
    """Simple content-based recommendations to have something to explain."""
    user_ratings = get_user_ratings(G, user_id)
    liked = {m: s for m, s in user_ratings.items() if s >= 4.0}
    all_rated = set(user_ratings)

    WEIGHTS = {"DIRECTED_BY": 3.0, "HAS_ACTOR": 2.0, "HAS_GENRE": 1.0}
    scores: dict[str, float] = defaultdict(float)

    for movie_id, rating in liked.items():
        boost = rating / 5.0
        for target in G.successors(movie_id):
            edge_type = G.edges[movie_id, target].get("type")
            if edge_type not in WEIGHTS:
                continue
            for pred in G.predecessors(target):
                if G.edges[pred, target].get("type") == edge_type and pred not in all_rated:
                    if G.nodes[pred].get("type") == "Movie":
                        scores[pred] += WEIGHTS[edge_type] * boost

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [m for m, _ in ranked[:top_k]]


def main():
    print("Loading movie knowledge graph...")
    G = load_graph()

    target_user = "user_1"
    target_name = G.nodes[target_user]["name"]

    print(f"\nGenerating explainable recommendations for {target_name}...\n")

    # Get top recommendations
    rec_movie_ids = get_content_based_recs(G, target_user, top_k=5)

    print("=" * 70)
    print(f"EXPLAINABLE RECOMMENDATIONS FOR: {target_name}")
    print("=" * 70)

    results = []

    for i, movie_id in enumerate(rec_movie_ids, 1):
        title = G.nodes[movie_id]["name"]
        year = G.nodes[movie_id].get("year", "")

        print(f"\n{'─' * 70}")
        print(f"  {i}. {title} ({year})")

        # Trace paths
        paths = trace_recommendation_paths(G, target_user, movie_id)

        print(f"\n  Graph paths ({len(paths)} connections found):")
        for p in paths:
            print(f"    {p['path']}")

        # Generate natural explanation
        print(f"\n  Generating natural-language explanation...")
        explanation = generate_natural_explanation(target_name, title, paths)
        print(f"\n  {explanation}")

        results.append({
            "rank": i,
            "movie": title,
            "year": year,
            "num_paths": len(paths),
            "paths": paths,
            "explanation": explanation,
        })

    # Save results
    output_path = OUTPUT_DIR / "explainable_recommendations.json"
    with open(output_path, "w") as f:
        json.dump({"user": target_name, "recommendations": results}, f, indent=2)
    print(f"\n\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
