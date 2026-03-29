"""Visualize the recommendation subgraph with pyvis.

Shows the target user, their liked movies, shared attributes (genres,
actors, directors), and recommended movies. Entities are color-coded
by type and recommendation paths are highlighted.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import networkx as nx
from pyvis.network import Network

# ── Path setup ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "movies_kg.json"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Color palette for entity types
COLORS = {
    "User": "#e74c3c",       # red
    "Movie": "#3498db",      # blue
    "Genre": "#2ecc71",      # green
    "Actor": "#f39c12",      # orange
    "Director": "#9b59b6",   # purple
    "Recommended": "#e91e63",  # pink (for recommended movies)
}

NODE_SIZES = {
    "User": 35,
    "Movie": 25,
    "Genre": 20,
    "Actor": 20,
    "Director": 20,
}


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


def get_content_recs(G: nx.DiGraph, user_id: str, top_k: int = 5) -> list[str]:
    """Get content-based recommendations (simplified)."""
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


def build_recommendation_subgraph(
    G: nx.DiGraph,
    user_id: str,
    recommended_movies: list[str],
) -> nx.DiGraph:
    """Extract the subgraph relevant to the recommendations.

    Includes:
    - Target user
    - User's liked movies
    - Recommended movies
    - Shared attributes connecting liked and recommended movies
    """
    subgraph = nx.DiGraph()
    user_ratings = get_user_ratings(G, user_id)
    liked = {m for m, s in user_ratings.items() if s >= 4.0}
    rec_set = set(recommended_movies)

    # Add user node
    subgraph.add_node(user_id, **G.nodes[user_id], is_target=True)

    # Add liked movies and RATED edges
    for movie_id in liked:
        subgraph.add_node(movie_id, **G.nodes[movie_id])
        score = user_ratings[movie_id]
        subgraph.add_edge(user_id, movie_id, type="RATED", score=score)

    # Add recommended movies
    for movie_id in rec_set:
        attrs = dict(G.nodes[movie_id])
        attrs["is_recommended"] = True
        subgraph.add_node(movie_id, **attrs)

    # Find shared attributes between liked and recommended movies
    for liked_movie in liked:
        for target in G.successors(liked_movie):
            edge_type = G.edges[liked_movie, target].get("type")
            if edge_type not in ("HAS_GENRE", "HAS_ACTOR", "DIRECTED_BY"):
                continue

            # Check if any recommended movie shares this attribute
            for pred in G.predecessors(target):
                if pred in rec_set and G.edges[pred, target].get("type") == edge_type:
                    # Add the shared attribute node and edges
                    subgraph.add_node(target, **G.nodes[target])
                    subgraph.add_edge(liked_movie, target, type=edge_type)
                    subgraph.add_edge(pred, target, type=edge_type, is_rec_path=True)

    return subgraph


def visualize_subgraph(
    subgraph: nx.DiGraph,
    user_name: str,
    output_filename: str = "recommendation_graph.html",
) -> str:
    """Create an interactive pyvis visualization.

    Returns:
        Path to the saved HTML file.
    """
    net = Network(
        height="700px",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="#1a1a2e",
        font_color="white",
    )

    # Physics settings for readable layout
    net.set_options("""
    {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -100,
                "centralGravity": 0.01,
                "springLength": 150,
                "springConstant": 0.05
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 200}
        },
        "edges": {
            "smooth": {"type": "curvedCW", "roundness": 0.1}
        }
    }
    """)

    # Add nodes
    for node_id, attrs in subgraph.nodes(data=True):
        node_type = attrs.get("type", "Unknown")
        name = attrs.get("name", node_id)
        is_recommended = attrs.get("is_recommended", False)
        is_target = attrs.get("is_target", False)

        if is_recommended:
            color = COLORS["Recommended"]
            label = f"* {name} *"
            size = 30
            shape = "star"
        elif is_target:
            color = COLORS["User"]
            label = f"[{name}]"
            size = 40
            shape = "diamond"
        else:
            color = COLORS.get(node_type, "#95a5a6")
            label = name
            size = NODE_SIZES.get(node_type, 20)
            shape = "dot"

        year = attrs.get("year", "")
        title_text = f"<b>{name}</b><br>Type: {node_type}"
        if year:
            title_text += f"<br>Year: {year}"
        if is_recommended:
            title_text += "<br><b>RECOMMENDED</b>"

        net.add_node(
            node_id,
            label=label,
            color=color,
            size=size,
            shape=shape,
            title=title_text,
        )

    # Add edges
    for src, tgt, attrs in subgraph.edges(data=True):
        edge_type = attrs.get("type", "")
        is_rec_path = attrs.get("is_rec_path", False)
        score = attrs.get("score")

        label = edge_type
        if score:
            label = f"{edge_type} ({score})"

        color = "#e91e63" if is_rec_path else "#7f8c8d"
        width = 3 if is_rec_path else 1.5
        dashes = True if is_rec_path else False

        net.add_edge(
            src,
            tgt,
            label=label,
            color=color,
            width=width,
            dashes=dashes,
            arrows="to",
        )

    output_path = OUTPUT_DIR / output_filename
    net.save_graph(str(output_path))
    return str(output_path)


def main():
    print("Loading movie knowledge graph...")
    G = load_graph()

    target_user = "user_1"
    target_name = G.nodes[target_user]["name"]

    print(f"\nGenerating recommendations for {target_name}...")
    rec_movies = get_content_recs(G, target_user, top_k=5)

    print(f"\nTop recommendations:")
    for i, movie_id in enumerate(rec_movies, 1):
        print(f"  {i}. {G.nodes[movie_id]['name']} ({G.nodes[movie_id].get('year', '')})")

    print("\nBuilding recommendation subgraph...")
    subgraph = build_recommendation_subgraph(G, target_user, rec_movies)
    print(f"  Subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")

    print("\nGenerating interactive visualization...")
    output_path = visualize_subgraph(subgraph, target_name)
    print(f"  Saved to {output_path}")

    # Also create a legend / summary
    print("\n--- Visualization Legend ---")
    print(f"  Diamond (red)   = Target user ({target_name})")
    print(f"  Star (pink)     = Recommended movies")
    print(f"  Circle (blue)   = Liked movies")
    print(f"  Circle (green)  = Genres")
    print(f"  Circle (orange) = Actors")
    print(f"  Circle (purple) = Directors")
    print(f"  Dashed edges    = Recommendation paths")
    print(f"  Solid edges     = Known connections")

    # Save subgraph data
    subgraph_data = {
        "user": target_name,
        "num_nodes": subgraph.number_of_nodes(),
        "num_edges": subgraph.number_of_edges(),
        "recommended_movies": [G.nodes[m]["name"] for m in rec_movies],
        "visualization": output_path,
    }
    summary_path = OUTPUT_DIR / "visualization_summary.json"
    with open(summary_path, "w") as f:
        json.dump(subgraph_data, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
