"""Build a NetworkX graph from the movie KG dataset.

Loads entities and relationships from movies_kg.json, constructs a typed
multi-graph, and prints statistics about nodes, edges, and the most
connected entities of each type.
"""

import json
import sys
from pathlib import Path
from collections import Counter

import networkx as nx

# ── Path setup ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "movies_kg.json"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_movie_data() -> dict:
    """Load the movie KG dataset from JSON."""
    with open(DATA_FILE) as f:
        return json.load(f)


def build_graph(data: dict) -> nx.DiGraph:
    """Build a directed graph from entities and relationships.

    Returns:
        A NetworkX DiGraph with typed nodes and edges.
    """
    G = nx.DiGraph()

    # Add nodes with their attributes
    for entity in data["entities"]:
        node_attrs = {k: v for k, v in entity.items() if k != "id"}
        G.add_node(entity["id"], **node_attrs)

    # Add edges with their attributes
    for rel in data["relationships"]:
        edge_attrs = {k: v for k, v in rel.items() if k not in ("source", "target")}
        G.add_edge(rel["source"], rel["target"], **edge_attrs)

    return G


def print_graph_stats(G: nx.DiGraph) -> None:
    """Print comprehensive statistics about the graph."""
    print("=" * 60)
    print("MOVIE KNOWLEDGE GRAPH STATISTICS")
    print("=" * 60)

    print(f"\nTotal nodes: {G.number_of_nodes()}")
    print(f"Total edges: {G.number_of_edges()}")

    # Count nodes by type
    type_counts = Counter()
    for _, attrs in G.nodes(data=True):
        type_counts[attrs.get("type", "Unknown")] += 1

    print("\n--- Nodes by Type ---")
    for node_type, count in sorted(type_counts.items()):
        print(f"  {node_type:12s}: {count}")

    # Count edges by type
    edge_type_counts = Counter()
    for _, _, attrs in G.edges(data=True):
        edge_type_counts[attrs.get("type", "Unknown")] += 1

    print("\n--- Edges by Type ---")
    for edge_type, count in sorted(edge_type_counts.items()):
        print(f"  {edge_type:15s}: {count}")


def print_most_connected(G: nx.DiGraph) -> None:
    """Print the most connected entities of each type."""
    print("\n--- Most Connected Entities ---")

    # Group nodes by type
    nodes_by_type: dict[str, list] = {}
    for node_id, attrs in G.nodes(data=True):
        node_type = attrs.get("type", "Unknown")
        nodes_by_type.setdefault(node_type, []).append(node_id)

    for node_type in sorted(nodes_by_type):
        nodes = nodes_by_type[node_type]
        # Total degree (in + out) for each node
        degrees = [(n, G.degree(n)) for n in nodes]
        degrees.sort(key=lambda x: x[1], reverse=True)
        top = degrees[:5]

        print(f"\n  Top {node_type}s (by total connections):")
        for node_id, degree in top:
            name = G.nodes[node_id].get("name", node_id)
            print(f"    {name:40s}  degree={degree}")


def print_sample_paths(G: nx.DiGraph) -> None:
    """Print a few interesting paths through the graph."""
    print("\n--- Sample Paths ---")

    # Find a user -> movie -> genre path
    users = [n for n, d in G.nodes(data=True) if d.get("type") == "User"]
    if users:
        user = users[0]
        user_name = G.nodes[user]["name"]
        movies_rated = [
            (t, G.edges[user, t])
            for t in G.successors(user)
            if G.edges[user, t].get("type") == "RATED"
        ]
        if movies_rated:
            movie_id, edge_data = movies_rated[0]
            movie_name = G.nodes[movie_id]["name"]
            score = edge_data.get("score", "?")
            print(f"\n  {user_name} --[RATED {score}]--> {movie_name}")

            genres = [
                G.nodes[t]["name"]
                for t in G.successors(movie_id)
                if G.edges[movie_id, t].get("type") == "HAS_GENRE"
            ]
            for genre in genres:
                print(f"    {movie_name} --[HAS_GENRE]--> {genre}")

            actors = [
                G.nodes[t]["name"]
                for t in G.successors(movie_id)
                if G.edges[movie_id, t].get("type") == "HAS_ACTOR"
            ]
            for actor in actors:
                print(f"    {movie_name} --[HAS_ACTOR]--> {actor}")

            directors = [
                G.nodes[t]["name"]
                for t in G.successors(movie_id)
                if G.edges[movie_id, t].get("type") == "DIRECTED_BY"
            ]
            for director in directors:
                print(f"    {movie_name} --[DIRECTED_BY]--> {director}")


def save_graph(G: nx.DiGraph) -> None:
    """Save the graph as JSON for other scripts to load."""
    data = nx.node_link_data(G)
    output_path = OUTPUT_DIR / "movie_kg_graph.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nGraph saved to {output_path}")


def main():
    print("Loading movie KG data...")
    data = load_movie_data()
    print(f"  Entities: {len(data['entities'])}")
    print(f"  Relationships: {len(data['relationships'])}")

    print("\nBuilding NetworkX graph...")
    G = build_graph(data)

    print_graph_stats(G)
    print_most_connected(G)
    print_sample_paths(G)
    save_graph(G)


if __name__ == "__main__":
    main()
