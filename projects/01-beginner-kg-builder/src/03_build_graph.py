"""Step 3: Build a NetworkX graph from extracted entities and relationships.

Reads entities.json and relationships.json, constructs a directed graph,
and saves it as GraphML to output/knowledge_graph.graphml.
"""

import json
from pathlib import Path

import networkx as nx

OUTPUT_DIR = Path(__file__).parent.parent / "output"


def build_knowledge_graph(entities: list[dict], relationships: list[dict]) -> nx.DiGraph:
    """Build a NetworkX directed graph from entities and relationships."""
    G = nx.DiGraph()

    # Add entity nodes
    for entity in entities:
        G.add_node(
            entity["name"],
            type=entity["type"],
            description=entity["description"],
            sources=", ".join(entity.get("sources", [])),
        )

    # Build a lookup for case-insensitive matching
    node_lookup = {name.lower(): name for name in G.nodes()}

    # Add relationship edges
    added = 0
    skipped = 0
    for rel in relationships:
        source_key = rel["source"].strip().lower()
        target_key = rel["target"].strip().lower()

        source = node_lookup.get(source_key)
        target = node_lookup.get(target_key)

        if source and target:
            G.add_edge(
                source,
                target,
                relation_type=rel["relation_type"],
                description=rel.get("description", ""),
            )
            added += 1
        else:
            skipped += 1

    return G, added, skipped


def print_graph_stats(G: nx.DiGraph):
    """Print summary statistics about the graph."""
    print(f"\nGraph Statistics:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")

    if G.number_of_nodes() > 0:
        print(f"  Weakly connected components: {nx.number_weakly_connected_components(G)}")
        print(f"  Density: {nx.density(G):.4f}")

        # Node type distribution
        type_counts = {}
        for _, data in G.nodes(data=True):
            t = data.get("type", "UNKNOWN")
            type_counts[t] = type_counts.get(t, 0) + 1
        print(f"\n  Node types:")
        for t, count in sorted(type_counts.items()):
            print(f"    {t}: {count}")

        # Most connected nodes
        degree_sorted = sorted(G.degree(), key=lambda x: x[1], reverse=True)
        print(f"\n  Most connected nodes:")
        for name, degree in degree_sorted[:10]:
            print(f"    {name}: {degree} connections")


def main():
    entities_path = OUTPUT_DIR / "entities.json"
    relationships_path = OUTPUT_DIR / "relationships.json"

    if not entities_path.exists() or not relationships_path.exists():
        print("Error: Run 01_extract_entities.py and 02_extract_relationships.py first!")
        return

    with open(entities_path) as f:
        entities = json.load(f)
    with open(relationships_path) as f:
        relationships = json.load(f)

    print(f"Building graph from {len(entities)} entities and {len(relationships)} relationships...")

    G, added, skipped = build_knowledge_graph(entities, relationships)
    print(f"  Edges added: {added}, skipped (missing nodes): {skipped}")

    print_graph_stats(G)

    # Save as GraphML
    graphml_path = OUTPUT_DIR / "knowledge_graph.graphml"
    nx.write_graphml(G, str(graphml_path))
    print(f"\nGraph saved to: {graphml_path}")

    # Also save as adjacency list for quick inspection
    adj_path = OUTPUT_DIR / "graph_adjacency.txt"
    with open(adj_path, "w") as f:
        for source, target, data in G.edges(data=True):
            f.write(f"{source} --[{data['relation_type']}]--> {target}\n")
    print(f"Adjacency list saved to: {adj_path}")


if __name__ == "__main__":
    main()
