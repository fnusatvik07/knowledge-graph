"""Step 4: Visualize the knowledge graph.

Creates two visualizations:
1. Static matplotlib plot (saved as PNG)
2. Interactive pyvis HTML visualization
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

OUTPUT_DIR = Path(__file__).parent.parent / "output"

# Color scheme for entity types
TYPE_COLORS = {
    "PERSON": "#FF6B6B",
    "ORGANIZATION": "#4ECDC4",
    "TECHNOLOGY": "#45B7D1",
    "CONCEPT": "#96CEB4",
    "EVENT": "#FFEAA7",
    "UNKNOWN": "#DFE6E9",
}


def visualize_matplotlib(G: nx.DiGraph, output_path: Path):
    """Create a static matplotlib visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(20, 14))

    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Node colors based on type
    node_colors = [TYPE_COLORS.get(G.nodes[n].get("type", "UNKNOWN"), "#DFE6E9") for n in G.nodes()]

    # Node sizes based on degree
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1
    node_sizes = [300 + (degrees[n] / max_degree) * 2000 for n in G.nodes()]

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#B2BEC3", arrows=True,
                           arrowsize=15, alpha=0.6, width=1.5,
                           connectionstyle="arc3,rad=0.1")

    # Draw edge labels
    edge_labels = {(u, v): d["relation_type"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                  font_size=6, alpha=0.7)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.9, edgecolors="#2D3436",
                           linewidths=1.5)

    # Draw labels
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_weight="bold")

    # Legend
    legend_patches = [mpatches.Patch(color=color, label=entity_type)
                      for entity_type, color in TYPE_COLORS.items()
                      if entity_type != "UNKNOWN"]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=10,
              title="Entity Types", title_fontsize=12)

    ax.set_title("Knowledge Graph", fontsize=16, fontweight="bold", pad=20)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Static visualization saved to: {output_path}")


def visualize_pyvis(G: nx.DiGraph, output_path: Path):
    """Create an interactive pyvis visualization."""
    from pyvis.network import Network

    net = Network(height="800px", width="100%", bgcolor="#ffffff",
                  font_color="#2D3436", directed=True)

    # Physics settings for better layout
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=200)

    # Add nodes
    degrees = dict(G.degree())
    for node, data in G.nodes(data=True):
        entity_type = data.get("type", "UNKNOWN")
        color = TYPE_COLORS.get(entity_type, "#DFE6E9")
        size = 15 + degrees.get(node, 0) * 5
        title = f"<b>{node}</b><br>Type: {entity_type}<br>{data.get('description', '')}"
        net.add_node(node, label=node, color=color, size=size, title=title)

    # Add edges
    for source, target, data in G.edges(data=True):
        rel_type = data.get("relation_type", "RELATED_TO")
        title = data.get("description", rel_type)
        net.add_edge(source, target, label=rel_type, title=title, color="#B2BEC3")

    net.save_graph(str(output_path))
    print(f"Interactive visualization saved to: {output_path}")
    print("  Open this HTML file in a browser to explore the graph interactively!")


def main():
    graphml_path = OUTPUT_DIR / "knowledge_graph.graphml"
    if not graphml_path.exists():
        print("Error: Run 03_build_graph.py first!")
        return

    G = nx.read_graphml(str(graphml_path))
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Static visualization
    visualize_matplotlib(G, OUTPUT_DIR / "knowledge_graph.png")

    # Interactive visualization
    visualize_pyvis(G, OUTPUT_DIR / "knowledge_graph.html")


if __name__ == "__main__":
    main()
