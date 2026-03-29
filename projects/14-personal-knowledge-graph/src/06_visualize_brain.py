"""Step 6: Visualize the personal knowledge graph with pyvis.

Creates an interactive HTML visualization of your personal KG.
- Notes as large nodes, concepts as small nodes
- Color by topic cluster (community detection)
- Highlights suggested missing links in red dashed lines

Inputs:  output/personal_kg.graphml, output/discoveries.json (optional)
Outputs: output/personal_kg_viz.html
"""

import json
import sys
from pathlib import Path

import networkx as nx
from pyvis.network import Network

# Add projects dir to path for shared imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"

# ── Color palettes ───────────────────────────────────────────────────────
CLUSTER_COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
]

NOTE_DEFAULT_COLOR = "#4e79a7"
CONCEPT_DEFAULT_COLOR = "#cccccc"
SUGGESTED_LINK_COLOR = "#e15759"


def assign_cluster_colors(G: nx.DiGraph) -> dict[str, str]:
    """Assign colors to nodes based on community detection."""
    undirected = G.to_undirected()
    node_colors = {}

    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(undirected))

        for i, community in enumerate(communities):
            color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            for node in community:
                node_colors[node] = color
    except Exception:
        # Fallback: default colors
        for node in G.nodes():
            node_type = G.nodes[node].get("node_type", "")
            if node_type == "NOTE":
                node_colors[node] = NOTE_DEFAULT_COLOR
            else:
                node_colors[node] = CONCEPT_DEFAULT_COLOR

    return node_colors


def build_visualization(G: nx.DiGraph, suggested_links: list[dict] | None = None) -> Network:
    """Build the pyvis network visualization."""
    net = Network(
        height="800px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#333333",
        directed=True,
        notebook=False,
    )

    # Physics settings for a nice layout
    net.set_options("""
    {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 200,
                "springConstant": 0.08
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 150}
        },
        "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
        }
    }
    """)

    # Assign cluster colors
    node_colors = assign_cluster_colors(G)

    # Add nodes
    for node, data in G.nodes(data=True):
        node_type = data.get("node_type", "")
        label = data.get("label", node)
        color = node_colors.get(node, CONCEPT_DEFAULT_COLOR)

        if node_type == "NOTE":
            tags = data.get("tags", "")
            word_count = data.get("word_count", 0)
            title_hover = f"<b>{label}</b><br>Tags: {tags}<br>Words: {word_count}"
            net.add_node(
                node,
                label=label,
                title=title_hover,
                color=color,
                size=30,
                shape="dot",
                font={"size": 16, "face": "Arial"},
                borderWidth=2,
            )
        elif node_type == "CONCEPT":
            mentioned_count = data.get("mentioned_count", 0)
            category = data.get("category", "")
            title_hover = f"<b>{label}</b><br>Category: {category}<br>Mentioned in: {mentioned_count} notes"
            net.add_node(
                node,
                label=label,
                title=title_hover,
                color=color,
                size=12,
                shape="dot",
                font={"size": 10, "face": "Arial", "color": "#666666"},
                borderWidth=1,
            )

    # Add edges
    for source, target, data in G.edges(data=True):
        edge_type = data.get("edge_type", "")

        if edge_type == "NOTE_LINKS_TO":
            net.add_edge(source, target, color="#333333", width=2, title="links to")
        elif edge_type == "MENTIONS_CONCEPT":
            relevance = data.get("relevance", "mentioned")
            width = 1.5 if relevance == "primary" else 0.5
            net.add_edge(source, target, color="#999999", width=width, title=f"mentions ({relevance})", dashes=True)
        elif edge_type == "CONCEPT_RELATED_TO":
            net.add_edge(source, target, color="#cccccc", width=0.5, title="related to", dashes=[5, 5])

    # Add suggested missing links (in red dashed lines)
    if suggested_links:
        for link in suggested_links:
            source = f"note:{link['note_a']}"
            target = f"note:{link['note_b']}"
            if source in G and target in G:
                shared = ", ".join(link.get("shared_concepts", [])[:3])
                net.add_edge(
                    source,
                    target,
                    color=SUGGESTED_LINK_COLOR,
                    width=2,
                    title=f"Suggested link (shared: {shared})",
                    dashes=[10, 10],
                )

    return net


def main():
    # Load graph
    graph_path = OUTPUT_DIR / "personal_kg.graphml"
    if not graph_path.exists():
        print("Run 03_build_personal_kg.py first.")
        return

    G = nx.read_graphml(str(graph_path))
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Load discoveries (optional)
    suggested_links = None
    discoveries_path = OUTPUT_DIR / "discoveries.json"
    if discoveries_path.exists():
        discoveries = json.loads(discoveries_path.read_text(encoding="utf-8"))
        suggested_links = discoveries.get("suggested_links", [])
        print(f"Loaded {len(suggested_links)} suggested links")

    # Build visualization
    print("Building visualization...")
    net = build_visualization(G, suggested_links)

    # Save
    output_path = OUTPUT_DIR / "personal_kg_viz.html"
    net.save_graph(str(output_path))
    print(f"\nSaved visualization to {output_path}")
    print("Open in a browser to explore your personal knowledge graph!")


if __name__ == "__main__":
    main()
