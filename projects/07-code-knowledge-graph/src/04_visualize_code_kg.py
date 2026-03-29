"""Visualize the code knowledge graph with pyvis.

Color-codes nodes by entity type:
  - Module: blue
  - Class: green
  - Function: orange
  - Import: purple

Shows call hierarchy, containment, and inheritance relationships.

Usage:
    python src/04_visualize_code_kg.py
    python src/04_visualize_code_kg.py --input output/code_kg.json
    python src/04_visualize_code_kg.py --layout hierarchical

References:
    - pyvis docs: https://pyvis.readthedocs.io/en/latest/
    - NetworkX visualization: https://networkx.org/documentation/stable/reference/drawing.html
    - LangChain graph visualization: https://python.langchain.com/docs/integrations/graphs/
"""

import argparse
import json
import sys
from pathlib import Path

# ── Shared imports ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import networkx as nx
from pyvis.network import Network

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"

# ── Color scheme ─────────────────────────────────────────────────────────
ENTITY_COLORS = {
    "Module": "#4A90D9",     # Blue
    "Class": "#27AE60",      # Green
    "Function": "#E67E22",   # Orange
    "Import": "#8E44AD",     # Purple
    "Unknown": "#95A5A6",    # Gray
}

ENTITY_SHAPES = {
    "Module": "box",
    "Class": "diamond",
    "Function": "dot",
    "Import": "triangle",
}

RELATIONSHIP_COLORS = {
    "CALLS": "#E74C3C",        # Red
    "IMPORTS": "#3498DB",      # Blue
    "DEFINES": "#2ECC71",      # Green
    "CONTAINS": "#95A5A6",     # Gray
    "INHERITS_FROM": "#F39C12", # Yellow
}


def build_pyvis_network(G: nx.DiGraph, layout: str = "physics") -> Network:
    """Build a pyvis Network from the NetworkX graph.

    Args:
        G: The code knowledge graph.
        layout: Layout mode - 'physics' (force-directed) or 'hierarchical'.

    Returns:
        A pyvis Network ready for rendering.
    """
    height = "800px"
    width = "100%"

    net = Network(
        height=height,
        width=width,
        directed=True,
        notebook=False,
        bgcolor="#FFFFFF",
        font_color="#333333",
    )

    # Configure layout
    if layout == "hierarchical":
        net.set_options("""
        {
            "layout": {
                "hierarchical": {
                    "enabled": true,
                    "direction": "UD",
                    "sortMethod": "hubsize",
                    "nodeSpacing": 200,
                    "levelSeparation": 150
                }
            },
            "physics": {
                "hierarchicalRepulsion": {
                    "centralGravity": 0.0,
                    "springLength": 200,
                    "springConstant": 0.01,
                    "nodeDistance": 200
                }
            }
        }
        """)
    else:
        net.set_options("""
        {
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.005,
                    "springLength": 200,
                    "springConstant": 0.08
                },
                "solver": "forceAtlas2Based",
                "stabilization": {
                    "iterations": 150
                }
            }
        }
        """)

    # Add nodes
    for node_id, data in G.nodes(data=True):
        entity_type = data.get("entity_type", "Unknown")
        name = data.get("name", node_id)
        color = ENTITY_COLORS.get(entity_type, ENTITY_COLORS["Unknown"])
        shape = ENTITY_SHAPES.get(entity_type, "dot")

        # Build tooltip with metadata
        tooltip_parts = [f"<b>{name}</b>", f"Type: {entity_type}"]
        if data.get("module"):
            tooltip_parts.append(f"Module: {data['module']}")
        if data.get("language"):
            tooltip_parts.append(f"Language: {data['language']}")
        if data.get("line_start"):
            tooltip_parts.append(f"Lines: {data['line_start']}-{data.get('line_end', '?')}")
        if data.get("params"):
            params = data["params"]
            if isinstance(params, str):
                params = json.loads(params) if params.startswith("[") else [params]
            tooltip_parts.append(f"Params: {', '.join(str(p) for p in params[:5])}")
        if data.get("docstring"):
            doc = str(data["docstring"])[:100]
            tooltip_parts.append(f"Doc: {doc}")
        tooltip = "<br>".join(tooltip_parts)

        # Size based on connectivity
        degree = G.degree(node_id)
        size = max(15, min(50, 15 + degree * 3))

        # Label: show short name, not full qualified path
        label = name if len(name) < 30 else name[:27] + "..."

        net.add_node(
            node_id,
            label=label,
            color=color,
            shape=shape,
            size=size,
            title=tooltip,
        )

    # Add edges
    for source, target, data in G.edges(data=True):
        relationship = data.get("relationship", "UNKNOWN")
        color = RELATIONSHIP_COLORS.get(relationship, "#BDC3C7")

        # Thinner lines for CONTAINS, thicker for CALLS
        width = 1
        if relationship == "CALLS":
            width = 2
        elif relationship == "INHERITS_FROM":
            width = 3
        elif relationship == "CONTAINS":
            width = 0.5

        # Dashes for CONTAINS edges (less visually prominent)
        dashes = relationship == "CONTAINS"

        net.add_edge(
            source,
            target,
            title=relationship,
            color=color,
            width=width,
            dashes=dashes,
            arrows="to",
        )

    return net


def add_legend(net: Network):
    """Add a color legend to the network visualization.

    Creates invisible fixed nodes in the corner to serve as a legend.
    """
    x_pos = -500
    y_pos = -400
    step = 50

    for i, (entity_type, color) in enumerate(ENTITY_COLORS.items()):
        if entity_type == "Unknown":
            continue
        net.add_node(
            f"legend_{entity_type}",
            label=f"  {entity_type}",
            color=color,
            shape=ENTITY_SHAPES.get(entity_type, "dot"),
            size=15,
            x=x_pos,
            y=y_pos + i * step,
            fixed=True,
            physics=False,
        )


def main():
    parser = argparse.ArgumentParser(description="Visualize the code knowledge graph.")
    parser.add_argument("--input", type=str, default=None, help="Path to code_kg.json")
    parser.add_argument("--layout", type=str, default="physics",
                        choices=["physics", "hierarchical"],
                        help="Graph layout algorithm")
    parser.add_argument("--output", type=str, default=None, help="Output HTML filename")
    args = parser.parse_args()

    # Load the graph
    input_file = Path(args.input) if args.input else OUTPUT_DIR / "code_kg.json"
    if not input_file.exists():
        print(f"ERROR: Graph file not found at {input_file}")
        print("Run 03_build_code_kg.py first.")
        sys.exit(1)

    print(f"Loading code knowledge graph from {input_file}...")
    with open(input_file) as f:
        json_data = json.load(f)

    G = nx.node_link_graph(json_data)
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Build visualization
    print(f"Building visualization (layout: {args.layout})...")
    net = build_pyvis_network(G, layout=args.layout)
    add_legend(net)

    # Save HTML
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = args.output or str(OUTPUT_DIR / "code_kg_visualization.html")
    net.save_graph(output_file)

    print(f"\nVisualization saved to {output_file}")
    print(f"Open in a browser to explore the interactive graph.")

    # Print color legend
    print(f"\nColor Legend:")
    for entity_type, color in ENTITY_COLORS.items():
        if entity_type != "Unknown":
            print(f"  {color} = {entity_type}")
    print(f"\nEdge Colors:")
    for rel, color in RELATIONSHIP_COLORS.items():
        print(f"  {color} = {rel}")


if __name__ == "__main__":
    main()
