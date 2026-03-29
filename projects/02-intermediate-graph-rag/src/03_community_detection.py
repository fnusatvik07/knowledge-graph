"""Step 3: Community detection on the knowledge graph.

Loads entity-relationship data from GraphRAG output (parquet files),
builds a NetworkX graph, applies the Leiden algorithm for community
detection, and visualizes the communities with distinct colors.
"""

import json
import sys
from pathlib import Path

# Add projects dir to path for shared imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

PROJECT_DIR = Path(__file__).resolve().parent.parent
GRAPHRAG_DIR = PROJECT_DIR / "output" / "graphrag"
OUTPUT_DIR = PROJECT_DIR / "output"


def load_entities_and_relationships():
    """Load entity and relationship data from GraphRAG parquet output.

    Falls back to building a sample graph from corpus if GraphRAG
    output is not available.
    """
    entities = []
    relationships = []

    # Try loading from GraphRAG parquet output
    try:
        import pandas as pd

        # GraphRAG stores output in parquet files
        # Look for entities and relationships in common output locations
        parquet_dirs = [
            GRAPHRAG_DIR / "output",
            GRAPHRAG_DIR / "artifacts",
            GRAPHRAG_DIR,
        ]

        entity_file = None
        rel_file = None

        for d in parquet_dirs:
            if not d.exists():
                continue
            for f in d.rglob("*.parquet"):
                fname = f.stem.lower()
                if "entit" in fname and entity_file is None:
                    entity_file = f
                if "relation" in fname and rel_file is None:
                    rel_file = f

        if entity_file:
            print(f"Loading entities from: {entity_file}")
            df_entities = pd.read_parquet(entity_file)
            print(f"  Found {len(df_entities)} entities")
            # Extract entity names and types
            for _, row in df_entities.iterrows():
                name = row.get("name", row.get("title", str(row.iloc[0])))
                etype = row.get("type", row.get("entity_type", "UNKNOWN"))
                entities.append({"name": str(name), "type": str(etype)})

        if rel_file:
            print(f"Loading relationships from: {rel_file}")
            df_rels = pd.read_parquet(rel_file)
            print(f"  Found {len(df_rels)} relationships")
            for _, row in df_rels.iterrows():
                source = row.get("source", row.get("head", str(row.iloc[0])))
                target = row.get("target", row.get("tail", str(row.iloc[1])))
                rel_type = row.get("type", row.get("relationship", "RELATED_TO"))
                relationships.append({
                    "source": str(source),
                    "target": str(target),
                    "type": str(rel_type),
                })

        if entities and relationships:
            return entities, relationships

    except ImportError:
        print("pandas not available, using fallback sample data")
    except Exception as e:
        print(f"Could not load GraphRAG output: {e}")

    # Fallback: build a sample graph from known corpus topics
    print("\nUsing sample entity-relationship data from corpus topics...")
    entities, relationships = _build_sample_graph()
    return entities, relationships


def _build_sample_graph():
    """Build a representative sample graph based on the corpus topics."""
    entities = [
        # Climate Change entities
        {"name": "IPCC", "type": "ORGANIZATION"},
        {"name": "Paris Agreement", "type": "EVENT"},
        {"name": "UNFCCC", "type": "ORGANIZATION"},
        {"name": "Carbon Dioxide", "type": "CONCEPT"},
        {"name": "Greenhouse Gases", "type": "CONCEPT"},
        {"name": "Global Warming", "type": "CONCEPT"},
        {"name": "Arctic", "type": "LOCATION"},
        # Renewable Energy entities
        {"name": "IEA", "type": "ORGANIZATION"},
        {"name": "IRENA", "type": "ORGANIZATION"},
        {"name": "Solar PV", "type": "TECHNOLOGY"},
        {"name": "Wind Energy", "type": "TECHNOLOGY"},
        {"name": "Green Hydrogen", "type": "TECHNOLOGY"},
        {"name": "Lithium-Ion Batteries", "type": "TECHNOLOGY"},
        {"name": "Inflation Reduction Act", "type": "EVENT"},
        {"name": "Tesla Megapack", "type": "TECHNOLOGY"},
        # EV entities
        {"name": "Tesla", "type": "ORGANIZATION"},
        {"name": "BYD", "type": "ORGANIZATION"},
        {"name": "Electric Vehicles", "type": "TECHNOLOGY"},
        {"name": "Solid-State Batteries", "type": "TECHNOLOGY"},
        {"name": "Vehicle-to-Grid", "type": "TECHNOLOGY"},
        {"name": "NACS", "type": "TECHNOLOGY"},
        {"name": "Elon Musk", "type": "PERSON"},
        # Carbon Capture entities
        {"name": "Climeworks", "type": "ORGANIZATION"},
        {"name": "Carbon Engineering", "type": "ORGANIZATION"},
        {"name": "Direct Air Capture", "type": "TECHNOLOGY"},
        {"name": "CCS", "type": "TECHNOLOGY"},
        {"name": "BECCS", "type": "TECHNOLOGY"},
        {"name": "45Q Tax Credit", "type": "EVENT"},
        {"name": "Sleipner Project", "type": "TECHNOLOGY"},
        # Agriculture entities
        {"name": "Regenerative Agriculture", "type": "CONCEPT"},
        {"name": "Soil Carbon Sequestration", "type": "CONCEPT"},
        {"name": "Precision Agriculture", "type": "TECHNOLOGY"},
        {"name": "Agrivoltaics", "type": "TECHNOLOGY"},
        {"name": "Rodale Institute", "type": "ORGANIZATION"},
        {"name": "FAO", "type": "ORGANIZATION"},
        {"name": "4 per 1000 Initiative", "type": "EVENT"},
    ]

    relationships = [
        # Climate cross-connections
        {"source": "IPCC", "target": "Climate Change", "type": "STUDIES"},
        {"source": "Paris Agreement", "target": "Global Warming", "type": "ADDRESSES"},
        {"source": "UNFCCC", "target": "Paris Agreement", "type": "ADOPTED"},
        {"source": "Greenhouse Gases", "target": "Global Warming", "type": "CAUSES"},
        {"source": "Carbon Dioxide", "target": "Greenhouse Gases", "type": "IS_TYPE_OF"},
        # Renewable energy connections
        {"source": "Solar PV", "target": "Carbon Dioxide", "type": "REDUCES"},
        {"source": "Wind Energy", "target": "Carbon Dioxide", "type": "REDUCES"},
        {"source": "IEA", "target": "Solar PV", "type": "TRACKS"},
        {"source": "IRENA", "target": "Wind Energy", "type": "TRACKS"},
        {"source": "Green Hydrogen", "target": "Solar PV", "type": "USES"},
        {"source": "Green Hydrogen", "target": "Wind Energy", "type": "USES"},
        {"source": "Lithium-Ion Batteries", "target": "Solar PV", "type": "ENABLES_STORAGE"},
        {"source": "Inflation Reduction Act", "target": "Solar PV", "type": "INCENTIVIZES"},
        {"source": "Inflation Reduction Act", "target": "Green Hydrogen", "type": "INCENTIVIZES"},
        {"source": "Tesla Megapack", "target": "Lithium-Ion Batteries", "type": "USES"},
        # EV connections
        {"source": "Tesla", "target": "Electric Vehicles", "type": "MANUFACTURES"},
        {"source": "BYD", "target": "Electric Vehicles", "type": "MANUFACTURES"},
        {"source": "Elon Musk", "target": "Tesla", "type": "LEADS"},
        {"source": "Electric Vehicles", "target": "Lithium-Ion Batteries", "type": "USES"},
        {"source": "Solid-State Batteries", "target": "Electric Vehicles", "type": "IMPROVES"},
        {"source": "Electric Vehicles", "target": "Carbon Dioxide", "type": "REDUCES"},
        {"source": "Vehicle-to-Grid", "target": "Electric Vehicles", "type": "EXTENDS"},
        {"source": "Vehicle-to-Grid", "target": "Solar PV", "type": "COMPLEMENTS"},
        {"source": "NACS", "target": "Electric Vehicles", "type": "STANDARDIZES"},
        # Carbon capture connections
        {"source": "CCS", "target": "Carbon Dioxide", "type": "CAPTURES"},
        {"source": "Direct Air Capture", "target": "Carbon Dioxide", "type": "REMOVES"},
        {"source": "Climeworks", "target": "Direct Air Capture", "type": "OPERATES"},
        {"source": "Carbon Engineering", "target": "Direct Air Capture", "type": "DEVELOPS"},
        {"source": "BECCS", "target": "CCS", "type": "COMBINES_WITH"},
        {"source": "45Q Tax Credit", "target": "CCS", "type": "INCENTIVIZES"},
        {"source": "Inflation Reduction Act", "target": "45Q Tax Credit", "type": "ENHANCED"},
        {"source": "Sleipner Project", "target": "CCS", "type": "DEMONSTRATES"},
        {"source": "IPCC", "target": "CCS", "type": "RECOMMENDS"},
        # Agriculture connections
        {"source": "Regenerative Agriculture", "target": "Soil Carbon Sequestration", "type": "ENABLES"},
        {"source": "Soil Carbon Sequestration", "target": "Carbon Dioxide", "type": "REMOVES"},
        {"source": "Rodale Institute", "target": "Regenerative Agriculture", "type": "RESEARCHES"},
        {"source": "4 per 1000 Initiative", "target": "Soil Carbon Sequestration", "type": "PROMOTES"},
        {"source": "Paris Agreement", "target": "4 per 1000 Initiative", "type": "INSPIRED"},
        {"source": "Agrivoltaics", "target": "Solar PV", "type": "USES"},
        {"source": "Agrivoltaics", "target": "Regenerative Agriculture", "type": "COMPLEMENTS"},
        {"source": "Precision Agriculture", "target": "Regenerative Agriculture", "type": "SUPPORTS"},
        {"source": "FAO", "target": "Regenerative Agriculture", "type": "PROMOTES"},
    ]

    # Add the "Climate Change" node which is referenced in relationships
    entities.append({"name": "Climate Change", "type": "CONCEPT"})

    return entities, relationships


def build_networkx_graph(entities, relationships):
    """Build a NetworkX graph from entities and relationships."""
    import networkx as nx

    G = nx.Graph()

    # Add nodes with attributes
    for entity in entities:
        G.add_node(entity["name"], entity_type=entity["type"])

    # Add edges with attributes
    for rel in relationships:
        if rel["source"] in G.nodes and rel["target"] in G.nodes:
            G.add_edge(rel["source"], rel["target"], relationship=rel["type"])
        else:
            # Add nodes that appear only in relationships
            G.add_node(rel["source"], entity_type="UNKNOWN")
            G.add_node(rel["target"], entity_type="UNKNOWN")
            G.add_edge(rel["source"], rel["target"], relationship=rel["type"])

    print(f"\nGraph statistics:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Connected components: {nx.number_connected_components(G)}")
    print(f"  Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.1f}")

    return G


def detect_communities(G):
    """Apply the Leiden algorithm for community detection.

    Falls back to Louvain if leidenalg is not available.
    """
    import networkx as nx

    communities = {}

    # Try Leiden algorithm first (preferred)
    try:
        import leidenalg
        import igraph as ig

        print("\nUsing Leiden algorithm for community detection...")

        # Convert NetworkX graph to igraph
        ig_graph = ig.Graph.from_networkx(G)

        # Run Leiden algorithm
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.ModularityVertexPartition,
            seed=42,
        )

        # Map back to node names
        node_names = list(G.nodes())
        for community_id, members in enumerate(partition):
            for member_idx in members:
                communities[node_names[member_idx]] = community_id

        modularity = partition.modularity
        print(f"  Modularity: {modularity:.4f}")

    except ImportError:
        # Fallback to Louvain (built into networkx)
        print("\nleidenalg not installed, falling back to Louvain algorithm...")
        print("  Install for Leiden: pip install leidenalg igraph")

        try:
            from networkx.algorithms.community import louvain_communities

            louvain_result = louvain_communities(G, seed=42)
            for community_id, members in enumerate(louvain_result):
                for member in members:
                    communities[member] = community_id

            modularity = nx.community.modularity(G, louvain_result)
            print(f"  Modularity: {modularity:.4f}")
        except Exception as e:
            # Last resort: use connected components
            print(f"  Louvain failed ({e}), using connected components...")
            for community_id, component in enumerate(nx.connected_components(G)):
                for node in component:
                    communities[node] = community_id

    num_communities = len(set(communities.values()))
    print(f"  Found {num_communities} communities")

    # Print community membership
    for cid in sorted(set(communities.values())):
        members = [n for n, c in communities.items() if c == cid]
        print(f"\n  Community {cid} ({len(members)} members):")
        for m in members:
            etype = G.nodes[m].get("entity_type", "UNKNOWN")
            print(f"    - {m} [{etype}]")

    return communities


def visualize_communities(G, communities):
    """Create a visualization of the graph colored by community."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("\nmatplotlib not installed, skipping visualization.")
        print("  Install with: pip install matplotlib")
        return

    import networkx as nx

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Color palette for communities
    colors = [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
        "#ffff33", "#a65628", "#f781bf", "#999999", "#66c2a5",
    ]

    num_communities = len(set(communities.values()))

    # Assign colors to nodes
    node_colors = []
    for node in G.nodes():
        cid = communities.get(node, 0)
        node_colors.append(colors[cid % len(colors)])

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    # Layout
    pos = nx.spring_layout(G, k=2.0, iterations=50, seed=42)

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, edge_color="gray")

    # Draw nodes
    node_sizes = [300 + 100 * G.degree(n) for n in G.nodes()]
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.8,
        edgecolors="black",
        linewidths=0.5,
    )

    # Draw labels
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=7,
        font_weight="bold",
    )

    # Legend
    legend_handles = []
    for cid in sorted(set(communities.values())):
        members = [n for n, c in communities.items() if c == cid]
        patch = mpatches.Patch(
            color=colors[cid % len(colors)],
            label=f"Community {cid} ({len(members)} nodes)",
        )
        legend_handles.append(patch)

    ax.legend(handles=legend_handles, loc="upper left", fontsize=9)
    ax.set_title("Knowledge Graph Communities (Leiden Algorithm)", fontsize=14)
    ax.axis("off")

    output_path = OUTPUT_DIR / "community_graph.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nVisualization saved to: {output_path}")

    # Also save community data as JSON
    community_data = {
        "num_communities": num_communities,
        "communities": {},
    }
    for cid in sorted(set(communities.values())):
        members = [n for n, c in communities.items() if c == cid]
        community_data["communities"][str(cid)] = members

    json_path = OUTPUT_DIR / "communities.json"
    with open(json_path, "w") as f:
        json.dump(community_data, f, indent=2)
    print(f"Community data saved to: {json_path}")


def main():
    print("=" * 60)
    print("Community Detection on Knowledge Graph")
    print("=" * 60)

    # Load data
    entities, relationships = load_entities_and_relationships()
    print(f"\nLoaded {len(entities)} entities and {len(relationships)} relationships")

    # Build graph
    import networkx as nx
    G = build_networkx_graph(entities, relationships)

    # Detect communities
    communities = detect_communities(G)

    # Visualize
    visualize_communities(G, communities)

    print("\n" + "=" * 60)
    print("Community detection complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
