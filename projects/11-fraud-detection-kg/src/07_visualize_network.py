"""Visualize the transaction network with pyvis.

Creates an interactive HTML visualization where:
- Nodes are colored by fraud risk score (green -> yellow -> red)
- Detected cycles are highlighted with thicker edges
- Node size reflects total transaction volume
- Hover shows account details

Usage:
    python src/07_visualize_network.py
"""

import sys
import json
import pickle
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import networkx as nx

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_graph() -> nx.DiGraph:
    """Load the transaction graph."""
    pickle_path = OUTPUT_DIR / "transaction_graph.pkl"
    if pickle_path.exists():
        with open(pickle_path, "rb") as f:
            return pickle.load(f)

    with open(DATA_DIR / "accounts.json") as f:
        accounts = json.load(f)
    with open(DATA_DIR / "transactions.json") as f:
        transactions = json.load(f)

    G = nx.DiGraph()
    for acct in accounts:
        G.add_node(acct["account_id"], **{k: v for k, v in acct.items() if k != "account_id"})
    for tx in transactions:
        G.add_edge(tx["from_account"], tx["to_account"],
                    tx_id=tx["tx_id"], amount=tx["amount"],
                    timestamp=tx["timestamp"], merchant_category=tx["merchant_category"])
    return G


def load_fraud_scores() -> dict:
    """Load computed fraud scores."""
    scores_path = OUTPUT_DIR / "fraud_scores.json"
    if scores_path.exists():
        with open(scores_path) as f:
            return json.load(f)
    return {}


def load_cycle_edges(G: nx.DiGraph) -> set:
    """Load cycle edges for highlighting."""
    cycle_path = OUTPUT_DIR / "cycle_detection_results.json"
    cycle_edges = set()
    if cycle_path.exists():
        with open(cycle_path) as f:
            cycle_data = json.load(f)
        for cycle_info in cycle_data.get("cycles", []):
            nodes = cycle_info.get("cycle_nodes", [])
            if cycle_info.get("suspicion_score", 0) >= 0.3:
                for i in range(len(nodes)):
                    src = nodes[i]
                    dst = nodes[(i + 1) % len(nodes)]
                    cycle_edges.add((src, dst))
    return cycle_edges


def risk_to_color(score: float) -> str:
    """Convert a risk score (0-1) to a hex color (green -> yellow -> red)."""
    if score <= 0.2:
        # Green
        return "#4CAF50"
    elif score <= 0.4:
        # Yellow-green
        r = int(76 + (255 - 76) * (score - 0.2) / 0.2)
        g = int(175 + (235 - 175) * (score - 0.2) / 0.2)
        return f"#{r:02x}{g:02x}50"
    elif score <= 0.6:
        # Yellow to orange
        r = 255
        g = int(235 - (235 - 152) * (score - 0.4) / 0.2)
        return f"#{r:02x}{g:02x}00"
    else:
        # Orange to red
        r = 255
        g = int(152 - 152 * (score - 0.6) / 0.4)
        return f"#{r:02x}{g:02x}00"


def main():
    print("=" * 60)
    print("Transaction Network Visualization")
    print("=" * 60)

    G = load_graph()
    fraud_scores = load_fraud_scores()
    cycle_edges = load_cycle_edges(G)

    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Fraud scores available: {len(fraud_scores)} accounts")
    print(f"Cycle edges to highlight: {len(cycle_edges)}")

    # ── Compute node sizes based on transaction volume ──────────────
    volumes = defaultdict(float)
    for u, v, d in G.edges(data=True):
        volumes[u] += d.get("amount", 0)
        volumes[v] += d.get("amount", 0)

    max_volume = max(volumes.values()) if volumes else 1

    # ── Build pyvis network ─────────────────────────────────────────
    try:
        from pyvis.network import Network
    except ImportError:
        print("\n  pyvis not installed. Install with: pip install pyvis")
        print("  Falling back to matplotlib visualization...")
        _matplotlib_fallback(G, fraud_scores, cycle_edges, volumes, max_volume)
        return

    net = Network(
        height="800px",
        width="100%",
        directed=True,
        bgcolor="#1a1a2e",
        font_color="white",
    )

    # Physics settings for better layout
    net.set_options("""
    {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -100,
                "centralGravity": 0.01,
                "springLength": 200,
                "springConstant": 0.08
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 150}
        },
        "nodes": {
            "font": {"size": 14, "color": "white"}
        },
        "edges": {
            "color": {"inherit": false},
            "smooth": {"type": "curvedCW", "roundness": 0.1}
        }
    }
    """)

    # Add nodes
    for node in G.nodes():
        data = G.nodes[node]
        name = data.get("name", node)
        acct_type = data.get("account_type", "unknown")

        # Get fraud score
        fraud_data = fraud_scores.get(node, {})
        score = fraud_data.get("fraud_score", data.get("risk_score", 0))

        # Color by risk
        color = risk_to_color(score)

        # Size by transaction volume
        vol = volumes.get(node, 0)
        size = 15 + 35 * (vol / max_volume)

        # Shape by account type
        shape = "dot" if acct_type == "personal" else "diamond"

        # Hover title with details
        title = (
            f"<b>{name}</b> ({node})<br>"
            f"Type: {acct_type}<br>"
            f"Fraud Score: {score:.3f}<br>"
            f"Volume: ${vol:,.2f}<br>"
            f"Created: {data.get('creation_date', 'N/A')}"
        )

        # Add feature breakdown if available
        feature_keys = [
            ("cycle_score", "Cycle Involvement"),
            ("burst_score", "Burst Activity"),
            ("time_anomaly_score", "Time Anomaly"),
            ("betweenness_score", "Betweenness"),
            ("imbalance_score", "Flow Imbalance"),
        ]
        for key, label in feature_keys:
            val = fraud_data.get(key, 0)
            if val and val > 0:
                title += f"<br>{label}: {val:.3f}"

        net.add_node(
            node,
            label=name.split()[0] if " " in name else name,
            title=title,
            color=color,
            size=size,
            shape=shape,
            borderWidth=3 if score > 0.5 else 1,
            borderWidthSelected=5,
        )

    # Add edges
    for u, v, d in G.edges(data=True):
        amount = d.get("amount", 0)
        tx_id = d.get("tx_id", "?")
        timestamp = d.get("timestamp", "?")
        category = d.get("merchant_category", "?")

        is_cycle_edge = (u, v) in cycle_edges

        title = (
            f"{tx_id}<br>"
            f"${amount:,.2f}<br>"
            f"{timestamp}<br>"
            f"Category: {category}"
        )
        if is_cycle_edge:
            title += "<br><b>PART OF SUSPICIOUS CYCLE</b>"

        net.add_edge(
            u, v,
            title=title,
            value=amount / 1000,  # Edge thickness
            color="red" if is_cycle_edge else "#666666",
            width=4 if is_cycle_edge else 1,
            dashes=False,
            arrows={"to": {"enabled": True, "scaleFactor": 0.5}},
        )

    # Save
    html_path = OUTPUT_DIR / "transaction_network.html"
    net.save_graph(str(html_path))
    print(f"\n  Interactive visualization saved to: {html_path}")
    print("  Open in a browser to explore the network.")

    # ── Also generate a static matplotlib version ───────────────────
    _matplotlib_fallback(G, fraud_scores, cycle_edges, volumes, max_volume)


def _matplotlib_fallback(G, fraud_scores, cycle_edges, volumes, max_volume):
    """Generate a static matplotlib visualization."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    print("\nGenerating static matplotlib visualization...")

    fig, ax = plt.subplots(figsize=(16, 12))

    pos = nx.spring_layout(G, seed=42, k=2.5)

    # Node colors based on fraud score
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        fraud_data = fraud_scores.get(node, {})
        score = fraud_data.get("fraud_score", G.nodes[node].get("risk_score", 0))
        node_colors.append(score)

        vol = volumes.get(node, 0)
        node_sizes.append(200 + 800 * (vol / max_volume))

    # Color map: green -> yellow -> red
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "risk", ["#4CAF50", "#FFEB3B", "#FF9800", "#F44336"]
    )

    # Draw non-cycle edges first (light)
    non_cycle_edges = [(u, v) for u, v in G.edges() if (u, v) not in cycle_edges]
    nx.draw_networkx_edges(
        G, pos, edgelist=non_cycle_edges,
        alpha=0.2, arrows=True, arrowsize=8,
        edge_color="#999999", ax=ax,
    )

    # Draw cycle edges (highlighted)
    cycle_edge_list = [(u, v) for u, v in G.edges() if (u, v) in cycle_edges]
    if cycle_edge_list:
        nx.draw_networkx_edges(
            G, pos, edgelist=cycle_edge_list,
            alpha=0.8, arrows=True, arrowsize=12,
            edge_color="red", width=2.5, ax=ax,
        )

    # Draw nodes
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        cmap=cmap,
        vmin=0, vmax=1,
        edgecolors="white",
        linewidths=1,
        alpha=0.9,
        ax=ax,
    )

    # Labels
    labels = {}
    for node in G.nodes():
        name = G.nodes[node].get("name", node)
        short = name.split()[0] if " " in name else name
        labels[node] = short
    nx.draw_networkx_labels(G, pos, labels, font_size=7, font_color="black", ax=ax)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Fraud Risk Score", fontsize=11)

    ax.set_title(
        "Transaction Network -- Fraud Risk Visualization",
        fontsize=14, fontweight="bold",
    )

    # Legend
    ax.plot([], [], "r-", linewidth=2.5, label="Suspicious cycle edge")
    ax.plot([], [], "-", color="#999999", alpha=0.4, label="Normal transaction")
    ax.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    chart_path = OUTPUT_DIR / "transaction_network_static.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"  Static chart saved to: {chart_path}")
    plt.show()


if __name__ == "__main__":
    main()
