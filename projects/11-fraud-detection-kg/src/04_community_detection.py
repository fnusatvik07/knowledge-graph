"""Apply community detection to find transaction clusters.

Uses the Louvain algorithm to partition accounts into communities based
on transaction patterns. Identifies accounts that bridge multiple
communities (potential mule accounts) and visualizes the results.

Usage:
    python src/04_community_detection.py
"""

import sys
import json
import pickle
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

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


def run_louvain(G: nx.DiGraph) -> dict:
    """Run Louvain community detection on the undirected version of the graph."""
    try:
        import community as community_louvain
    except ImportError:
        print("  Install python-louvain: pip install python-louvain")
        print("  Falling back to NetworkX greedy modularity...")
        # Fallback to NetworkX built-in
        G_undirected = G.to_undirected()
        communities = nx.community.greedy_modularity_communities(G_undirected)
        partition = {}
        for i, comm in enumerate(communities):
            for node in comm:
                partition[node] = i
        return partition

    G_undirected = G.to_undirected()

    # Weight edges by total transaction amount
    for u, v in G_undirected.edges():
        total_amount = 0
        if G.has_edge(u, v):
            total_amount += G[u][v].get("amount", 0)
        if G.has_edge(v, u):
            total_amount += G[v][u].get("amount", 0)
        G_undirected[u][v]["weight"] = total_amount

    partition = community_louvain.best_partition(G_undirected, weight="weight", random_state=42)
    return partition


def find_bridge_accounts(G: nx.DiGraph, partition: dict) -> dict:
    """Find accounts that bridge multiple communities."""
    bridge_scores = {}

    for node in G.nodes():
        node_community = partition.get(node)
        neighbor_communities = set()

        # Check both incoming and outgoing neighbors
        for neighbor in set(G.predecessors(node)) | set(G.successors(node)):
            neighbor_communities.add(partition.get(neighbor))

        # Remove own community
        neighbor_communities.discard(node_community)

        # Bridge score = number of other communities connected to
        bridge_scores[node] = len(neighbor_communities)

    return bridge_scores


def visualize_communities(G: nx.DiGraph, partition: dict, bridge_scores: dict):
    """Visualize the graph with communities colored differently."""
    fig, ax = plt.subplots(figsize=(14, 10))

    G_undirected = G.to_undirected()
    pos = nx.spring_layout(G_undirected, seed=42, k=2.0)

    # Color by community
    communities = set(partition.values())
    cmap = plt.cm.get_cmap("tab10", len(communities))

    for comm_id in sorted(communities):
        nodes_in_comm = [n for n, c in partition.items() if c == comm_id]
        node_sizes = []
        for n in nodes_in_comm:
            # Size by total transaction volume
            volume = sum(d.get("amount", 0) for _, _, d in G.edges(n, data=True))
            volume += sum(d.get("amount", 0) for _, _, d in G.in_edges(n, data=True))
            node_sizes.append(max(200, volume / 50))

        # Highlight bridge accounts with different edge color
        edge_colors = []
        for n in nodes_in_comm:
            if bridge_scores.get(n, 0) > 0:
                edge_colors.append("red")
            else:
                edge_colors.append("white")

        nx.draw_networkx_nodes(
            G_undirected,
            pos,
            nodelist=nodes_in_comm,
            node_color=[cmap(comm_id)] * len(nodes_in_comm),
            node_size=node_sizes,
            edgecolors=edge_colors,
            linewidths=2,
            alpha=0.8,
            ax=ax,
        )

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.2, arrows=True, arrowsize=10, ax=ax)

    # Labels
    labels = {n: G.nodes[n].get("name", n).split()[0] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)

    ax.set_title("Transaction Network Communities (Louvain)", fontsize=14, fontweight="bold")
    ax.text(
        0.02, 0.02,
        "Red border = bridge account (connects communities)",
        transform=ax.transAxes, fontsize=9, color="red",
    )

    # Legend for communities
    for comm_id in sorted(communities):
        count = sum(1 for c in partition.values() if c == comm_id)
        ax.scatter([], [], c=[cmap(comm_id)], s=100, label=f"Community {comm_id} ({count} accounts)")
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    chart_path = OUTPUT_DIR / "community_detection.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"\n  Community chart saved to: {chart_path}")
    plt.show()


def main():
    print("=" * 60)
    print("Community Detection in Transaction Network")
    print("=" * 60)

    G = load_graph()
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ── Run community detection ─────────────────────────────────────
    print("\nRunning Louvain community detection...")
    partition = run_louvain(G)

    communities = defaultdict(list)
    for node, comm_id in partition.items():
        communities[comm_id].append(node)

    print(f"\n  Found {len(communities)} communities")
    modularity = nx.community.modularity(
        G.to_undirected(),
        [set(members) for members in communities.values()],
    )
    print(f"  Modularity: {modularity:.4f}")

    # ── Print community details ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("Community Details")
    print("=" * 60)

    for comm_id in sorted(communities.keys()):
        members = communities[comm_id]
        print(f"\n  Community {comm_id} ({len(members)} accounts):")

        # Internal vs external edges
        internal_edges = 0
        external_edges = 0
        internal_amount = 0
        external_amount = 0

        for u, v, d in G.edges(data=True):
            if partition.get(u) == comm_id or partition.get(v) == comm_id:
                if partition.get(u) == comm_id and partition.get(v) == comm_id:
                    internal_edges += 1
                    internal_amount += d.get("amount", 0)
                else:
                    external_edges += 1
                    external_amount += d.get("amount", 0)

        for node in sorted(members):
            name = G.nodes[node].get("name", "?")
            acct_type = G.nodes[node].get("account_type", "?")
            risk = G.nodes[node].get("risk_score", 0)
            flag = " [FLAGGED]" if risk > 0.5 else ""
            print(f"    {node} ({name:20s})  type={acct_type:10s}  risk={risk:.2f}{flag}")

        print(f"    Internal edges: {internal_edges}  (${internal_amount:,.2f})")
        print(f"    External edges: {external_edges}  (${external_amount:,.2f})")

    # ── Bridge account analysis ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("Bridge Account Analysis")
    print("=" * 60)
    print("(Accounts connecting multiple communities -- potential mule accounts)")

    bridge_scores = find_bridge_accounts(G, partition)

    bridges = [(node, score) for node, score in bridge_scores.items() if score > 0]
    bridges.sort(key=lambda x: -x[1])

    if bridges:
        for node, score in bridges:
            name = G.nodes[node].get("name", "?")
            own_comm = partition.get(node)
            risk = G.nodes[node].get("risk_score", 0)
            flag = " *** SUSPICIOUS ***" if risk > 0.5 else ""
            print(f"  {node} ({name:20s})  community={own_comm}  bridges={score} other communities  risk={risk:.2f}{flag}")
    else:
        print("  No bridge accounts detected.")

    # ── Visualize ───────────────────────────────────────────────────
    print("\nGenerating community visualization...")
    visualize_communities(G, partition, bridge_scores)

    # Save results
    community_results = {
        "num_communities": len(communities),
        "modularity": round(modularity, 4),
        "partition": partition,
        "bridge_scores": bridge_scores,
    }

    results_path = OUTPUT_DIR / "community_detection_results.json"
    with open(results_path, "w") as f:
        json.dump(community_results, f, indent=2)
    print(f"  Results saved to: {results_path}")


if __name__ == "__main__":
    main()
