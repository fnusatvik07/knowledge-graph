"""Run graph algorithms to find suspicious patterns in the transaction network.

Applies structural analysis to surface anomalous accounts:
- Degree analysis (unusually many transactions)
- PageRank (important nodes in the transaction network)
- Betweenness centrality (bridges between clusters)
- Connected components (isolated transaction clusters)
- Weighted in/out degree (money flow imbalance)

Usage:
    python src/02_graph_analysis.py
"""

import sys
import json
import pickle
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import networkx as nx
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_graph() -> nx.DiGraph:
    """Load the transaction graph from pickle or rebuild it."""
    pickle_path = OUTPUT_DIR / "transaction_graph.pkl"
    if pickle_path.exists():
        with open(pickle_path, "rb") as f:
            return pickle.load(f)

    # Rebuild if pickle not found
    print("  Graph pickle not found, rebuilding...")
    from importlib import import_module
    sys.path.insert(0, str(PROJECT_DIR / "src"))

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


def degree_analysis(G: nx.DiGraph) -> dict:
    """Identify accounts with anomalous degree (too many connections)."""
    print("\n" + "=" * 60)
    print("1. Degree Analysis")
    print("=" * 60)

    total_degrees = dict(G.degree())
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())

    # Compute z-scores for total degree
    values = list(total_degrees.values())
    mean_deg = np.mean(values)
    std_deg = np.std(values)

    print(f"\n  Mean degree: {mean_deg:.1f}")
    print(f"  Std degree:  {std_deg:.1f}")

    degree_scores = {}
    print(f"\n  Accounts with degree > mean + 1 std ({mean_deg + std_deg:.1f}):")
    for node, deg in sorted(total_degrees.items(), key=lambda x: -x[1]):
        z_score = (deg - mean_deg) / std_deg if std_deg > 0 else 0
        degree_scores[node] = max(0, z_score)
        if deg > mean_deg + std_deg:
            name = G.nodes[node].get("name", "?")
            print(f"    {node} ({name:20s})  degree={deg}  in={in_degrees[node]}  out={out_degrees[node]}  z={z_score:.2f}")

    return degree_scores


def pagerank_analysis(G: nx.DiGraph) -> dict:
    """Find important nodes using PageRank."""
    print("\n" + "=" * 60)
    print("2. PageRank Analysis")
    print("=" * 60)

    # Weight edges by transaction amount
    pr = nx.pagerank(G, weight="amount", alpha=0.85)

    # Normalize to [0, 1]
    max_pr = max(pr.values())
    min_pr = min(pr.values())
    pr_range = max_pr - min_pr if max_pr != min_pr else 1

    pr_scores = {node: (val - min_pr) / pr_range for node, val in pr.items()}

    print("\n  Top 10 accounts by PageRank (weighted by amount):")
    for node, score in sorted(pr.items(), key=lambda x: -x[1])[:10]:
        name = G.nodes[node].get("name", "?")
        print(f"    {node} ({name:20s})  PageRank={score:.6f}  normalized={pr_scores[node]:.3f}")

    return pr_scores


def betweenness_analysis(G: nx.DiGraph) -> dict:
    """Find bridge accounts using betweenness centrality."""
    print("\n" + "=" * 60)
    print("3. Betweenness Centrality Analysis")
    print("=" * 60)

    bc = nx.betweenness_centrality(G, weight=None)

    # Normalize
    max_bc = max(bc.values()) if max(bc.values()) > 0 else 1
    bc_scores = {node: val / max_bc for node, val in bc.items()}

    print("\n  Top 10 accounts by betweenness centrality:")
    print("  (High betweenness = bridges between different groups)")
    for node, score in sorted(bc.items(), key=lambda x: -x[1])[:10]:
        name = G.nodes[node].get("name", "?")
        print(f"    {node} ({name:20s})  betweenness={score:.4f}  normalized={bc_scores[node]:.3f}")

    return bc_scores


def component_analysis(G: nx.DiGraph):
    """Analyze connected components for isolated clusters."""
    print("\n" + "=" * 60)
    print("4. Connected Components Analysis")
    print("=" * 60)

    # Weakly connected components (ignore edge direction)
    weak_components = list(nx.weakly_connected_components(G))
    print(f"\n  Weakly connected components: {len(weak_components)}")
    for i, comp in enumerate(sorted(weak_components, key=len, reverse=True)):
        print(f"    Component {i+1}: {len(comp)} nodes")
        if len(comp) <= 10:
            for node in sorted(comp):
                name = G.nodes[node].get("name", "?")
                print(f"      {node} ({name})")

    # Strongly connected components (respect edge direction)
    strong_components = [c for c in nx.strongly_connected_components(G) if len(c) > 1]
    print(f"\n  Strongly connected components (size > 1): {len(strong_components)}")
    print("  (These indicate cyclic transaction flows)")
    for i, comp in enumerate(sorted(strong_components, key=len, reverse=True)):
        members = sorted(comp)
        names = [f"{n}({G.nodes[n].get('name', '?')})" for n in members]
        print(f"    SCC {i+1} ({len(comp)} nodes): {', '.join(names)}")


def flow_imbalance_analysis(G: nx.DiGraph) -> dict:
    """Detect accounts with unusual money flow imbalance."""
    print("\n" + "=" * 60)
    print("5. Money Flow Imbalance Analysis")
    print("=" * 60)

    inflow = defaultdict(float)
    outflow = defaultdict(float)

    for u, v, d in G.edges(data=True):
        outflow[u] += d["amount"]
        inflow[v] += d["amount"]

    # Compute imbalance ratio: |outflow - inflow| / max(outflow, inflow)
    imbalance_scores = {}
    print("\n  Accounts with significant flow imbalance:")
    print(f"  {'Account':<10} {'Name':<22} {'Inflow':>12} {'Outflow':>12} {'Net':>12} {'Ratio':>8}")
    print(f"  {'-'*10} {'-'*22} {'-'*12} {'-'*12} {'-'*12} {'-'*8}")

    for node in G.nodes():
        inf = inflow.get(node, 0)
        outf = outflow.get(node, 0)
        net = outf - inf
        max_flow = max(inf, outf)
        ratio = abs(net) / max_flow if max_flow > 0 else 0
        imbalance_scores[node] = ratio

    # Sort by imbalance ratio descending
    sorted_nodes = sorted(imbalance_scores.items(), key=lambda x: -x[1])
    for node, ratio in sorted_nodes[:15]:
        name = G.nodes[node].get("name", "?")
        inf = inflow.get(node, 0)
        outf = outflow.get(node, 0)
        net = outf - inf
        direction = "OUT" if net > 0 else "IN"
        print(f"  {node:<10} {name:<22} ${inf:>10,.2f} ${outf:>10,.2f} ${net:>+10,.2f} {ratio:>7.2f}")

    return imbalance_scores


def main():
    print("=" * 60)
    print("Transaction Graph Analysis")
    print("=" * 60)

    G = load_graph()
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Run all analyses
    degree_scores = degree_analysis(G)
    pr_scores = pagerank_analysis(G)
    bc_scores = betweenness_analysis(G)
    component_analysis(G)
    imbalance_scores = flow_imbalance_analysis(G)

    # Save scores for fraud scoring script
    analysis_results = {
        node: {
            "name": G.nodes[node].get("name", "?"),
            "degree_score": degree_scores.get(node, 0),
            "pagerank_score": pr_scores.get(node, 0),
            "betweenness_score": bc_scores.get(node, 0),
            "imbalance_score": imbalance_scores.get(node, 0),
        }
        for node in G.nodes()
    }

    results_path = OUTPUT_DIR / "graph_analysis_scores.json"
    with open(results_path, "w") as f:
        json.dump(analysis_results, f, indent=2)
    print(f"\n  Analysis scores saved to: {results_path}")


if __name__ == "__main__":
    main()
