"""Build a transaction knowledge graph from synthetic data.

Constructs a NetworkX directed graph where nodes are accounts (with metadata)
and edges are transactions (with amount, timestamp, merchant category).

Usage:
    python src/01_build_transaction_graph.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import networkx as nx

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

ACCOUNTS_PATH = DATA_DIR / "accounts.json"
TRANSACTIONS_PATH = DATA_DIR / "transactions.json"


def load_data():
    """Load accounts and transactions from JSON files."""
    with open(ACCOUNTS_PATH) as f:
        accounts = json.load(f)
    with open(TRANSACTIONS_PATH) as f:
        transactions = json.load(f)
    return accounts, transactions


def build_graph(accounts: list[dict], transactions: list[dict]) -> nx.DiGraph:
    """Build a directed graph from accounts and transactions."""
    G = nx.DiGraph()

    # Add account nodes with metadata
    for acct in accounts:
        G.add_node(
            acct["account_id"],
            name=acct["name"],
            account_type=acct["type"],
            creation_date=acct["creation_date"],
            risk_score=acct["risk_score"],
        )

    # Add transaction edges
    for tx in transactions:
        G.add_edge(
            tx["from_account"],
            tx["to_account"],
            tx_id=tx["tx_id"],
            amount=tx["amount"],
            timestamp=tx["timestamp"],
            merchant_category=tx["merchant_category"],
        )

    return G


def print_graph_stats(G: nx.DiGraph):
    """Print comprehensive graph statistics."""
    print("\n" + "=" * 60)
    print("Transaction Graph Statistics")
    print("=" * 60)

    print(f"\n  Nodes (accounts): {G.number_of_nodes()}")
    print(f"  Edges (transactions): {G.number_of_edges()}")
    print(f"  Density: {nx.density(G):.4f}")

    # Account type breakdown
    personal = sum(1 for _, d in G.nodes(data=True) if d.get("account_type") == "personal")
    business = sum(1 for _, d in G.nodes(data=True) if d.get("account_type") == "business")
    print(f"\n  Personal accounts: {personal}")
    print(f"  Business accounts: {business}")

    # Degree statistics
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]
    print(f"\n  In-degree  -- min: {min(in_degrees)}, max: {max(in_degrees)}, avg: {sum(in_degrees)/len(in_degrees):.1f}")
    print(f"  Out-degree -- min: {min(out_degrees)}, max: {max(out_degrees)}, avg: {sum(out_degrees)/len(out_degrees):.1f}")

    # Transaction amount statistics
    amounts = [d["amount"] for _, _, d in G.edges(data=True)]
    print(f"\n  Transaction amounts:")
    print(f"    Min:   ${min(amounts):,.2f}")
    print(f"    Max:   ${max(amounts):,.2f}")
    print(f"    Mean:  ${sum(amounts)/len(amounts):,.2f}")
    print(f"    Total: ${sum(amounts):,.2f}")

    # Merchant category breakdown
    from collections import Counter
    categories = Counter(d["merchant_category"] for _, _, d in G.edges(data=True))
    print(f"\n  Merchant categories:")
    for cat, count in categories.most_common():
        print(f"    {cat:15s}  {count} transactions")

    # Pre-flagged suspicious accounts (risk_score > 0.5)
    suspicious = [
        (n, d) for n, d in G.nodes(data=True)
        if d.get("risk_score", 0) > 0.5
    ]
    print(f"\n  Pre-flagged suspicious accounts (risk > 0.5): {len(suspicious)}")
    for node, data in sorted(suspicious, key=lambda x: -x[1]["risk_score"]):
        print(f"    {node} ({data['name']:20s})  risk={data['risk_score']:.2f}  type={data['account_type']}")

    # Top accounts by transaction volume (total amount out)
    print(f"\n  Top 10 accounts by outgoing volume:")
    outgoing_volume = {}
    for u, v, d in G.edges(data=True):
        outgoing_volume[u] = outgoing_volume.get(u, 0) + d["amount"]
    for acct, vol in sorted(outgoing_volume.items(), key=lambda x: -x[1])[:10]:
        name = G.nodes[acct].get("name", "?")
        print(f"    {acct} ({name:20s})  ${vol:>12,.2f}")


def main():
    print("=" * 60)
    print("Building Transaction Knowledge Graph")
    print("=" * 60)

    accounts, transactions = load_data()
    print(f"\nLoaded {len(accounts)} accounts and {len(transactions)} transactions")

    G = build_graph(accounts, transactions)
    print_graph_stats(G)

    # Save graph for use by other scripts
    graph_path = OUTPUT_DIR / "transaction_graph.graphml"
    nx.write_graphml(G, str(graph_path))
    print(f"\n  Graph saved to: {graph_path}")

    # Also save as pickle for full fidelity
    import pickle
    pickle_path = OUTPUT_DIR / "transaction_graph.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(G, f)
    print(f"  Graph pickle saved to: {pickle_path}")


if __name__ == "__main__":
    main()
