"""Detect circular money flows in the transaction graph.

Finds cycles where money flows A->B->C->A, which may indicate money
laundering or other fraudulent circular transfer schemes. Reports cycle
lengths, amounts involved, and participating accounts.

Usage:
    python src/03_cycle_detection.py
"""

import sys
import json
import pickle
from pathlib import Path
from datetime import datetime

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


def load_transactions() -> list[dict]:
    """Load raw transaction data for detailed analysis."""
    with open(DATA_DIR / "transactions.json") as f:
        return json.load(f)


def find_simple_cycles(G: nx.DiGraph, max_length: int = 6) -> list[list]:
    """Find all simple cycles up to a given length."""
    all_cycles = list(nx.simple_cycles(G))
    # Filter by length
    short_cycles = [c for c in all_cycles if len(c) <= max_length]
    return sorted(short_cycles, key=len)


def analyze_cycle(G: nx.DiGraph, cycle: list, transactions: list[dict]) -> dict:
    """Analyze a single cycle for suspicious properties."""
    cycle_edges = []
    total_amount = 0
    timestamps = []

    for i in range(len(cycle)):
        src = cycle[i]
        dst = cycle[(i + 1) % len(cycle)]

        # Find all transactions on this edge
        edge_txs = [
            tx for tx in transactions
            if tx["from_account"] == src and tx["to_account"] == dst
        ]

        for tx in edge_txs:
            cycle_edges.append(tx)
            total_amount += tx["amount"]
            timestamps.append(datetime.fromisoformat(tx["timestamp"]))

    # Calculate timing properties
    if timestamps:
        timestamps.sort()
        time_span = (timestamps[-1] - timestamps[0]).total_seconds() / 3600  # hours
        min_interval = min(
            (timestamps[i+1] - timestamps[i]).total_seconds() / 60
            for i in range(len(timestamps) - 1)
        ) if len(timestamps) > 1 else float("inf")
    else:
        time_span = 0
        min_interval = float("inf")

    # Check if transactions happen at unusual hours
    unusual_hours = sum(
        1 for ts in timestamps
        if ts.hour < 6 or ts.hour > 22
    )

    # Compute amount consistency (similar amounts suggest layering)
    amounts = [tx["amount"] for tx in cycle_edges]
    if len(amounts) > 1:
        import numpy as np
        amount_cv = np.std(amounts) / np.mean(amounts) if np.mean(amounts) > 0 else 0
    else:
        amount_cv = 0

    return {
        "cycle_nodes": cycle,
        "length": len(cycle),
        "total_amount": total_amount,
        "num_transactions": len(cycle_edges),
        "time_span_hours": round(time_span, 2),
        "min_interval_minutes": round(min_interval, 2),
        "unusual_hour_count": unusual_hours,
        "amount_cv": round(amount_cv, 4),
        "transactions": [tx["tx_id"] for tx in cycle_edges],
        "amounts": amounts,
    }


def score_cycle_suspicion(analysis: dict) -> float:
    """Score how suspicious a cycle is (0.0 to 1.0)."""
    score = 0.0

    # Short time span = more suspicious
    if analysis["time_span_hours"] < 2:
        score += 0.3
    elif analysis["time_span_hours"] < 12:
        score += 0.15

    # High amounts = more suspicious
    if analysis["total_amount"] > 20000:
        score += 0.25
    elif analysis["total_amount"] > 10000:
        score += 0.15

    # Similar amounts = layering pattern
    if analysis["amount_cv"] < 0.15:
        score += 0.2
    elif analysis["amount_cv"] < 0.3:
        score += 0.1

    # Unusual hours = more suspicious
    if analysis["unusual_hour_count"] > 0:
        score += 0.15

    # Rapid succession
    if analysis["min_interval_minutes"] < 30:
        score += 0.1

    return min(score, 1.0)


def main():
    print("=" * 60)
    print("Cycle Detection: Finding Circular Money Flows")
    print("=" * 60)

    G = load_graph()
    transactions = load_transactions()
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ── Find cycles ─────────────────────────────────────────────────
    print("\nSearching for cycles (up to length 6)...")
    cycles = find_simple_cycles(G, max_length=6)
    print(f"  Found {len(cycles)} cycles total")

    # Group by length
    from collections import Counter
    length_counts = Counter(len(c) for c in cycles)
    for length, count in sorted(length_counts.items()):
        print(f"    Length {length}: {count} cycles")

    # ── Analyze each cycle ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Cycle Analysis")
    print("=" * 60)

    cycle_results = []
    for cycle in cycles:
        analysis = analyze_cycle(G, cycle, transactions)
        suspicion = score_cycle_suspicion(analysis)
        analysis["suspicion_score"] = round(suspicion, 3)
        cycle_results.append(analysis)

    # Sort by suspicion score
    cycle_results.sort(key=lambda x: -x["suspicion_score"])

    for i, result in enumerate(cycle_results):
        nodes = result["cycle_nodes"]
        node_names = [
            f"{n}({G.nodes[n].get('name', '?')})"
            for n in nodes
        ]
        cycle_path = " -> ".join(node_names) + f" -> {node_names[0]}"

        flag = " *** HIGH RISK ***" if result["suspicion_score"] >= 0.5 else ""
        print(f"\n  Cycle {i+1}: {cycle_path}")
        print(f"    Length:             {result['length']} nodes")
        print(f"    Total amount:       ${result['total_amount']:,.2f}")
        print(f"    Transactions:       {result['num_transactions']}")
        print(f"    Time span:          {result['time_span_hours']:.1f} hours")
        print(f"    Min interval:       {result['min_interval_minutes']:.0f} minutes")
        print(f"    Unusual hours:      {result['unusual_hour_count']}")
        print(f"    Amount consistency:  CV={result['amount_cv']:.3f}")
        print(f"    Suspicion score:    {result['suspicion_score']:.3f}{flag}")
        if result["amounts"]:
            print(f"    Amounts:            {['${:,.0f}'.format(a) for a in result['amounts']]}")

    # ── Summary ─────────────────────────────────────────────────────
    high_risk = [r for r in cycle_results if r["suspicion_score"] >= 0.5]
    print("\n" + "=" * 60)
    print("Cycle Detection Summary")
    print("=" * 60)
    print(f"  Total cycles found:    {len(cycles)}")
    print(f"  High-risk cycles:      {len(high_risk)}")

    # Accounts involved in high-risk cycles
    flagged_accounts = set()
    for result in high_risk:
        flagged_accounts.update(result["cycle_nodes"])
    print(f"  Accounts in high-risk cycles: {len(flagged_accounts)}")
    for acct in sorted(flagged_accounts):
        name = G.nodes[acct].get("name", "?")
        risk = G.nodes[acct].get("risk_score", 0)
        print(f"    {acct} ({name})  pre-existing risk={risk:.2f}")

    # Save results
    # Convert sets to lists for JSON serialization
    cycle_scores = {}
    for result in cycle_results:
        for node in result["cycle_nodes"]:
            if node not in cycle_scores:
                cycle_scores[node] = 0
            cycle_scores[node] = max(cycle_scores[node], result["suspicion_score"])

    results_path = OUTPUT_DIR / "cycle_detection_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "cycles": [
                {k: v for k, v in r.items()}
                for r in cycle_results
            ],
            "account_cycle_scores": cycle_scores,
        }, f, indent=2)
    print(f"\n  Results saved to: {results_path}")


if __name__ == "__main__":
    main()
