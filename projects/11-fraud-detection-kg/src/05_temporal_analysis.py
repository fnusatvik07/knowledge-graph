"""Time-based fraud detection on the transaction network.

Analyzes temporal patterns to detect:
- Burst activity: sudden spikes in transaction frequency
- Time-of-day anomalies: transactions at unusual hours (2-5 AM)
- Velocity checks: rapid successive transactions from the same account
- Timeline visualization with matplotlib

Usage:
    python src/05_temporal_analysis.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    """Load accounts and transactions."""
    with open(DATA_DIR / "accounts.json") as f:
        accounts = {a["account_id"]: a for a in json.load(f)}
    with open(DATA_DIR / "transactions.json") as f:
        transactions = json.load(f)
    # Parse timestamps
    for tx in transactions:
        tx["datetime"] = datetime.fromisoformat(tx["timestamp"])
    transactions.sort(key=lambda x: x["datetime"])
    return accounts, transactions


def burst_detection(accounts: dict, transactions: list[dict], window_hours: int = 4) -> dict:
    """Detect accounts with sudden bursts of transaction activity."""
    print("\n" + "=" * 60)
    print("1. Burst Detection")
    print("=" * 60)
    print(f"   (Window size: {window_hours} hours)")

    # Group transactions by account (outgoing)
    account_txs = defaultdict(list)
    for tx in transactions:
        account_txs[tx["from_account"]].append(tx)

    burst_scores = {}

    for acct_id, txs in account_txs.items():
        if len(txs) < 2:
            burst_scores[acct_id] = 0
            continue

        txs_sorted = sorted(txs, key=lambda x: x["datetime"])

        # Sliding window: count transactions in each window
        max_count = 0
        max_window_start = None

        for i, tx in enumerate(txs_sorted):
            window_end = tx["datetime"] + timedelta(hours=window_hours)
            count = sum(1 for t in txs_sorted[i:] if t["datetime"] <= window_end)
            if count > max_count:
                max_count = count
                max_window_start = tx["datetime"]

        # Average rate over entire period
        if len(txs_sorted) > 1:
            total_hours = (txs_sorted[-1]["datetime"] - txs_sorted[0]["datetime"]).total_seconds() / 3600
            avg_rate = len(txs_sorted) / max(total_hours / window_hours, 1)
        else:
            avg_rate = 1

        # Burst score = how much the peak exceeds the average
        burst_ratio = max_count / max(avg_rate, 1)
        burst_scores[acct_id] = min(burst_ratio / 5.0, 1.0)  # Normalize to [0, 1]

        if max_count >= 3:
            name = accounts.get(acct_id, {}).get("name", "?")
            print(f"\n  {acct_id} ({name}):")
            print(f"    Peak: {max_count} transactions in {window_hours}h window")
            print(f"    Window start: {max_window_start}")
            print(f"    Average rate: {avg_rate:.1f} per {window_hours}h window")
            print(f"    Burst ratio:  {burst_ratio:.1f}x")
            print(f"    Burst score:  {burst_scores[acct_id]:.3f}")

    # Fill in accounts with no outgoing transactions
    for acct_id in accounts:
        if acct_id not in burst_scores:
            burst_scores[acct_id] = 0

    return burst_scores


def time_of_day_analysis(accounts: dict, transactions: list[dict]) -> dict:
    """Detect transactions at unusual hours (2-5 AM)."""
    print("\n" + "=" * 60)
    print("2. Time-of-Day Anomaly Detection")
    print("=" * 60)
    print("   (Unusual hours: midnight to 6 AM)")

    account_unusual = defaultdict(list)
    account_total = defaultdict(int)

    for tx in transactions:
        hour = tx["datetime"].hour
        account_total[tx["from_account"]] += 1
        if hour < 6:  # Midnight to 6 AM
            account_unusual[tx["from_account"]].append(tx)

    time_scores = {}
    for acct_id in accounts:
        total = account_total.get(acct_id, 0)
        unusual = len(account_unusual.get(acct_id, []))
        ratio = unusual / total if total > 0 else 0
        time_scores[acct_id] = ratio

    # Report accounts with unusual-hour transactions
    flagged = [
        (acct_id, txs) for acct_id, txs in account_unusual.items()
        if len(txs) > 0
    ]
    flagged.sort(key=lambda x: -len(x[1]))

    for acct_id, txs in flagged:
        name = accounts.get(acct_id, {}).get("name", "?")
        total = account_total[acct_id]
        ratio = len(txs) / total * 100
        print(f"\n  {acct_id} ({name}):")
        print(f"    Unusual-hour transactions: {len(txs)} / {total} ({ratio:.0f}%)")
        for tx in txs[:5]:
            print(f"      {tx['tx_id']}  {tx['timestamp']}  ${tx['amount']:,.2f}  -> {tx['to_account']}")

    return time_scores


def velocity_check(accounts: dict, transactions: list[dict], max_gap_minutes: int = 30) -> dict:
    """Detect rapid successive transactions (velocity check)."""
    print("\n" + "=" * 60)
    print("3. Velocity Check (Rapid Successive Transactions)")
    print("=" * 60)
    print(f"   (Flagging transactions within {max_gap_minutes} minutes)")

    account_txs = defaultdict(list)
    for tx in transactions:
        account_txs[tx["from_account"]].append(tx)

    velocity_scores = {}

    for acct_id, txs in account_txs.items():
        if len(txs) < 2:
            velocity_scores[acct_id] = 0
            continue

        txs_sorted = sorted(txs, key=lambda x: x["datetime"])
        rapid_pairs = []

        for i in range(len(txs_sorted) - 1):
            gap = (txs_sorted[i + 1]["datetime"] - txs_sorted[i]["datetime"]).total_seconds() / 60
            if gap <= max_gap_minutes:
                rapid_pairs.append((txs_sorted[i], txs_sorted[i + 1], gap))

        if rapid_pairs:
            velocity_scores[acct_id] = min(len(rapid_pairs) / 5.0, 1.0)
            name = accounts.get(acct_id, {}).get("name", "?")
            print(f"\n  {acct_id} ({name}): {len(rapid_pairs)} rapid transaction pairs")
            for tx1, tx2, gap in rapid_pairs[:5]:
                print(f"    {tx1['tx_id']} ({tx1['timestamp']}) -> {tx2['tx_id']} ({tx2['timestamp']})  gap={gap:.0f}min")
                print(f"      ${tx1['amount']:,.2f} -> {tx1['to_account']},  ${tx2['amount']:,.2f} -> {tx2['to_account']}")
        else:
            velocity_scores[acct_id] = 0

    for acct_id in accounts:
        if acct_id not in velocity_scores:
            velocity_scores[acct_id] = 0

    return velocity_scores


def visualize_timeline(accounts: dict, transactions: list[dict]):
    """Create a timeline visualization of transactions."""
    print("\n\nGenerating timeline visualization...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])

    # ── Plot 1: Transaction timeline ────────────────────────────────
    ax1 = axes[0]
    dates = [tx["datetime"] for tx in transactions]
    amounts = [tx["amount"] for tx in transactions]

    # Color by whether transaction is at unusual hour
    colors = ["red" if tx["datetime"].hour < 6 else "#2196F3" for tx in transactions]

    ax1.scatter(dates, amounts, c=colors, alpha=0.7, s=40, edgecolors="white", linewidth=0.5)
    ax1.set_ylabel("Transaction Amount ($)", fontsize=11)
    ax1.set_title("Transaction Timeline", fontsize=14, fontweight="bold")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax1.grid(alpha=0.3)

    # Legend
    ax1.scatter([], [], c="red", s=40, label="Unusual hours (midnight-6AM)")
    ax1.scatter([], [], c="#2196F3", s=40, label="Normal hours")
    ax1.legend(loc="upper left", fontsize=9)

    # ── Plot 2: Transaction frequency by hour ───────────────────────
    ax2 = axes[1]
    hours = [tx["datetime"].hour for tx in transactions]
    hour_counts = [0] * 24
    for h in hours:
        hour_counts[h] += 1

    bar_colors = ["red" if h < 6 else "#2196F3" for h in range(24)]
    ax2.bar(range(24), hour_counts, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("Hour of Day", fontsize=11)
    ax2.set_ylabel("Transaction Count", fontsize=11)
    ax2.set_title("Transaction Distribution by Hour", fontsize=12, fontweight="bold")
    ax2.set_xticks(range(24))
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    chart_path = OUTPUT_DIR / "temporal_analysis.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"  Timeline chart saved to: {chart_path}")
    plt.show()


def main():
    print("=" * 60)
    print("Temporal Fraud Analysis")
    print("=" * 60)

    accounts, transactions = load_data()
    print(f"Loaded {len(accounts)} accounts, {len(transactions)} transactions")
    print(f"Date range: {transactions[0]['timestamp']} to {transactions[-1]['timestamp']}")

    # Run all temporal analyses
    burst_scores = burst_detection(accounts, transactions)
    time_scores = time_of_day_analysis(accounts, transactions)
    velocity_scores = velocity_check(accounts, transactions)

    # Visualize
    visualize_timeline(accounts, transactions)

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Temporal Analysis Summary")
    print("=" * 60)

    print(f"\n  {'Account':<10} {'Name':<22} {'Burst':>7} {'Time':>7} {'Velocity':>9}")
    print(f"  {'-'*10} {'-'*22} {'-'*7} {'-'*7} {'-'*9}")

    # Show accounts with any temporal anomaly
    all_accounts = sorted(accounts.keys())
    flagged = []
    for acct_id in all_accounts:
        b = burst_scores.get(acct_id, 0)
        t = time_scores.get(acct_id, 0)
        v = velocity_scores.get(acct_id, 0)
        if b > 0 or t > 0 or v > 0:
            name = accounts[acct_id]["name"]
            flagged.append((acct_id, name, b, t, v))

    flagged.sort(key=lambda x: -(x[2] + x[3] + x[4]))
    for acct_id, name, b, t, v in flagged:
        print(f"  {acct_id:<10} {name:<22} {b:>7.3f} {t:>7.3f} {v:>9.3f}")

    # Save temporal scores
    temporal_results = {
        acct_id: {
            "burst_score": burst_scores.get(acct_id, 0),
            "time_anomaly_score": time_scores.get(acct_id, 0),
            "velocity_score": velocity_scores.get(acct_id, 0),
        }
        for acct_id in accounts
    }

    results_path = OUTPUT_DIR / "temporal_analysis_results.json"
    with open(results_path, "w") as f:
        json.dump(temporal_results, f, indent=2)
    print(f"\n  Temporal scores saved to: {results_path}")


if __name__ == "__main__":
    main()
