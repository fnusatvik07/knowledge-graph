"""Combine all signals into a composite fraud risk score for each account.

Aggregates features from graph analysis, cycle detection, community detection,
and temporal analysis into a single fraud risk score per account. Uses LLM
to generate human-readable fraud investigation reports for top suspects.

Usage:
    python src/06_fraud_scoring.py
    python src/06_fraud_scoring.py --provider openai --top-k 5
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared.llm_clients import chat_completion

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_all_scores() -> dict:
    """Load scores from all analysis scripts."""
    scores = {}

    # Account metadata
    with open(DATA_DIR / "accounts.json") as f:
        accounts = {a["account_id"]: a for a in json.load(f)}

    for acct_id, acct in accounts.items():
        scores[acct_id] = {
            "name": acct["name"],
            "type": acct["type"],
            "creation_date": acct["creation_date"],
            "base_risk_score": acct["risk_score"],
        }

    # Graph analysis scores
    graph_path = OUTPUT_DIR / "graph_analysis_scores.json"
    if graph_path.exists():
        with open(graph_path) as f:
            graph_scores = json.load(f)
        for acct_id, data in graph_scores.items():
            if acct_id in scores:
                scores[acct_id]["degree_score"] = data.get("degree_score", 0)
                scores[acct_id]["pagerank_score"] = data.get("pagerank_score", 0)
                scores[acct_id]["betweenness_score"] = data.get("betweenness_score", 0)
                scores[acct_id]["imbalance_score"] = data.get("imbalance_score", 0)
    else:
        print("  Warning: graph_analysis_scores.json not found. Run 02_graph_analysis.py first.")

    # Cycle detection scores
    cycle_path = OUTPUT_DIR / "cycle_detection_results.json"
    if cycle_path.exists():
        with open(cycle_path) as f:
            cycle_data = json.load(f)
        cycle_scores = cycle_data.get("account_cycle_scores", {})
        for acct_id in scores:
            scores[acct_id]["cycle_score"] = cycle_scores.get(acct_id, 0)
    else:
        print("  Warning: cycle_detection_results.json not found. Run 03_cycle_detection.py first.")

    # Community detection (bridge scores)
    community_path = OUTPUT_DIR / "community_detection_results.json"
    if community_path.exists():
        with open(community_path) as f:
            community_data = json.load(f)
        bridge_scores = community_data.get("bridge_scores", {})
        max_bridge = max(bridge_scores.values()) if bridge_scores else 1
        for acct_id in scores:
            raw = bridge_scores.get(acct_id, 0)
            scores[acct_id]["bridge_score"] = raw / max_bridge if max_bridge > 0 else 0
    else:
        print("  Warning: community_detection_results.json not found. Run 04_community_detection.py first.")

    # Temporal analysis scores
    temporal_path = OUTPUT_DIR / "temporal_analysis_results.json"
    if temporal_path.exists():
        with open(temporal_path) as f:
            temporal_data = json.load(f)
        for acct_id in scores:
            t_data = temporal_data.get(acct_id, {})
            scores[acct_id]["burst_score"] = t_data.get("burst_score", 0)
            scores[acct_id]["time_anomaly_score"] = t_data.get("time_anomaly_score", 0)
            scores[acct_id]["velocity_score"] = t_data.get("velocity_score", 0)
    else:
        print("  Warning: temporal_analysis_results.json not found. Run 05_temporal_analysis.py first.")

    return scores


def compute_fraud_score(account_scores: dict) -> float:
    """Compute a weighted composite fraud risk score."""
    weights = {
        "base_risk_score": 0.10,
        "degree_score": 0.10,
        "pagerank_score": 0.05,
        "betweenness_score": 0.10,
        "imbalance_score": 0.10,
        "cycle_score": 0.25,
        "bridge_score": 0.05,
        "burst_score": 0.10,
        "time_anomaly_score": 0.10,
        "velocity_score": 0.05,
    }

    total = 0
    weight_sum = 0
    for feature, weight in weights.items():
        value = account_scores.get(feature, 0)
        if value is not None:
            total += weight * value
            weight_sum += weight

    # Normalize by actual weights used
    if weight_sum > 0:
        total = total / weight_sum

    return min(total, 1.0)


def generate_fraud_report(acct_id: str, acct_data: dict, fraud_score: float,
                          provider: str = "openai", model: str = None) -> str:
    """Use LLM to generate a human-readable fraud investigation report."""
    # Build feature summary
    features = []
    feature_names = {
        "base_risk_score": "Pre-existing Risk Score",
        "degree_score": "Unusual Transaction Count (Degree Anomaly)",
        "pagerank_score": "Network Importance (PageRank)",
        "betweenness_score": "Bridge Between Groups (Betweenness)",
        "imbalance_score": "Money Flow Imbalance",
        "cycle_score": "Circular Money Flow Involvement",
        "bridge_score": "Cross-Community Bridging",
        "burst_score": "Transaction Burst Activity",
        "time_anomaly_score": "Unusual Transaction Hours",
        "velocity_score": "Rapid Successive Transactions",
    }

    for key, label in feature_names.items():
        value = acct_data.get(key, 0)
        if value and value > 0:
            level = "HIGH" if value > 0.5 else "MEDIUM" if value > 0.2 else "LOW"
            features.append(f"- {label}: {value:.3f} ({level})")

    features_str = "\n".join(features) if features else "- No significant anomalies detected"

    prompt = f"""Generate a concise fraud investigation report for this account.

Account ID: {acct_id}
Account Name: {acct_data['name']}
Account Type: {acct_data['type']}
Creation Date: {acct_data['creation_date']}
Composite Fraud Score: {fraud_score:.3f} (scale 0-1, where 1 = highest risk)

Risk Signal Breakdown:
{features_str}

Write a professional 3-4 paragraph investigation summary that:
1. States the overall risk level and key concern
2. Describes the most suspicious signals and what patterns they indicate
3. Recommends next investigation steps
Keep it concise and actionable. Do not use bullet points in the report body."""

    system = "You are a financial fraud analyst writing an investigation report. Be precise and professional."

    try:
        report = chat_completion(prompt, system=system, provider=provider, model=model)
        return report
    except Exception as e:
        return f"[Report generation failed: {e}]"


def main():
    parser = argparse.ArgumentParser(description="Fraud Risk Scoring")
    parser.add_argument("--provider", default="openai", help="LLM provider")
    parser.add_argument("--model", default=None, help="LLM model")
    parser.add_argument("--top-k", type=int, default=5, help="Number of reports to generate")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM report generation")
    args = parser.parse_args()

    print("=" * 60)
    print("Composite Fraud Risk Scoring")
    print("=" * 60)

    # ── Load all scores ─────────────────────────────────────────────
    all_scores = load_all_scores()
    print(f"Loaded scores for {len(all_scores)} accounts")

    # ── Compute composite scores ────────────────────────────────────
    fraud_scores = {}
    for acct_id, data in all_scores.items():
        fraud_scores[acct_id] = compute_fraud_score(data)

    # ── Print ranked results ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Account Fraud Risk Ranking")
    print("=" * 60)

    ranked = sorted(fraud_scores.items(), key=lambda x: -x[1])

    print(f"\n  {'Rank':<6} {'Account':<10} {'Name':<22} {'Type':<10} {'Score':>8} {'Level':<12}")
    print(f"  {'-'*6} {'-'*10} {'-'*22} {'-'*10} {'-'*8} {'-'*12}")

    for rank, (acct_id, score) in enumerate(ranked, 1):
        data = all_scores[acct_id]
        level = "CRITICAL" if score > 0.6 else "HIGH" if score > 0.4 else "MEDIUM" if score > 0.2 else "LOW"
        print(f"  {rank:<6} {acct_id:<10} {data['name']:<22} {data['type']:<10} {score:>8.3f} {level:<12}")

    # ── Feature breakdown for top suspects ──────────────────────────
    print("\n" + "=" * 60)
    print(f"Feature Breakdown (Top {args.top_k} Suspects)")
    print("=" * 60)

    feature_keys = [
        "base_risk_score", "degree_score", "pagerank_score", "betweenness_score",
        "imbalance_score", "cycle_score", "bridge_score", "burst_score",
        "time_anomaly_score", "velocity_score",
    ]

    for acct_id, score in ranked[:args.top_k]:
        data = all_scores[acct_id]
        print(f"\n  {acct_id} ({data['name']})  -- Composite Score: {score:.3f}")
        for key in feature_keys:
            val = data.get(key, 0)
            if val and val > 0:
                bar = "#" * int(val * 20)
                print(f"    {key:25s}  {val:>6.3f}  |{bar}")

    # ── Generate LLM fraud reports ──────────────────────────────────
    if not args.skip_llm:
        print("\n" + "=" * 60)
        print(f"LLM Fraud Investigation Reports (Top {args.top_k})")
        print("=" * 60)

        reports = {}
        for acct_id, score in ranked[:args.top_k]:
            data = all_scores[acct_id]
            print(f"\n{'~' * 60}")
            print(f"Report: {acct_id} ({data['name']}) -- Score: {score:.3f}")
            print("~" * 60)

            report = generate_fraud_report(
                acct_id, data, score,
                provider=args.provider, model=args.model,
            )
            print(report)
            reports[acct_id] = report

    # ── Save results ────────────────────────────────────────────────
    final_results = {
        acct_id: {
            **all_scores[acct_id],
            "fraud_score": fraud_scores[acct_id],
        }
        for acct_id in all_scores
    }

    results_path = OUTPUT_DIR / "fraud_scores.json"
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\n  Fraud scores saved to: {results_path}")

    # Summary stats
    print("\n" + "=" * 60)
    print("Scoring Summary")
    print("=" * 60)
    scores_list = list(fraud_scores.values())
    critical = sum(1 for s in scores_list if s > 0.6)
    high = sum(1 for s in scores_list if 0.4 < s <= 0.6)
    medium = sum(1 for s in scores_list if 0.2 < s <= 0.4)
    low = sum(1 for s in scores_list if s <= 0.2)
    print(f"  CRITICAL (>0.6): {critical} accounts")
    print(f"  HIGH (0.4-0.6):  {high} accounts")
    print(f"  MEDIUM (0.2-0.4): {medium} accounts")
    print(f"  LOW (<0.2):      {low} accounts")


if __name__ == "__main__":
    main()
