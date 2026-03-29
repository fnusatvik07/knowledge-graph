"""Compare all trained KG embedding models side by side.

Loads saved metrics from TransE, RotatE, and ComplEx training runs.
Generates a matplotlib bar chart comparing MRR and Hits@K across models.
Discusses which model works best and why.

Usage:
    python src/05_model_comparison.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_metrics():
    """Load saved metrics for all trained models."""
    models = {}
    for name in ["transe", "rotate", "complex"]:
        path = OUTPUT_DIR / f"{name}_metrics.json"
        if path.exists():
            with open(path) as f:
                models[name] = json.load(f)
        else:
            print(f"  Warning: {path} not found. Run training script first.")
    return models


def plot_comparison(models: dict):
    """Generate a bar chart comparing models on all metrics."""
    metric_keys = ["mrr", "hits_at_1", "hits_at_3", "hits_at_10"]
    metric_labels = ["MRR", "Hits@1", "Hits@3", "Hits@10"]
    model_names = [m["model"] for m in models.values()]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    x = np.arange(len(metric_labels))
    width = 0.25
    offsets = np.arange(len(model_names)) - (len(model_names) - 1) / 2

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (key, data) in enumerate(models.items()):
        values = [data.get(mk, 0) or 0 for mk in metric_keys]
        bars = ax.bar(
            x + offsets[i] * width,
            values,
            width,
            label=data["model"],
            color=colors[i % len(colors)],
            edgecolor="white",
            linewidth=0.5,
        )
        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("KG Embedding Model Comparison on FB15k-237", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, max(0.6, max(
        data.get("hits_at_10", 0) or 0 for data in models.values()
    ) + 0.1))
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    chart_path = OUTPUT_DIR / "model_comparison.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved to: {chart_path}")
    plt.show()


def analyze_results(models: dict):
    """Analyze which model performs best and discuss why."""
    print("\n" + "=" * 60)
    print("Analysis: Which Model Works Best?")
    print("=" * 60)

    # Find best model per metric
    metric_keys = ["mrr", "hits_at_1", "hits_at_3", "hits_at_10"]
    metric_labels = ["MRR", "Hits@1", "Hits@3", "Hits@10"]

    for mk, ml in zip(metric_keys, metric_labels):
        best_name = None
        best_val = -1
        for key, data in models.items():
            val = data.get(mk, 0) or 0
            if val > best_val:
                best_val = val
                best_name = data["model"]
        print(f"  Best {ml}: {best_name} ({best_val:.4f})")

    # Overall winner by MRR
    best_mrr_model = max(models.values(), key=lambda m: m.get("mrr", 0) or 0)
    print(f"\n  Overall winner (by MRR): {best_mrr_model['model']}")

    print("""
Discussion:
-----------
TransE is the simplest model (translations) and typically has the lowest
scores. It struggles with 1-to-N relations that are common in FB15k-237.

RotatE (rotations in complex space) usually outperforms TransE because
it can model symmetry, antisymmetry, and composition patterns. The
rotation formulation is more expressive.

ComplEx (Hermitian dot product) is a bilinear model that naturally
handles symmetric and antisymmetric relations. It is often competitive
with RotatE and sometimes wins on specific metrics.

In practice, the best model depends on the specific KG structure:
  - Many symmetric relations -> ComplEx or RotatE
  - Many composition patterns -> RotatE
  - Simple 1-to-1 relations -> TransE is sufficient
  - Large KGs -> TransE is most memory-efficient

For production use, consider also:
  - DistMult (simpler bilinear, good for symmetric relations)
  - ConvE (convolutional, better for sparse KGs)
  - TuckER (tensor decomposition, state-of-the-art on some benchmarks)
""")


def main():
    print("=" * 60)
    print("KG Embedding Model Comparison")
    print("=" * 60)

    models = load_metrics()

    if len(models) == 0:
        print("\nNo trained models found. Run training scripts first:")
        print("  python src/02_train_transe.py")
        print("  python src/03_train_rotate.py")
        print("  python src/04_train_complex.py")
        return

    # ── Print comparison table ──────────────────────────────────────
    print(f"\n  {'Metric':<12}", end="")
    for data in models.values():
        print(f" {data['model']:>10}", end="")
    print()
    print(f"  {'-'*12}", end="")
    for _ in models:
        print(f" {'-'*10}", end="")
    print()

    for mk in ["mrr", "hits_at_1", "hits_at_3", "hits_at_10"]:
        print(f"  {mk:<12}", end="")
        for data in models.values():
            val = data.get(mk)
            if val is not None:
                print(f" {val:>10.4f}", end="")
            else:
                print(f" {'N/A':>10}", end="")
        print()

    # ── Generate comparison chart ───────────────────────────────────
    print("\nGenerating comparison chart...")
    plot_comparison(models)

    # ── Analyze results ─────────────────────────────────────────────
    analyze_results(models)


if __name__ == "__main__":
    main()
