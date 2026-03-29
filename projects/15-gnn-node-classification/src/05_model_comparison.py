"""Compare GCN, GraphSAGE, and GAT model performance side by side.

Loads metrics from all three models and generates:
    1. Summary table of test accuracy and F1 scores
    2. Training loss curves (matplotlib)
    3. Validation accuracy curves (matplotlib)
    4. Bar chart comparing final metrics
    5. Analysis of when each architecture excels
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

try:
    import torch
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print("Install: pip install torch matplotlib")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"

MODELS = {
    "GCN": OUTPUT_DIR / "gcn_metrics.json",
    "GraphSAGE": OUTPUT_DIR / "graphsage_metrics.json",
    "GAT": OUTPUT_DIR / "gat_metrics.json",
}

COLORS = {"GCN": "#2196F3", "GraphSAGE": "#4CAF50", "GAT": "#FF9800"}


def load_all_metrics() -> dict:
    """Load metrics for all available models."""
    metrics = {}
    for name, path in MODELS.items():
        if path.exists():
            with open(path) as f:
                metrics[name] = json.load(f)
            print(f"  Loaded {name} metrics")
        else:
            print(f"  WARNING: {name} metrics not found at {path}")
    return metrics


def print_summary_table(metrics: dict):
    """Print a formatted comparison table."""
    print(f"\n{'Model':>12} | {'Test Acc':>9} | {'Test F1':>9} | {'Best Val Acc':>12} | "
          f"{'Best Epoch':>10} | {'Params':>8}")
    print("-" * 75)
    for name, m in metrics.items():
        print(f"{name:>12} | {m['test_accuracy']:>9.4f} | {m['test_f1_macro']:>9.4f} | "
              f"{m['best_val_accuracy']:>12.4f} | {m['best_epoch']:>10} | "
              f"{m['total_params']:>8}")

    # Find best model
    best_name = max(metrics, key=lambda n: metrics[n]["test_accuracy"])
    print(f"\nBest model by test accuracy: {best_name} "
          f"({metrics[best_name]['test_accuracy']:.4f})")


def plot_training_curves(metrics: dict):
    """Plot training loss and validation accuracy curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Training loss curves
    ax = axes[0]
    for name, m in metrics.items():
        ax.plot(m["history"]["train_loss"], label=name, color=COLORS.get(name),
                alpha=0.8, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss over Epochs")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Validation accuracy curves
    ax = axes[1]
    for name, m in metrics.items():
        ax.plot(m["history"]["val_acc"], label=name, color=COLORS.get(name),
                alpha=0.8, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Validation Accuracy over Epochs")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Bar chart of final metrics
    ax = axes[2]
    x_labels = list(metrics.keys())
    x = range(len(x_labels))
    width = 0.35

    acc_values = [metrics[n]["test_accuracy"] for n in x_labels]
    f1_values = [metrics[n]["test_f1_macro"] for n in x_labels]
    colors = [COLORS.get(n, "#999") for n in x_labels]

    bars1 = ax.bar([i - width / 2 for i in x], acc_values, width, label="Accuracy",
                   color=colors, alpha=0.8)
    bars2 = ax.bar([i + width / 2 for i in x], f1_values, width, label="Macro F1",
                   color=colors, alpha=0.5, hatch="//")

    ax.set_ylabel("Score")
    ax.set_title("Test Performance Comparison")
    ax.set_xticks(list(x))
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}",
                ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}",
                ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    chart_path = OUTPUT_DIR / "model_comparison.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved comparison chart to {chart_path}")


def print_analysis(metrics: dict):
    """Print analysis of when each architecture excels."""
    print(f"\n{'='*60}")
    print("ARCHITECTURE ANALYSIS")
    print(f"{'='*60}")

    print("""
GCN (Graph Convolutional Network):
    - Treats all neighbors equally with normalized aggregation
    - Simplest architecture, fewest parameters
    - Works well when graph structure is homogeneous
    - Spectral-based: tied to the specific graph seen during training
    - Best for: transductive settings where all nodes are known at train time

GraphSAGE (Sample and Aggregate):
    - Samples a fixed-size neighborhood, aggregates with mean/pool/LSTM
    - Concatenates self-features with aggregated neighbor features
    - KEY ADVANTAGE: inductive — can generalize to unseen nodes
    - Best for: dynamic graphs where new nodes are added after training
    - Best for: large graphs where full-batch training is infeasible

GAT (Graph Attention Network):
    - Learns attention weights: not all neighbors contribute equally
    - Multi-head attention stabilizes learning (like Transformer heads)
    - More parameters but potentially more expressive
    - Best for: heterogeneous graphs where neighbor importance varies
    - Best for: KGs where different relationship types carry different weight
""")

    # Specific comparison based on results
    if len(metrics) >= 2:
        sorted_models = sorted(metrics.items(), key=lambda x: x[1]["test_accuracy"],
                               reverse=True)
        best, worst = sorted_models[0], sorted_models[-1]
        gap = best[1]["test_accuracy"] - worst[1]["test_accuracy"]
        print(f"On this KG dataset:")
        print(f"  - {best[0]} achieved the highest accuracy ({best[1]['test_accuracy']:.4f})")
        print(f"  - Gap between best and worst: {gap:.4f}")
        if gap < 0.05:
            print(f"  - The models performed similarly, suggesting graph structure is ")
            print(f"    relatively uniform (attention does not add much)")
        else:
            print(f"  - Significant difference, suggesting the graph has heterogeneous")
            print(f"    neighborhood patterns where attention helps")


def main():
    print("=" * 60)
    print("Step 5: Model Comparison (GCN vs GraphSAGE vs GAT)")
    print("=" * 60)

    print("\nLoading model metrics...")
    metrics = load_all_metrics()

    if not metrics:
        print("\nERROR: No model metrics found. Train at least one model first.")
        print("Run: python src/02_train_gcn.py")
        sys.exit(1)

    print_summary_table(metrics)

    if len(metrics) >= 2:
        print("\nGenerating comparison charts...")
        plot_training_curves(metrics)

    print_analysis(metrics)

    print("\nDone!")


if __name__ == "__main__":
    main()
