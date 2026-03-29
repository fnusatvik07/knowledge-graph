"""Train a RotatE knowledge graph embedding model on FB15k-237.

RotatE models relations as rotations in complex embedding space: t ~ h * r,
where embeddings are complex-valued and r has unit modulus (Sun et al., 2019).

RotatE can model symmetry, antisymmetry, inversion, and composition patterns,
making it more expressive than TransE.

Usage:
    python src/03_train_rotate.py
    python src/03_train_rotate.py --epochs 50 --dim 64
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pykeen.pipeline import pipeline

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Train RotatE on FB15k-237")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    args = parser.parse_args()

    print("=" * 60)
    print("Training RotatE on FB15k-237")
    print("=" * 60)
    print(f"  Embedding dim: {args.dim}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size:    {args.batch_size}")
    print()

    # ── Train with PyKEEN pipeline ──────────────────────────────────
    result = pipeline(
        dataset="FB15k-237",
        model="RotatE",
        model_kwargs=dict(
            embedding_dim=args.dim,
        ),
        optimizer="Adam",
        optimizer_kwargs=dict(
            lr=args.lr,
        ),
        training_kwargs=dict(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
        ),
        evaluation_kwargs=dict(
            batch_size=256,
        ),
        random_seed=42,
    )

    # ── Print evaluation results ────────────────────────────────────
    print("\n" + "=" * 60)
    print("RotatE Evaluation Results (Filtered Ranking)")
    print("=" * 60)

    metrics = result.metric_results.to_dict()

    mrr = metrics.get("both.realistic.inverse_harmonic_mean_rank", "N/A")
    hits_at_1 = metrics.get("both.realistic.hits_at_1", "N/A")
    hits_at_3 = metrics.get("both.realistic.hits_at_3", "N/A")
    hits_at_10 = metrics.get("both.realistic.hits_at_10", "N/A")

    print(f"  MRR:      {mrr:.4f}" if isinstance(mrr, float) else f"  MRR:      {mrr}")
    print(f"  Hits@1:   {hits_at_1:.4f}" if isinstance(hits_at_1, float) else f"  Hits@1:   {hits_at_1}")
    print(f"  Hits@3:   {hits_at_3:.4f}" if isinstance(hits_at_3, float) else f"  Hits@3:   {hits_at_3}")
    print(f"  Hits@10:  {hits_at_10:.4f}" if isinstance(hits_at_10, float) else f"  Hits@10:  {hits_at_10}")

    # ── Save model and metrics ──────────────────────────────────────
    model_dir = OUTPUT_DIR / "rotate_model"
    result.save_to_directory(str(model_dir))
    print(f"\n  Model saved to: {model_dir}")

    metrics_summary = {
        "model": "RotatE",
        "embedding_dim": args.dim,
        "epochs": args.epochs,
        "mrr": float(mrr) if isinstance(mrr, float) else None,
        "hits_at_1": float(hits_at_1) if isinstance(hits_at_1, float) else None,
        "hits_at_3": float(hits_at_3) if isinstance(hits_at_3, float) else None,
        "hits_at_10": float(hits_at_10) if isinstance(hits_at_10, float) else None,
    }

    metrics_path = OUTPUT_DIR / "rotate_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"  Metrics saved to: {metrics_path}")

    # ── Compare with TransE ─────────────────────────────────────────
    transe_metrics_path = OUTPUT_DIR / "transe_metrics.json"
    if transe_metrics_path.exists():
        with open(transe_metrics_path) as f:
            transe = json.load(f)
        print("\n" + "-" * 60)
        print("Comparison: RotatE vs TransE")
        print("-" * 60)
        print(f"  {'Metric':<12} {'TransE':>10} {'RotatE':>10} {'Delta':>10}")
        print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
        for metric_name in ["mrr", "hits_at_1", "hits_at_3", "hits_at_10"]:
            t_val = transe.get(metric_name)
            r_val = metrics_summary.get(metric_name)
            if t_val is not None and r_val is not None:
                delta = r_val - t_val
                sign = "+" if delta > 0 else ""
                print(f"  {metric_name:<12} {t_val:>10.4f} {r_val:>10.4f} {sign}{delta:>9.4f}")

    # ── About RotatE ────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("About RotatE")
    print("-" * 60)
    print("""
RotatE (Sun et al., 2019) models relations as rotations in complex space.

Scoring function: ||h * r - t||  (where r has unit modulus |r_i| = 1)

Each relation is a rotation angle per dimension. This allows modeling:
  - Symmetry: r is 0 or pi rotation
  - Antisymmetry: r is not 0 or pi
  - Inversion: r1 = conjugate(r2)
  - Composition: r3 = r1 * r2

Advantages over TransE:
  - Can model symmetric relations (TransE cannot)
  - Better at composition patterns
  - Generally higher MRR and Hits@K on benchmarks

Expected FB15k-237 performance (100 epochs, dim=128):
  MRR ~ 0.28-0.33, Hits@10 ~ 0.48-0.53
""")


if __name__ == "__main__":
    main()
