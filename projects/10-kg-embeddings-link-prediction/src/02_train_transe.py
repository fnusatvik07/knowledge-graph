"""Train a TransE knowledge graph embedding model on FB15k-237.

TransE models relations as translations in embedding space: h + r ~ t.
It is the simplest and most foundational KG embedding model (Bordes et al., 2013).

TransE works well for 1-to-1 relations but struggles with 1-to-N, N-to-1,
and N-to-N relations because the same head+relation always maps to one point.

PyKEEN pipeline: https://pykeen.readthedocs.io/en/stable/api/pykeen.pipeline.pipeline.html

Usage:
    python src/02_train_transe.py
    python src/02_train_transe.py --epochs 50 --dim 64
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
    parser = argparse.ArgumentParser(description="Train TransE on FB15k-237")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    args = parser.parse_args()

    print("=" * 60)
    print("Training TransE on FB15k-237")
    print("=" * 60)
    print(f"  Embedding dim: {args.dim}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size:    {args.batch_size}")
    print()

    # ── Train with PyKEEN pipeline ──────────────────────────────────
    result = pipeline(
        dataset="FB15k-237",
        model="TransE",
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
    print("TransE Evaluation Results (Filtered Ranking)")
    print("=" * 60)

    metrics = result.metric_results.to_dict()

    # Extract key metrics (both sides averaged)
    mrr = metrics.get("both.realistic.inverse_harmonic_mean_rank", "N/A")
    hits_at_1 = metrics.get("both.realistic.hits_at_1", "N/A")
    hits_at_3 = metrics.get("both.realistic.hits_at_3", "N/A")
    hits_at_10 = metrics.get("both.realistic.hits_at_10", "N/A")

    print(f"  MRR:      {mrr:.4f}" if isinstance(mrr, float) else f"  MRR:      {mrr}")
    print(f"  Hits@1:   {hits_at_1:.4f}" if isinstance(hits_at_1, float) else f"  Hits@1:   {hits_at_1}")
    print(f"  Hits@3:   {hits_at_3:.4f}" if isinstance(hits_at_3, float) else f"  Hits@3:   {hits_at_3}")
    print(f"  Hits@10:  {hits_at_10:.4f}" if isinstance(hits_at_10, float) else f"  Hits@10:  {hits_at_10}")

    # ── Save model and metrics ──────────────────────────────────────
    model_dir = OUTPUT_DIR / "transe_model"
    result.save_to_directory(str(model_dir))
    print(f"\n  Model saved to: {model_dir}")

    # Save metrics as JSON for comparison script
    metrics_summary = {
        "model": "TransE",
        "embedding_dim": args.dim,
        "epochs": args.epochs,
        "mrr": float(mrr) if isinstance(mrr, float) else None,
        "hits_at_1": float(hits_at_1) if isinstance(hits_at_1, float) else None,
        "hits_at_3": float(hits_at_3) if isinstance(hits_at_3, float) else None,
        "hits_at_10": float(hits_at_10) if isinstance(hits_at_10, float) else None,
    }

    metrics_path = OUTPUT_DIR / "transe_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"  Metrics saved to: {metrics_path}")

    # ── About TransE ────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("About TransE")
    print("-" * 60)
    print("""
TransE (Bordes et al., 2013) is the foundational KG embedding model.

Scoring function: ||h + r - t||  (L1 or L2 norm)

Intuition: The relation vector 'r' translates head entity 'h'
to the position of tail entity 't' in embedding space.

Strengths:
  - Simple and efficient
  - Works well for 1-to-1 relations
  - Good baseline for comparison

Weaknesses:
  - Cannot model 1-to-N, N-to-1, or N-to-N relations well
    (e.g., "born_in" with many people born in the same city)
  - All entities sharing a relation map to the same point

Expected FB15k-237 performance (100 epochs, dim=128):
  MRR ~ 0.25-0.30, Hits@10 ~ 0.45-0.50
""")


if __name__ == "__main__":
    main()
