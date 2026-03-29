"""Train a ComplEx knowledge graph embedding model on FB15k-237.

ComplEx uses complex-valued embeddings with a Hermitian dot product scoring
function (Trouillon et al., 2016). It naturally handles symmetric and
antisymmetric relations through the imaginary component.

Usage:
    python src/04_train_complex.py
    python src/04_train_complex.py --epochs 50 --dim 64
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
    parser = argparse.ArgumentParser(description="Train ComplEx on FB15k-237")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    args = parser.parse_args()

    print("=" * 60)
    print("Training ComplEx on FB15k-237")
    print("=" * 60)
    print(f"  Embedding dim: {args.dim}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size:    {args.batch_size}")
    print()

    # ── Train with PyKEEN pipeline ──────────────────────────────────
    result = pipeline(
        dataset="FB15k-237",
        model="ComplEx",
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
    print("ComplEx Evaluation Results (Filtered Ranking)")
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
    model_dir = OUTPUT_DIR / "complex_model"
    result.save_to_directory(str(model_dir))
    print(f"\n  Model saved to: {model_dir}")

    metrics_summary = {
        "model": "ComplEx",
        "embedding_dim": args.dim,
        "epochs": args.epochs,
        "mrr": float(mrr) if isinstance(mrr, float) else None,
        "hits_at_1": float(hits_at_1) if isinstance(hits_at_1, float) else None,
        "hits_at_3": float(hits_at_3) if isinstance(hits_at_3, float) else None,
        "hits_at_10": float(hits_at_10) if isinstance(hits_at_10, float) else None,
    }

    metrics_path = OUTPUT_DIR / "complex_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"  Metrics saved to: {metrics_path}")

    # ── Compare all three models ────────────────────────────────────
    print("\n" + "-" * 60)
    print("Comparison: All Trained Models")
    print("-" * 60)

    all_models = []
    for name in ["transe", "rotate", "complex"]:
        path = OUTPUT_DIR / f"{name}_metrics.json"
        if path.exists():
            with open(path) as f:
                all_models.append(json.load(f))

    if len(all_models) > 1:
        print(f"  {'Metric':<12}", end="")
        for m in all_models:
            print(f" {m['model']:>10}", end="")
        print()
        print(f"  {'-'*12}", end="")
        for _ in all_models:
            print(f" {'-'*10}", end="")
        print()
        for metric_name in ["mrr", "hits_at_1", "hits_at_3", "hits_at_10"]:
            print(f"  {metric_name:<12}", end="")
            for m in all_models:
                val = m.get(metric_name)
                if val is not None:
                    print(f" {val:>10.4f}", end="")
                else:
                    print(f" {'N/A':>10}", end="")
            print()

    # ── About ComplEx ───────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("About ComplEx")
    print("-" * 60)
    print("""
ComplEx (Trouillon et al., 2016) uses complex-valued embeddings.

Scoring function: Re(<h, r, conj(t)>)  (Hermitian dot product)

Where Re() is the real part and conj() is complex conjugation.
The asymmetry of the Hermitian product naturally distinguishes
(h, r, t) from (t, r, h), allowing it to model:
  - Symmetric relations: imaginary parts cancel out
  - Antisymmetric relations: imaginary parts contribute
  - 1-to-N and N-to-1 relations: different from TransE

Advantages:
  - Handles symmetric AND antisymmetric relations
  - Bilinear model (efficient dot-product scoring)
  - Often competitive with or better than RotatE

Expected FB15k-237 performance (100 epochs, dim=128):
  MRR ~ 0.27-0.32, Hits@10 ~ 0.47-0.52
""")


if __name__ == "__main__":
    main()
