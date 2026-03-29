"""Use trained KG embedding models to predict missing links.

Given (entity, relation, ?), predict top-K tail entities.
Given (?, relation, entity), predict top-K head entities.
Shows examples with interpretable entity names from FB15k-237.

Usage:
    python src/06_link_prediction.py
    python src/06_link_prediction.py --model rotate --top-k 10
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from pykeen.datasets import FB15k237
from pykeen.models import Model

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"


def load_best_model():
    """Load the best trained model based on saved MRR metrics."""
    best_model_name = None
    best_mrr = -1

    for name in ["transe", "rotate", "complex"]:
        metrics_path = OUTPUT_DIR / f"{name}_metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            mrr = metrics.get("mrr", 0) or 0
            if mrr > best_mrr:
                best_mrr = mrr
                best_model_name = name

    return best_model_name


def main():
    parser = argparse.ArgumentParser(description="KG Link Prediction")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["transe", "rotate", "complex"],
        help="Model to use (default: best by MRR)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of predictions")
    args = parser.parse_args()

    # ── Select model ────────────────────────────────────────────────
    model_name = args.model or load_best_model()
    if model_name is None:
        print("No trained models found. Run training scripts first.")
        return

    model_dir = OUTPUT_DIR / f"{model_name}_model"
    if not model_dir.exists():
        print(f"Model directory not found: {model_dir}")
        print("Run the corresponding training script first.")
        return

    print("=" * 60)
    print(f"Link Prediction with {model_name.upper()}")
    print("=" * 60)

    # ── Load model and dataset ──────────────────────────────────────
    print(f"\nLoading model from {model_dir}...")
    model = torch.load(model_dir / "trained_model.pkl", map_location="cpu")
    model.eval()

    print("Loading FB15k-237 dataset...")
    dataset = FB15k237()

    entity_to_id = dataset.training.entity_to_id
    relation_to_id = dataset.training.relation_to_id
    id_to_entity = {v: k for k, v in entity_to_id.items()}
    id_to_relation = {v: k for k, v in relation_to_id.items()}

    print(f"  Entities:  {len(entity_to_id):,}")
    print(f"  Relations: {len(relation_to_id):,}")

    # ── Tail prediction: (head, relation, ?) ────────────────────────
    print("\n" + "=" * 60)
    print("Tail Prediction: (head, relation, ?) -> top-K tails")
    print("=" * 60)

    # Pick some interesting example triples from the test set
    test_triples = dataset.testing.triples
    # Select a few diverse examples
    example_indices = [0, 10, 50, 100, 200]

    for idx in example_indices:
        if idx >= len(test_triples):
            continue

        h_label, r_label, t_label = test_triples[idx]
        h_id = entity_to_id.get(h_label)
        r_id = relation_to_id.get(r_label)

        if h_id is None or r_id is None:
            continue

        print(f"\n  Query: ({h_label}, {r_label}, ?)")
        print(f"  Ground truth: {t_label}")

        # Get predictions using PyKEEN's predict method
        h_tensor = torch.tensor([h_id], dtype=torch.long)
        r_tensor = torch.tensor([r_id], dtype=torch.long)

        # Score all possible tail entities
        all_entity_ids = torch.arange(len(entity_to_id), dtype=torch.long)
        h_expanded = h_tensor.expand(len(entity_to_id))
        r_expanded = r_tensor.expand(len(entity_to_id))

        with torch.no_grad():
            # Create triple tensor: (num_entities, 3)
            triples = torch.stack([h_expanded, r_expanded, all_entity_ids], dim=1)
            scores = model.score_hrt(triples).squeeze()

        # Get top-K predictions
        top_k_scores, top_k_indices = torch.topk(scores, args.top_k)

        print(f"  Top-{args.top_k} predictions:")
        for rank, (score, eid) in enumerate(zip(top_k_scores, top_k_indices), 1):
            entity_name = id_to_entity.get(eid.item(), f"entity_{eid.item()}")
            marker = " <-- correct!" if entity_name == t_label else ""
            print(f"    {rank}. {entity_name:50s}  (score: {score.item():.4f}){marker}")

    # ── Head prediction: (?, relation, tail) ────────────────────────
    print("\n" + "=" * 60)
    print("Head Prediction: (?, relation, tail) -> top-K heads")
    print("=" * 60)

    for idx in example_indices[:3]:
        if idx >= len(test_triples):
            continue

        h_label, r_label, t_label = test_triples[idx]
        t_id = entity_to_id.get(t_label)
        r_id = relation_to_id.get(r_label)

        if t_id is None or r_id is None:
            continue

        print(f"\n  Query: (?, {r_label}, {t_label})")
        print(f"  Ground truth: {h_label}")

        t_tensor = torch.tensor([t_id], dtype=torch.long)
        r_tensor = torch.tensor([r_id], dtype=torch.long)

        all_entity_ids = torch.arange(len(entity_to_id), dtype=torch.long)
        t_expanded = t_tensor.expand(len(entity_to_id))
        r_expanded = r_tensor.expand(len(entity_to_id))

        with torch.no_grad():
            triples = torch.stack([all_entity_ids, r_expanded, t_expanded], dim=1)
            scores = model.score_hrt(triples).squeeze()

        top_k_scores, top_k_indices = torch.topk(scores, args.top_k)

        print(f"  Top-{args.top_k} predictions:")
        for rank, (score, eid) in enumerate(zip(top_k_scores, top_k_indices), 1):
            entity_name = id_to_entity.get(eid.item(), f"entity_{eid.item()}")
            marker = " <-- correct!" if entity_name == h_label else ""
            print(f"    {rank}. {entity_name:50s}  (score: {score.item():.4f}){marker}")

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("Link Prediction Summary")
    print("-" * 60)
    print("""
Link prediction is the core downstream task for KG embeddings.

Given a query (h, r, ?) or (?, r, t), the model scores all possible
entities and returns the highest-scoring candidates.

Applications:
  - Knowledge base completion (filling in missing facts)
  - Recommendation systems (predicting user-item relationships)
  - Drug discovery (predicting drug-target interactions)
  - Social network analysis (predicting future connections)

The quality of predictions depends on:
  - Model architecture (TransE vs RotatE vs ComplEx)
  - Training data coverage
  - Relation type (1-to-1 is easier than N-to-N)
  - Entity frequency (rare entities are harder to predict)
""")


if __name__ == "__main__":
    main()
