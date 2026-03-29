"""Train KG embeddings on a custom small knowledge graph.

Instead of a benchmark dataset, this script trains TransE on a custom KG
about AI and technology entities (from data/custom_kg.tsv). This bridges
the gap between benchmark evaluation and real-world KG embedding use.

The custom KG is loaded from a TSV file (head, relation, tail) and
converted to PyKEEN's TriplesFactory format.

Usage:
    python src/07_custom_kg_embeddings.py
    python src/07_custom_kg_embeddings.py --epochs 200 --dim 64
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

CUSTOM_KG_PATH = DATA_DIR / "custom_kg.tsv"


def load_custom_kg(path: Path) -> np.ndarray:
    """Load a custom KG from a TSV file (head, relation, tail)."""
    triples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) == 3:
                triples.append(parts)
    return np.array(triples)


def main():
    parser = argparse.ArgumentParser(description="Train embeddings on custom KG")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    args = parser.parse_args()

    print("=" * 60)
    print("Custom KG Embeddings (AI/Tech Domain)")
    print("=" * 60)

    # ── Load custom KG ──────────────────────────────────────────────
    if not CUSTOM_KG_PATH.exists():
        print(f"Error: Custom KG file not found at {CUSTOM_KG_PATH}")
        return

    triples = load_custom_kg(CUSTOM_KG_PATH)
    print(f"\nLoaded {len(triples)} triples from {CUSTOM_KG_PATH.name}")

    # ── Dataset statistics ──────────────────────────────────────────
    entities = set(triples[:, 0]) | set(triples[:, 2])
    relations = set(triples[:, 1])
    print(f"  Entities:  {len(entities)}")
    print(f"  Relations: {len(relations)}")
    print(f"  Triples:   {len(triples)}")

    print("\n  Relation types:")
    from collections import Counter
    rel_counts = Counter(triples[:, 1])
    for rel, count in rel_counts.most_common():
        print(f"    {rel:30s}  {count} triples")

    print("\n  Sample triples:")
    for h, r, t in triples[:10]:
        print(f"    ({h}) --[{r}]--> ({t})")

    # ── Create PyKEEN TriplesFactory ────────────────────────────────
    tf = TriplesFactory.from_labeled_triples(triples)

    # Split into train/test (80/20 for small KG)
    training, testing = tf.split([0.8, 0.2], random_state=42)

    print(f"\n  Training triples: {training.num_triples}")
    print(f"  Testing triples:  {testing.num_triples}")

    # ── Train TransE ────────────────────────────────────────────────
    print(f"\nTraining TransE (dim={args.dim}, epochs={args.epochs})...")

    result = pipeline(
        training=training,
        testing=testing,
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
            batch_size=32,
        ),
        random_seed=42,
    )

    # ── Evaluation ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Custom KG Evaluation Results")
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

    # ── Link prediction on custom KG ────────────────────────────────
    print("\n" + "=" * 60)
    print("Link Prediction on Custom KG")
    print("=" * 60)

    model = result.model
    model.eval()

    entity_to_id = training.entity_to_id
    relation_to_id = training.relation_to_id
    id_to_entity = {v: k for k, v in entity_to_id.items()}

    # Example predictions
    prediction_queries = [
        ("Python", "used_for", None),          # What is Python used for?
        ("GPT-4", "developed_by", None),       # Who developed GPT-4?
        (None, "is_a", "Programming Language"), # What entities are programming languages?
        ("TensorFlow", "used_for", None),      # What is TensorFlow used for?
        (None, "is_a", "Company"),             # What entities are companies?
    ]

    for head, rel, tail in prediction_queries:
        if rel not in relation_to_id:
            continue

        r_id = relation_to_id[rel]

        if tail is None and head in entity_to_id:
            # Tail prediction: (head, relation, ?)
            h_id = entity_to_id[head]
            print(f"\n  Query: ({head}, {rel}, ?)")

            h_tensor = torch.tensor([h_id] * len(entity_to_id), dtype=torch.long)
            r_tensor = torch.tensor([r_id] * len(entity_to_id), dtype=torch.long)
            t_tensor = torch.arange(len(entity_to_id), dtype=torch.long)

            with torch.no_grad():
                triples_tensor = torch.stack([h_tensor, r_tensor, t_tensor], dim=1)
                scores = model.score_hrt(triples_tensor).squeeze()

            top_k = min(5, len(entity_to_id))
            top_scores, top_indices = torch.topk(scores, top_k)
            print(f"  Top-{top_k} predictions:")
            for rank, (score, eid) in enumerate(zip(top_scores, top_indices), 1):
                name = id_to_entity.get(eid.item(), "?")
                print(f"    {rank}. {name:30s}  (score: {score.item():.4f})")

        elif head is None and tail in entity_to_id:
            # Head prediction: (?, relation, tail)
            t_id = entity_to_id[tail]
            print(f"\n  Query: (?, {rel}, {tail})")

            h_tensor = torch.arange(len(entity_to_id), dtype=torch.long)
            r_tensor = torch.tensor([r_id] * len(entity_to_id), dtype=torch.long)
            t_tensor = torch.tensor([t_id] * len(entity_to_id), dtype=torch.long)

            with torch.no_grad():
                triples_tensor = torch.stack([h_tensor, r_tensor, t_tensor], dim=1)
                scores = model.score_hrt(triples_tensor).squeeze()

            top_k = min(5, len(entity_to_id))
            top_scores, top_indices = torch.topk(scores, top_k)
            print(f"  Top-{top_k} predictions:")
            for rank, (score, eid) in enumerate(zip(top_scores, top_indices), 1):
                name = id_to_entity.get(eid.item(), "?")
                print(f"    {rank}. {name:30s}  (score: {score.item():.4f})")

    # ── Save model ──────────────────────────────────────────────────
    model_dir = OUTPUT_DIR / "custom_kg_model"
    result.save_to_directory(str(model_dir))
    print(f"\n  Custom KG model saved to: {model_dir}")

    print("\n" + "-" * 60)
    print("Custom KG Embeddings: Key Takeaways")
    print("-" * 60)
    print("""
Training embeddings on a small custom KG demonstrates:

1. Real-world KGs are much smaller than benchmarks, so overfitting is
   a risk. Use fewer epochs and smaller embedding dimensions.

2. The quality of predictions depends on KG density. Sparse KGs with
   few triples per entity will have poor predictions.

3. Custom KGs let you predict domain-specific missing links:
   - What technologies might a company use?
   - What concepts are related to a given topic?
   - What entities might have a given property?

4. For production use, combine embedding predictions with rule-based
   validation and human review.
""")


if __name__ == "__main__":
    main()
