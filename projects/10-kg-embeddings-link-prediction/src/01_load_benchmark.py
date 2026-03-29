"""Load the FB15k-237 benchmark dataset and explore its statistics.

FB15k-237 is a widely used knowledge graph embedding benchmark derived from
Freebase. It contains ~15K entities, ~237 relation types, and ~310K triples.
The "-237" variant removes inverse relations that made the original FB15k
trivially solvable via simple rules.

PyKEEN datasets: https://pykeen.readthedocs.io/en/stable/api/pykeen.datasets.html

Usage:
    python src/01_load_benchmark.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pykeen.datasets import FB15k237

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 60)
    print("Loading FB15k-237 Benchmark Dataset")
    print("=" * 60)

    # PyKEEN downloads and caches the dataset automatically
    dataset = FB15k237()

    # ── Dataset overview ────────────────────────────────────────────
    print(f"\nDataset: {dataset.__class__.__name__}")
    print(f"  Entities:  {dataset.num_entities:,}")
    print(f"  Relations: {dataset.num_relations:,}")

    # ── Split statistics ────────────────────────────────────────────
    print(f"\nSplit sizes:")
    print(f"  Training:   {dataset.training.num_triples:,} triples")
    print(f"  Validation: {dataset.validation.num_triples:,} triples")
    print(f"  Testing:    {dataset.testing.num_triples:,} triples")
    total = (
        dataset.training.num_triples
        + dataset.validation.num_triples
        + dataset.testing.num_triples
    )
    print(f"  Total:      {total:,} triples")

    # ── Sample triples ──────────────────────────────────────────────
    print("\n--- Sample training triples (first 10) ---")
    training_triples = dataset.training.triples
    for i, (h, r, t) in enumerate(training_triples[:10]):
        print(f"  {i + 1:2d}. ({h})  --[{r}]-->  ({t})")

    # ── Relation distribution ───────────────────────────────────────
    print("\n--- Relation type distribution (top 15) ---")
    from collections import Counter

    relation_counts = Counter(row[1] for row in training_triples)
    for rel, count in relation_counts.most_common(15):
        pct = count / len(training_triples) * 100
        print(f"  {rel:50s}  {count:5d}  ({pct:.1f}%)")

    # ── Entity degree distribution ──────────────────────────────────
    print("\n--- Entity degree statistics ---")
    entity_counts = Counter()
    for h, r, t in training_triples:
        entity_counts[h] += 1
        entity_counts[t] += 1

    degrees = list(entity_counts.values())
    import numpy as np

    degrees_arr = np.array(degrees)
    print(f"  Mean degree:   {degrees_arr.mean():.1f}")
    print(f"  Median degree: {np.median(degrees_arr):.1f}")
    print(f"  Max degree:    {degrees_arr.max()}")
    print(f"  Min degree:    {degrees_arr.min()}")

    print("\n--- Top 10 most connected entities ---")
    for entity, deg in entity_counts.most_common(10):
        print(f"  {entity:50s}  degree={deg}")

    # ── About the dataset ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("About FB15k-237")
    print("=" * 60)
    print("""
FB15k-237 is derived from Freebase, a large collaborative knowledge base.
Key characteristics:
  - Entities represent real-world things (people, places, films, etc.)
  - Relations represent typed relationships between entities
  - The '-237' variant removes inverse/reciprocal relations from FB15k,
    which previously allowed models to cheat by memorizing simple rules
  - Standard benchmark for evaluating KG embedding models
  - Task: Given (head, relation, ?) or (?, relation, tail), predict
    the missing entity (link prediction)

The dataset is pre-split into train/validation/test sets.
Models are trained on the training set and evaluated on the test set
using filtered ranking metrics (MRR, Hits@K).
""")


if __name__ == "__main__":
    main()
