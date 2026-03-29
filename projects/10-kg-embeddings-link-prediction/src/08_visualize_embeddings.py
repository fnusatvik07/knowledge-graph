"""Visualize KG entity embeddings in 2D using t-SNE.

Loads trained entity embeddings from a PyKEEN model, reduces them to 2D
with t-SNE, and plots with matplotlib. Colors entities by type to show
that similar entities cluster together in embedding space.

Usage:
    python src/08_visualize_embeddings.py
    python src/08_visualize_embeddings.py --model rotate --perplexity 30
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR.mkdir(exist_ok=True)


def visualize_benchmark_embeddings(model_name: str, perplexity: int, top_n: int):
    """Visualize embeddings from a model trained on FB15k-237."""
    from pykeen.datasets import FB15k237

    model_dir = OUTPUT_DIR / f"{model_name}_model"
    if not model_dir.exists():
        print(f"  Model not found at {model_dir}. Run training script first.")
        return

    print(f"Loading {model_name} model...")
    model = torch.load(model_dir / "trained_model.pkl", map_location="cpu")
    model.eval()

    print("Loading FB15k-237 for entity labels...")
    dataset = FB15k237()
    entity_to_id = dataset.training.entity_to_id
    id_to_entity = {v: k for k, v in entity_to_id.items()}

    # Extract entity embeddings
    with torch.no_grad():
        entity_embeddings = model.entity_representations[0]()
        if torch.is_complex(entity_embeddings):
            # For ComplEx/RotatE: concatenate real and imaginary parts
            embeddings = torch.cat(
                [entity_embeddings.real, entity_embeddings.imag], dim=-1
            ).numpy()
        else:
            embeddings = entity_embeddings.numpy()

    print(f"  Embedding shape: {embeddings.shape}")

    # ── Heuristic entity type classification ────────────────────────
    # FB15k-237 entities have Freebase MIDs, but we can infer types
    # from the relations they participate in
    training_triples = dataset.training.triples

    # Classify entities by their most common relation patterns
    from collections import Counter, defaultdict

    entity_relations = defaultdict(list)
    for h, r, t in training_triples:
        entity_relations[h].append(("head", r))
        entity_relations[t].append(("tail", r))

    # Simple type heuristic based on relation names
    def classify_entity(entity_name: str, rels: list) -> str:
        rel_names = [r for _, r in rels]
        rel_str = " ".join(rel_names).lower()
        if "film" in rel_str or "movie" in rel_str:
            return "Film/Media"
        elif "music" in rel_str or "artist" in rel_str or "album" in rel_str:
            return "Music"
        elif "location" in rel_str or "country" in rel_str or "place" in rel_str:
            return "Location"
        elif "person" in rel_str or "people" in rel_str or "gender" in rel_str:
            return "Person"
        elif "language" in rel_str:
            return "Language"
        elif "sport" in rel_str or "team" in rel_str:
            return "Sports"
        elif "education" in rel_str or "university" in rel_str:
            return "Education"
        else:
            return "Other"

    # Classify top-N entities by degree
    entity_degree = Counter()
    for h, r, t in training_triples:
        entity_degree[h] += 1
        entity_degree[t] += 1

    top_entities = [e for e, _ in entity_degree.most_common(top_n)]
    top_ids = [entity_to_id[e] for e in top_entities if e in entity_to_id]
    top_types = [
        classify_entity(e, entity_relations.get(e, []))
        for e in top_entities
        if e in entity_to_id
    ]

    # Get embeddings for top entities
    top_embeddings = embeddings[top_ids]

    # ── t-SNE dimensionality reduction ──────────────────────────────
    print(f"Running t-SNE (perplexity={perplexity}) on {len(top_ids)} entities...")
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(top_ids) - 1),
        random_state=42,
        n_iter=1000,
    )
    coords = tsne.fit_transform(top_embeddings)

    # ── Plot ────────────────────────────────────────────────────────
    type_set = sorted(set(top_types))
    color_map = plt.cm.get_cmap("tab10", len(type_set))
    type_to_color = {t: color_map(i) for i, t in enumerate(type_set)}

    fig, ax = plt.subplots(figsize=(14, 10))

    for entity_type in type_set:
        mask = [i for i, t in enumerate(top_types) if t == entity_type]
        if not mask:
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            label=entity_type,
            color=type_to_color[entity_type],
            alpha=0.7,
            s=40,
            edgecolors="white",
            linewidth=0.5,
        )

    ax.set_title(
        f"Entity Embeddings ({model_name.upper()}) - t-SNE Projection",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.legend(title="Entity Type", loc="best", fontsize=9)
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    chart_path = OUTPUT_DIR / f"{model_name}_embeddings_tsne.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved to: {chart_path}")
    plt.show()


def visualize_custom_kg_embeddings(perplexity: int):
    """Visualize embeddings from the custom KG model."""
    model_dir = OUTPUT_DIR / "custom_kg_model"
    if not model_dir.exists():
        print("  Custom KG model not found. Run 07_custom_kg_embeddings.py first.")
        return

    print("Loading custom KG model...")
    model = torch.load(model_dir / "trained_model.pkl", map_location="cpu")
    model.eval()

    # Load custom KG to get entity names and types
    custom_kg_path = DATA_DIR / "custom_kg.tsv"
    triples = []
    with open(custom_kg_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) == 3:
                triples.append(parts)

    # Determine entity types from "is_a" relations
    entity_type_map = {}
    for h, r, t in triples:
        if r == "is_a":
            entity_type_map[h] = t

    # Extract embeddings
    with torch.no_grad():
        entity_embeddings = model.entity_representations[0]()
        if torch.is_complex(entity_embeddings):
            embeddings = torch.cat(
                [entity_embeddings.real, entity_embeddings.imag], dim=-1
            ).numpy()
        else:
            embeddings = entity_embeddings.numpy()

    # Get entity mapping from saved model metadata
    metadata_path = model_dir / "training_triples" / "entity_to_id.tsv"
    if not metadata_path.exists():
        # Try alternative path
        import glob
        tsv_files = glob.glob(str(model_dir / "**" / "entity_to_id*"), recursive=True)
        if tsv_files:
            metadata_path = Path(tsv_files[0])
        else:
            print("  Could not find entity mapping. Skipping custom KG visualization.")
            return

    id_to_entity = {}
    with open(metadata_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                id_to_entity[int(parts[1])] = parts[0]

    n_entities = len(id_to_entity)
    entity_names = [id_to_entity.get(i, f"entity_{i}") for i in range(n_entities)]
    entity_types = [entity_type_map.get(name, "Unknown") for name in entity_names]

    print(f"  Entities: {n_entities}")
    print(f"  Embedding shape: {embeddings.shape}")

    # ── t-SNE ───────────────────────────────────────────────────────
    effective_perplexity = min(perplexity, n_entities - 1)
    print(f"Running t-SNE (perplexity={effective_perplexity})...")
    tsne = TSNE(
        n_components=2,
        perplexity=max(2, effective_perplexity),
        random_state=42,
        n_iter=1000,
    )
    coords = tsne.fit_transform(embeddings[:n_entities])

    # ── Plot ────────────────────────────────────────────────────────
    type_set = sorted(set(entity_types))
    color_map = plt.cm.get_cmap("tab10", len(type_set))
    type_to_color = {t: color_map(i) for i, t in enumerate(type_set)}

    fig, ax = plt.subplots(figsize=(12, 9))

    for entity_type in type_set:
        mask = [i for i, t in enumerate(entity_types) if t == entity_type]
        if not mask:
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            label=entity_type,
            color=type_to_color[entity_type],
            alpha=0.8,
            s=80,
            edgecolors="white",
            linewidth=0.5,
        )
        # Label each point
        for i in mask:
            ax.annotate(
                entity_names[i],
                (coords[i, 0], coords[i, 1]),
                fontsize=7,
                alpha=0.8,
                xytext=(5, 5),
                textcoords="offset points",
            )

    ax.set_title(
        "Custom KG Entity Embeddings (TransE) - t-SNE Projection",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.legend(title="Entity Type", loc="best", fontsize=9)
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    chart_path = OUTPUT_DIR / "custom_kg_embeddings_tsne.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved to: {chart_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize KG Embeddings")
    parser.add_argument(
        "--model",
        type=str,
        default="transe",
        choices=["transe", "rotate", "complex"],
        help="Model to visualize",
    )
    parser.add_argument("--perplexity", type=int, default=30, help="t-SNE perplexity")
    parser.add_argument("--top-n", type=int, default=500, help="Number of entities to plot (benchmark)")
    args = parser.parse_args()

    print("=" * 60)
    print("KG Embedding Visualization")
    print("=" * 60)

    # ── Benchmark model visualization ───────────────────────────────
    print(f"\n--- FB15k-237 Embeddings ({args.model}) ---")
    visualize_benchmark_embeddings(args.model, args.perplexity, args.top_n)

    # ── Custom KG visualization ─────────────────────────────────────
    print(f"\n--- Custom KG Embeddings ---")
    visualize_custom_kg_embeddings(args.perplexity)

    print("\n" + "-" * 60)
    print("Visualization Insights")
    print("-" * 60)
    print("""
What to look for in the t-SNE plots:

1. Clustering by type: Entities of the same type (e.g., all movies,
   all people) should cluster together. This shows the embeddings
   capture semantic similarity.

2. Relation structure: Entities connected by the same relation type
   should be nearby. For example, countries and their capitals.

3. Outliers: Entities far from any cluster may be poorly embedded
   (too few triples) or genuinely unique.

4. Embedding quality: Better models (higher MRR) typically produce
   cleaner, more separated clusters.

Note: t-SNE is a stochastic method, so results vary between runs.
The perplexity parameter controls the balance between local and
global structure (higher = more global).
""")


if __name__ == "__main__":
    main()
