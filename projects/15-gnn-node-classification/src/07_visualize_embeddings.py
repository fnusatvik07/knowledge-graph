"""Visualize learned GNN node embeddings using t-SNE.

Extracts intermediate representations from the best trained GNN model,
reduces to 2D with t-SNE, and plots with matplotlib. Nodes are colored
by their true entity type label.

The key insight: if the GNN learned meaningful representations, entities
of the same type should cluster together in the embedding space — even though
the model was only trained on graph structure and node features, not on
any explicit similarity metric.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, SAGEConv, GATConv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print("Install: pip install torch torch_geometric matplotlib scikit-learn")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
DATA_PATH = OUTPUT_DIR / "kg_data.pt"

COLORS = {
    "PERSON": "#E53935",
    "ORGANIZATION": "#1E88E5",
    "TECHNOLOGY": "#43A047",
    "CONCEPT": "#FB8C00",
}

TSNE_PERPLEXITY = 10  # Lower for small datasets (default 30 is for larger ones)
TSNE_SEED = 42


def find_best_model() -> tuple:
    """Find the best model based on saved metrics."""
    best_acc = -1
    best_name = None
    best_metrics = None
    for name, fname in [("GCN", "gcn_metrics.json"),
                        ("GraphSAGE", "graphsage_metrics.json"),
                        ("GAT", "gat_metrics.json")]:
        path = OUTPUT_DIR / fname
        if path.exists():
            with open(path) as f:
                m = json.load(f)
            if m["test_accuracy"] > best_acc:
                best_acc = m["test_accuracy"]
                best_name = name
                best_metrics = m
    return best_name, best_metrics


def load_model_and_get_embeddings(model_name: str, data, metrics: dict) -> np.ndarray:
    """Load model and extract intermediate embeddings."""
    num_features = data.num_node_features
    num_classes = data.num_classes
    hidden_dim = metrics["hidden_dim"]

    if model_name == "GCN":
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = GCNConv(num_features, hidden_dim)
                self.conv2 = GCNConv(hidden_dim, num_classes)
            def get_embeddings(self, data):
                x = self.conv1(data.x, data.edge_index)
                return F.relu(x)
            def forward(self, data):
                x = self.get_embeddings(data)
                x = F.dropout(x, p=0.5, training=self.training)
                return self.conv2(x, data.edge_index)
        model = Model()
        model.load_state_dict(torch.load(OUTPUT_DIR / "gcn_model.pt", weights_only=True))

    elif model_name == "GraphSAGE":
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = SAGEConv(num_features, hidden_dim)
                self.conv2 = SAGEConv(hidden_dim, num_classes)
            def get_embeddings(self, data):
                x = self.conv1(data.x, data.edge_index)
                return F.relu(x)
            def forward(self, data):
                x = self.get_embeddings(data)
                x = F.dropout(x, p=0.5, training=self.training)
                return self.conv2(x, data.edge_index)
        model = Model()
        model.load_state_dict(torch.load(OUTPUT_DIR / "graphsage_model.pt", weights_only=True))

    elif model_name == "GAT":
        num_heads = metrics.get("num_heads", 4)
        output_heads = metrics.get("output_heads", 1)
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = GATConv(num_features, hidden_dim, heads=num_heads, dropout=0.6)
                self.conv2 = GATConv(hidden_dim * num_heads, num_classes,
                                     heads=output_heads, concat=False, dropout=0.6)
            def get_embeddings(self, data):
                x = self.conv1(data.x, data.edge_index)
                return F.elu(x)
            def forward(self, data):
                x = F.dropout(data.x, p=0.6, training=self.training)
                x = self.conv1(x, data.edge_index)
                x = F.elu(x)
                x = F.dropout(x, p=0.6, training=self.training)
                return self.conv2(x, data.edge_index)
        model = Model()
        model.load_state_dict(torch.load(OUTPUT_DIR / "gat_model.pt", weights_only=True))

    model.eval()
    with torch.no_grad():
        embeddings = model.get_embeddings(data).cpu().numpy()

    return embeddings


def plot_embeddings(embeddings_2d: np.ndarray, labels: np.ndarray,
                    class_names: list, entity_names: list, model_name: str):
    """Create t-SNE visualization with labels and coloring."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))

    for class_idx, class_name in enumerate(class_names):
        mask = labels == class_idx
        color = COLORS.get(class_name, "#999999")
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=color,
            label=f"{class_name} ({mask.sum()})",
            s=120,
            alpha=0.8,
            edgecolors="white",
            linewidth=0.5,
        )

    # Add entity name labels
    for i, name in enumerate(entity_names):
        # Shorten long names
        short_name = name if len(name) <= 15 else name[:13] + ".."
        ax.annotate(
            short_name,
            (embeddings_2d[i, 0], embeddings_2d[i, 1]),
            fontsize=6,
            alpha=0.7,
            ha="center",
            va="bottom",
            xytext=(0, 5),
            textcoords="offset points",
        )

    ax.set_title(f"GNN Node Embeddings (t-SNE) — {model_name}\n"
                 f"Entities colored by type, clustered by learned graph representation",
                 fontsize=13)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    chart_path = OUTPUT_DIR / "embedding_tsne.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved t-SNE plot to {chart_path}")


def compute_cluster_quality(embeddings_2d: np.ndarray, labels: np.ndarray):
    """Compute simple cluster quality metrics."""
    from sklearn.metrics import silhouette_score

    try:
        score = silhouette_score(embeddings_2d, labels)
        print(f"\nCluster Quality (Silhouette Score): {score:.4f}")
        print(f"  (1.0 = perfect clusters, 0.0 = overlapping, <0 = wrong clusters)")
        if score > 0.5:
            print(f"  The GNN learned clearly separable embeddings for entity types.")
        elif score > 0.2:
            print(f"  The GNN learned partially separable embeddings.")
        else:
            print(f"  Embeddings overlap significantly — graph structure alone may not")
            print(f"  fully distinguish entity types. Consider using LLM text embeddings")
            print(f"  as node features (set USE_LLM_EMBEDDINGS=True in 01_prepare_data.py).")
    except Exception as e:
        print(f"  Could not compute silhouette score: {e}")


def main():
    print("=" * 60)
    print("Step 7: Visualize GNN Node Embeddings (t-SNE)")
    print("=" * 60)

    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found. Run 01_prepare_data.py first.")
        sys.exit(1)

    data = torch.load(DATA_PATH, weights_only=False)
    class_names = data.class_names if hasattr(data, "class_names") else [str(i) for i in range(data.num_classes)]
    entity_names = data.entity_names if hasattr(data, "entity_names") else [f"Node_{i}" for i in range(data.num_nodes)]

    # Find best model
    best_name, best_metrics = find_best_model()
    if best_name is None:
        print("ERROR: No trained models found. Run training scripts first.")
        sys.exit(1)

    print(f"\nUsing best model: {best_name} (test acc: {best_metrics['test_accuracy']:.4f})")

    # Extract embeddings
    print("Extracting node embeddings from trained model...")
    embeddings = load_model_and_get_embeddings(best_name, data, best_metrics)
    print(f"Embedding shape: {embeddings.shape}")

    # t-SNE reduction
    print(f"\nRunning t-SNE (perplexity={TSNE_PERPLEXITY})...")
    tsne = TSNE(
        n_components=2,
        perplexity=min(TSNE_PERPLEXITY, len(embeddings) - 1),
        random_state=TSNE_SEED,
        n_iter=1000,
    )
    embeddings_2d = tsne.fit_transform(embeddings)
    print(f"Reduced to shape: {embeddings_2d.shape}")

    # Plot
    print("\nGenerating visualization...")
    labels = data.y.cpu().numpy()
    plot_embeddings(embeddings_2d, labels, class_names, entity_names, best_name)

    # Cluster quality
    compute_cluster_quality(embeddings_2d, labels)

    print("\nDone!")
    print("\nKey Insight: If same-type entities cluster together in the t-SNE plot,")
    print("the GNN has learned that graph structure encodes entity type information.")
    print("This is WHY GNNs are powerful for KGs — they learn from connectivity patterns.")


if __name__ == "__main__":
    main()
