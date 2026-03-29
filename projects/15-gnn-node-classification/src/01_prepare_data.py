"""Convert KG JSON dataset to PyTorch Geometric Data format.

Builds node features (text embeddings or one-hot), edge_index tensor,
label tensor, and train/val/test masks. Saves as .pt file.
"""

import json
import sys
from pathlib import Path

# -- path setup for shared imports --
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np

try:
    import torch
    from torch_geometric.data import Data
except ImportError:
    print("ERROR: PyTorch Geometric is required but not installed.")
    print()
    print("Install PyTorch first:")
    print("  pip install torch torchvision torchaudio")
    print()
    print("Then install PyTorch Geometric:")
    print("  pip install torch_geometric")
    print()
    print("For detailed instructions (CUDA versions, etc.):")
    print("  https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = DATA_DIR / "kg_dataset.json"
OUTPUT_PATH = OUTPUT_DIR / "kg_data.pt"

USE_LLM_EMBEDDINGS = False  # Set True to use LLM embeddings (slower but richer)
EMBEDDING_DIM = 64          # Dimension for one-hot fallback (padded/truncated)

LABEL_MAP = {"PERSON": 0, "ORGANIZATION": 1, "TECHNOLOGY": 2, "CONCEPT": 3}

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------


def load_kg_dataset(path: Path) -> dict:
    """Load the KG JSON dataset."""
    with open(path) as f:
        data = json.load(f)
    print(f"Loaded KG: {len(data['entities'])} entities, {len(data['relationships'])} relationships")
    return data


def build_node_features_onehot(entities: list) -> torch.Tensor:
    """Build one-hot encoded node features (identity matrix, padded/truncated)."""
    n = len(entities)
    dim = max(n, EMBEDDING_DIM)
    features = torch.eye(dim)[:n]  # (n, dim) — each node gets a unique one-hot
    print(f"Built one-hot features: shape {features.shape}")
    return features


def build_node_features_llm(entities: list) -> torch.Tensor:
    """Build node features using LLM text embeddings of entity descriptions."""
    from shared.llm_clients import get_embedding

    embeddings = []
    for i, entity in enumerate(entities):
        text = f"{entity['name']}: {entity['description']}"
        emb = get_embedding(text)
        embeddings.append(emb)
        if (i + 1) % 10 == 0:
            print(f"  Embedded {i + 1}/{len(entities)} entities...")

    features = torch.tensor(np.array(embeddings), dtype=torch.float)
    print(f"Built LLM embedding features: shape {features.shape}")
    return features


def build_edge_index(relationships: list, id_to_idx: dict) -> torch.Tensor:
    """Build edge_index tensor from relationships.

    Creates bidirectional edges (undirected graph) for better message passing.
    """
    sources = []
    targets = []
    skipped = 0

    for rel in relationships:
        src_id = rel["source"]
        tgt_id = rel["target"]
        if src_id not in id_to_idx or tgt_id not in id_to_idx:
            skipped += 1
            continue
        src_idx = id_to_idx[src_id]
        tgt_idx = id_to_idx[tgt_id]
        # Add both directions for undirected message passing
        sources.extend([src_idx, tgt_idx])
        targets.extend([tgt_idx, src_idx])

    if skipped:
        print(f"  Warning: skipped {skipped} edges with unknown entity IDs")

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    print(f"Built edge_index: {edge_index.shape[1]} edges (bidirectional)")
    return edge_index


def build_labels(entities: list) -> torch.Tensor:
    """Build label tensor from entity types."""
    labels = []
    for entity in entities:
        label = LABEL_MAP.get(entity.get("type"), -1)
        if label == -1:
            print(f"  Warning: unknown type '{entity.get('type')}' for {entity['name']}")
        labels.append(label)
    return torch.tensor(labels, dtype=torch.long)


def create_masks(n: int, train_ratio: float = 0.70, val_ratio: float = 0.15,
                 seed: int = 42) -> tuple:
    """Create train/val/test boolean masks with stratification-aware random split."""
    torch.manual_seed(seed)
    perm = torch.randperm(n)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train:n_train + n_val]] = True
    test_mask[perm[n_train + n_val:]] = True

    print(f"Masks — train: {train_mask.sum().item()}, val: {val_mask.sum().item()}, "
          f"test: {test_mask.sum().item()}")
    return train_mask, val_mask, test_mask


def main():
    print("=" * 60)
    print("Step 1: Prepare PyTorch Geometric Data from KG JSON")
    print("=" * 60)

    # Load
    dataset = load_kg_dataset(DATASET_PATH)
    entities = dataset["entities"]
    relationships = dataset["relationships"]

    # Entity ID -> index mapping
    id_to_idx = {e["id"]: i for i, e in enumerate(entities)}

    # Node features
    print("\nBuilding node features...")
    if USE_LLM_EMBEDDINGS:
        x = build_node_features_llm(entities)
    else:
        x = build_node_features_onehot(entities)

    # Edge index
    print("\nBuilding edge index...")
    edge_index = build_edge_index(relationships, id_to_idx)

    # Labels
    print("\nBuilding labels...")
    y = build_labels(entities)
    for label_name, label_id in LABEL_MAP.items():
        count = (y == label_id).sum().item()
        print(f"  {label_name}: {count} entities")

    # Masks
    print("\nCreating train/val/test masks (70/15/15)...")
    train_mask, val_mask, test_mask = create_masks(len(entities))

    # Build PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    # Store metadata for later use
    data.num_classes = len(LABEL_MAP)
    data.class_names = list(LABEL_MAP.keys())
    data.entity_names = [e["name"] for e in entities]
    data.entity_ids = [e["id"] for e in entities]
    data.label_map = LABEL_MAP

    print(f"\nPyG Data object:")
    print(f"  {data}")
    print(f"  Num nodes: {data.num_nodes}")
    print(f"  Num edges: {data.num_edges}")
    print(f"  Num features: {data.num_node_features}")
    print(f"  Num classes: {data.num_classes}")
    print(f"  Has isolated nodes: {data.has_isolated_nodes()}")
    print(f"  Has self-loops: {data.has_self_loops()}")
    print(f"  Is undirected: {data.is_undirected()}")

    # Save
    torch.save(data, OUTPUT_PATH)
    print(f"\nSaved to {OUTPUT_PATH}")
    print("Done!")


if __name__ == "__main__":
    main()
