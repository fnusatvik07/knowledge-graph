"""Use the best trained GNN model to predict types for unlabeled entities.

This is the practical payoff: given a KG with some entities missing their type
labels, use the trained GNN to predict types with confidence scores.

Workflow:
    1. Load the best model (highest test accuracy from comparison)
    2. Load the graph data
    3. Run forward pass to get predictions for ALL nodes
    4. Show predictions vs. true labels for test set (sanity check)
    5. Simulate "unknown" entities and predict their types
    6. Show top predictions ranked by confidence
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, SAGEConv, GATConv
except ImportError:
    print("ERROR: PyTorch Geometric required. See README.md for installation.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
DATA_PATH = OUTPUT_DIR / "kg_data.pt"

# Import model classes (duplicated here for standalone use)
from _02_train_gcn import GCN  # noqa: E402  -- we try relative import first


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------
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


def load_model(model_name: str, data, metrics: dict):
    """Instantiate and load the best model."""
    num_features = data.num_node_features
    num_classes = data.num_classes
    hidden_dim = metrics["hidden_dim"]

    if model_name == "GCN":
        from _02_train_gcn import GCN
        model = GCN(num_features, num_classes, hidden_dim)
        model.load_state_dict(torch.load(OUTPUT_DIR / "gcn_model.pt", weights_only=True))
    elif model_name == "GraphSAGE":
        from _03_train_graphsage import GraphSAGE
        model = GraphSAGE(num_features, num_classes, hidden_dim)
        model.load_state_dict(torch.load(OUTPUT_DIR / "graphsage_model.pt", weights_only=True))
    elif model_name == "GAT":
        from _04_train_gat import GAT
        model = GAT(num_features, num_classes, hidden_dim,
                     num_heads=metrics.get("num_heads", 4),
                     output_heads=metrics.get("output_heads", 1))
        model.load_state_dict(torch.load(OUTPUT_DIR / "gat_model.pt", weights_only=True))
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def load_model_standalone(model_name: str, data, metrics: dict):
    """Load model with inline class definitions (no imports needed)."""
    num_features = data.num_node_features
    num_classes = data.num_classes
    hidden_dim = metrics["hidden_dim"]

    if model_name == "GCN":
        class GCNModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = GCNConv(num_features, hidden_dim)
                self.conv2 = GCNConv(hidden_dim, num_classes)
            def forward(self, data):
                x = self.conv1(data.x, data.edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
                x = self.conv2(x, data.edge_index)
                return x
        model = GCNModel()
        model.load_state_dict(torch.load(OUTPUT_DIR / "gcn_model.pt", weights_only=True))

    elif model_name == "GraphSAGE":
        class SAGEModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = SAGEConv(num_features, hidden_dim)
                self.conv2 = SAGEConv(hidden_dim, num_classes)
            def forward(self, data):
                x = self.conv1(data.x, data.edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
                x = self.conv2(x, data.edge_index)
                return x
        model = SAGEModel()
        model.load_state_dict(torch.load(OUTPUT_DIR / "graphsage_model.pt", weights_only=True))

    elif model_name == "GAT":
        num_heads = metrics.get("num_heads", 4)
        output_heads = metrics.get("output_heads", 1)
        class GATModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = GATConv(num_features, hidden_dim, heads=num_heads, dropout=0.6)
                self.conv2 = GATConv(hidden_dim * num_heads, num_classes,
                                     heads=output_heads, concat=False, dropout=0.6)
            def forward(self, data):
                x = F.dropout(data.x, p=0.6, training=self.training)
                x = self.conv1(x, data.edge_index)
                x = F.elu(x)
                x = F.dropout(x, p=0.6, training=self.training)
                x = self.conv2(x, data.edge_index)
                return x
        model = GATModel()
        model.load_state_dict(torch.load(OUTPUT_DIR / "gat_model.pt", weights_only=True))

    return model


def main():
    print("=" * 60)
    print("Step 6: Predict Unknown Entity Types")
    print("=" * 60)

    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found. Run 01_prepare_data.py first.")
        sys.exit(1)

    data = torch.load(DATA_PATH, weights_only=False)
    class_names = data.class_names if hasattr(data, "class_names") else [str(i) for i in range(data.num_classes)]
    entity_names = data.entity_names if hasattr(data, "entity_names") else [f"Node_{i}" for i in range(data.num_nodes)]

    # Find and load the best model
    best_name, best_metrics = find_best_model()
    if best_name is None:
        print("ERROR: No trained models found. Run training scripts first.")
        sys.exit(1)

    print(f"\nBest model: {best_name} (test acc: {best_metrics['test_accuracy']:.4f})")
    print("Loading model...")
    model = load_model_standalone(best_name, data, best_metrics)
    model.eval()

    # Run predictions
    with torch.no_grad():
        logits = model(data)
        probs = F.softmax(logits, dim=1)
        predictions = probs.argmax(dim=1)
        confidences = probs.max(dim=1).values

    # --- Sanity check: test set predictions ---
    print(f"\n{'='*50}")
    print("SANITY CHECK: Test Set Predictions")
    print(f"{'='*50}")
    test_indices = data.test_mask.nonzero(as_tuple=True)[0]
    correct = 0
    for idx in test_indices:
        i = idx.item()
        true_label = class_names[data.y[i].item()]
        pred_label = class_names[predictions[i].item()]
        conf = confidences[i].item()
        match = "OK" if true_label == pred_label else "MISS"
        print(f"  [{match:4s}] {entity_names[i]:25s} | True: {true_label:15s} | "
              f"Pred: {pred_label:15s} | Conf: {conf:.3f}")
        if match == "OK":
            correct += 1
    print(f"\nTest accuracy: {correct}/{len(test_indices)} = {correct/len(test_indices):.4f}")

    # --- Simulate unknown entities ---
    # Treat validation set as "unknown" to demonstrate prediction
    print(f"\n{'='*50}")
    print("PREDICTIONS FOR 'UNKNOWN' ENTITIES")
    print("(Using validation set as stand-in for unlabeled nodes)")
    print(f"{'='*50}")

    val_indices = data.val_mask.nonzero(as_tuple=True)[0]

    # Collect predictions and sort by confidence
    pred_results = []
    for idx in val_indices:
        i = idx.item()
        pred_label = class_names[predictions[i].item()]
        conf = confidences[i].item()
        # Top-2 predictions
        top2_probs, top2_indices = probs[i].topk(2)
        second_label = class_names[top2_indices[1].item()]
        second_conf = top2_probs[1].item()
        pred_results.append({
            "entity": entity_names[i],
            "predicted_type": pred_label,
            "confidence": conf,
            "second_type": second_label,
            "second_confidence": second_conf,
            "true_type": class_names[data.y[i].item()],
        })

    # Sort by confidence (highest first)
    pred_results.sort(key=lambda x: x["confidence"], reverse=True)

    print(f"\n{'Entity':25s} | {'Predicted':15s} | {'Conf':>6s} | {'2nd Choice':15s} | "
          f"{'2nd Conf':>8s} | {'True':15s}")
    print("-" * 100)
    for r in pred_results:
        match = "+" if r["predicted_type"] == r["true_type"] else "-"
        print(f"{match} {r['entity']:24s} | {r['predicted_type']:15s} | {r['confidence']:>6.3f} | "
              f"{r['second_type']:15s} | {r['second_confidence']:>8.3f} | {r['true_type']:15s}")

    # Summary statistics
    high_conf = [r for r in pred_results if r["confidence"] > 0.8]
    low_conf = [r for r in pred_results if r["confidence"] < 0.5]
    print(f"\nSummary:")
    print(f"  High confidence (>0.8): {len(high_conf)}/{len(pred_results)} predictions")
    print(f"  Low confidence (<0.5):  {len(low_conf)}/{len(pred_results)} predictions")
    print(f"  Average confidence:     {sum(r['confidence'] for r in pred_results)/len(pred_results):.3f}")

    # --- All node predictions ---
    print(f"\n{'='*50}")
    print("FULL GRAPH PREDICTIONS (All Nodes)")
    print(f"{'='*50}")
    for class_name in class_names:
        pred_count = (predictions == class_names.index(class_name)).sum().item()
        true_count = (data.y == class_names.index(class_name)).sum().item()
        print(f"  {class_name:15s} — Predicted: {pred_count:3d}, True: {true_count:3d}")

    print("\nDone! The GNN has learned to predict entity types from graph structure.")
    print("In a real KG, you would use these predictions to fill in missing type labels.")


if __name__ == "__main__":
    main()
