"""Train GraphSAGE for node classification.

Architecture:
    Input -> SAGEConv -> ReLU -> Dropout -> SAGEConv -> LogSoftmax

GraphSAGE differs from GCN by:
    - Sampling a fixed-size neighborhood (instead of using all neighbors)
    - Concatenating self-features with aggregated neighbor features
    - Supporting inductive learning (can generalize to unseen nodes)

References:
    Hamilton et al., "Inductive Representation Learning on Large Graphs" (2017)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv
except ImportError:
    print("ERROR: PyTorch Geometric required. See README.md for installation.")
    sys.exit(1)

from sklearn.metrics import classification_report, f1_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
DATA_PATH = OUTPUT_DIR / "kg_data.pt"
MODEL_PATH = OUTPUT_DIR / "graphsage_model.pt"
METRICS_PATH = OUTPUT_DIR / "graphsage_metrics.json"

HIDDEN_DIM = 32
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
DROPOUT = 0.5
EPOCHS = 200
SEED = 42


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class GraphSAGE(torch.nn.Module):
    """2-layer GraphSAGE network.

    SAGEConv aggregates neighbor features using mean/LSTM/pool aggregation,
    then concatenates with the node's own features. This enables inductive
    learning — the model can generalize to unseen nodes.
    """

    def __init__(self, num_features: int, num_classes: int, hidden_dim: int = HIDDEN_DIM,
                 dropout: float = DROPOUT):
        super().__init__()
        self.conv1 = SAGEConv(num_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        return x

    def get_embeddings(self, data):
        """Extract intermediate embeddings (after first conv layer)."""
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return x


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_epoch(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out = model(data)
    pred = out[mask].argmax(dim=1)
    correct = (pred == data.y[mask]).sum().item()
    total = mask.sum().item()
    acc = correct / total if total > 0 else 0.0
    f1 = f1_score(data.y[mask].cpu(), pred.cpu(), average="macro", zero_division=0)
    return acc, f1


def main():
    print("=" * 60)
    print("Step 3: Train GraphSAGE for Node Classification")
    print("=" * 60)

    torch.manual_seed(SEED)

    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found. Run 01_prepare_data.py first.")
        sys.exit(1)

    data = torch.load(DATA_PATH, weights_only=False)
    print(f"Loaded data: {data.num_nodes} nodes, {data.num_edges} edges, "
          f"{data.num_node_features} features, {data.num_classes} classes")

    model = GraphSAGE(
        num_features=data.num_node_features,
        num_classes=data.num_classes,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    print(f"\nModel: {model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    print(f"\nTraining for {EPOCHS} epochs...")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Acc':>8} | {'Val F1':>8}")
    print("-" * 45)

    history = {"train_loss": [], "val_acc": [], "val_f1": []}
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, data, optimizer)
        val_acc, val_f1 = evaluate(model, data, data.val_mask)

        history["train_loss"].append(loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), MODEL_PATH)

        if epoch % 20 == 0 or epoch == 1:
            print(f"{epoch:>6} | {loss:>10.4f} | {val_acc:>8.4f} | {val_f1:>8.4f}")

    print(f"\nBest validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

    # Test evaluation
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    test_acc, test_f1 = evaluate(model, data, data.test_mask)

    print(f"\n{'='*40}")
    print(f"TEST RESULTS (GraphSAGE)")
    print(f"{'='*40}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Macro F1: {test_f1:.4f}")

    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out[data.test_mask].argmax(dim=1).cpu()
        true = data.y[data.test_mask].cpu()

    class_names = data.class_names if hasattr(data, "class_names") else None
    print(f"\nClassification Report:")
    print(classification_report(true, pred, target_names=class_names, zero_division=0))

    # Compare with GCN if available
    gcn_metrics_path = OUTPUT_DIR / "gcn_metrics.json"
    if gcn_metrics_path.exists():
        with open(gcn_metrics_path) as f:
            gcn_metrics = json.load(f)
        print(f"\nComparison with GCN:")
        print(f"  GCN       — Acc: {gcn_metrics['test_accuracy']:.4f}, F1: {gcn_metrics['test_f1_macro']:.4f}")
        print(f"  GraphSAGE — Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
        diff = test_acc - gcn_metrics["test_accuracy"]
        print(f"  Difference: {diff:+.4f} accuracy")

    # Save metrics
    metrics = {
        "model": "GraphSAGE",
        "hidden_dim": HIDDEN_DIM,
        "epochs": EPOCHS,
        "best_epoch": best_epoch,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "best_val_accuracy": best_val_acc,
        "total_params": total_params,
        "history": history,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {METRICS_PATH}")
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
