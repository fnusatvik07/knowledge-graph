"""Train GAT (Graph Attention Network) for node classification.

Architecture:
    Input -> GATConv (multi-head) -> ELU -> Dropout -> GATConv (single head) -> output

GAT differs from GCN/GraphSAGE by:
    - Learning attention weights for each neighbor (not treating all equally)
    - Multi-head attention for stability (concatenate K attention heads)
    - Masked attention: only attends to neighbor nodes, not the full graph

References:
    Velickovic et al., "Graph Attention Networks" (2018)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GATConv
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
MODEL_PATH = OUTPUT_DIR / "gat_model.pt"
METRICS_PATH = OUTPUT_DIR / "gat_metrics.json"

HIDDEN_DIM = 16         # Per-head dimension
NUM_HEADS = 4           # Number of attention heads (layer 1)
OUTPUT_HEADS = 1        # Number of attention heads (output layer)
LEARNING_RATE = 0.005
WEIGHT_DECAY = 5e-4
DROPOUT = 0.6           # Slightly higher dropout — GAT is more expressive
EPOCHS = 200
SEED = 42


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class GAT(torch.nn.Module):
    """2-layer Graph Attention Network with multi-head attention.

    Layer 1: K attention heads, each producing HIDDEN_DIM features -> concat -> K*HIDDEN_DIM
    Layer 2: 1 attention head producing num_classes logits -> average over heads
    """

    def __init__(self, num_features: int, num_classes: int, hidden_dim: int = HIDDEN_DIM,
                 num_heads: int = NUM_HEADS, output_heads: int = OUTPUT_HEADS,
                 dropout: float = DROPOUT):
        super().__init__()
        self.dropout = dropout

        # Layer 1: multi-head attention
        # Output dim = hidden_dim * num_heads (concatenation of heads)
        self.conv1 = GATConv(
            num_features, hidden_dim,
            heads=num_heads,
            dropout=dropout,
        )

        # Layer 2: single-head attention for classification
        # Input dim = hidden_dim * num_heads (from concatenated layer 1)
        self.conv2 = GATConv(
            hidden_dim * num_heads, num_classes,
            heads=output_heads,
            concat=False,  # Average instead of concatenate for output
            dropout=dropout,
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)  # ELU activation (original GAT paper uses ELU, not ReLU)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        return x

    def get_embeddings(self, data):
        """Extract intermediate embeddings (after first attention layer)."""
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x)
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
    print("Step 4: Train GAT for Node Classification")
    print("=" * 60)

    torch.manual_seed(SEED)

    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found. Run 01_prepare_data.py first.")
        sys.exit(1)

    data = torch.load(DATA_PATH, weights_only=False)
    print(f"Loaded data: {data.num_nodes} nodes, {data.num_edges} edges, "
          f"{data.num_node_features} features, {data.num_classes} classes")

    model = GAT(
        num_features=data.num_node_features,
        num_classes=data.num_classes,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        output_heads=OUTPUT_HEADS,
        dropout=DROPOUT,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    print(f"\nModel: {model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    print(f"Attention heads: {NUM_HEADS} (layer 1), {OUTPUT_HEADS} (layer 2)")

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
    print(f"TEST RESULTS (GAT)")
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

    # Compare with previous models
    print(f"\n{'='*40}")
    print(f"COMPARISON")
    print(f"{'='*40}")
    for model_name, fname in [("GCN", "gcn_metrics.json"), ("GraphSAGE", "graphsage_metrics.json")]:
        path = OUTPUT_DIR / fname
        if path.exists():
            with open(path) as f:
                m = json.load(f)
            print(f"  {model_name:12s} — Acc: {m['test_accuracy']:.4f}, F1: {m['test_f1_macro']:.4f}")
    print(f"  {'GAT':12s} — Acc: {test_acc:.4f}, F1: {test_f1:.4f}")

    # Save metrics
    metrics = {
        "model": "GAT",
        "hidden_dim": HIDDEN_DIM,
        "num_heads": NUM_HEADS,
        "output_heads": OUTPUT_HEADS,
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
