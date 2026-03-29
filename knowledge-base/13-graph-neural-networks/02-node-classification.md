# Node Classification on Knowledge Graphs

## Overview

Node classification is the task of predicting labels for nodes in a graph based on their features and neighborhood structure. In a KG context, this means predicting missing entity types, categories, or properties -- for example, classifying proteins by function, categorizing research papers by topic, or predicting user preferences.

---

## Task Definition

Given:
- A graph G = (V, E) with node features X
- Labels Y for a subset of nodes V_labeled (subset of V)

Predict:
- Labels for unlabeled nodes V_unlabeled = V \ V_labeled

This is **semi-supervised learning**: we use both labeled and unlabeled nodes during training, because the graph structure (edges) provides signal even for unlabeled nodes.

---

## Tools: PyTorch Geometric (PyG)

PyTorch Geometric is the most widely used library for GNNs in Python.

### Installation

```bash
pip install torch-geometric

# Install dependencies (match your PyTorch + CUDA version)
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
```

Reference: https://pytorch-geometric.readthedocs.io/

### Data Format

PyG uses `torch_geometric.data.Data` objects:

```python
import torch
from torch_geometric.data import Data

# Example: small graph with 4 nodes, 4 edges
edge_index = torch.tensor([
    [0, 1, 1, 2],  # source nodes
    [1, 0, 2, 3],  # target nodes
], dtype=torch.long)

x = torch.tensor([
    [1.0, 0.0, 0.0],  # node 0 features
    [0.0, 1.0, 0.0],  # node 1 features
    [0.0, 0.0, 1.0],  # node 2 features
    [1.0, 1.0, 0.0],  # node 3 features
], dtype=torch.float)

y = torch.tensor([0, 1, 1, 0], dtype=torch.long)  # node labels

data = Data(x=x, edge_index=edge_index, y=y)
print(data)
# Data(x=[4, 3], edge_index=[2, 4], y=[4])
```

---

## Building a KG-Like Dataset

Let us build a small knowledge graph for entity type prediction:

```python
import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder

# Simulated KG: entities with text-derived features and type labels
entities = [
    {"name": "Python", "type": "ProgrammingLanguage", "features": [1, 0, 0, 1, 0]},
    {"name": "Java", "type": "ProgrammingLanguage", "features": [1, 0, 0, 1, 1]},
    {"name": "Google", "type": "Company", "features": [0, 1, 0, 0, 1]},
    {"name": "Microsoft", "type": "Company", "features": [0, 1, 0, 0, 1]},
    {"name": "Guido", "type": "Person", "features": [0, 0, 1, 0, 0]},
    {"name": "Sundar", "type": "Person", "features": [0, 0, 1, 0, 0]},
    {"name": "TensorFlow", "type": "Library", "features": [1, 0, 0, 1, 0]},
    {"name": "PyTorch", "type": "Library", "features": [1, 0, 0, 1, 0]},
]

# Edges (undirected): index pairs
edges = [
    (0, 4),  # Python -- created_by -- Guido
    (1, 3),  # Java -- used_by -- Microsoft
    (2, 5),  # Google -- ceo -- Sundar
    (2, 6),  # Google -- develops -- TensorFlow
    (3, 1),  # Microsoft -- develops -- Java
    (6, 0),  # TensorFlow -- uses -- Python
    (7, 0),  # PyTorch -- uses -- Python
    (7, 3),  # PyTorch -- supported_by -- Microsoft
]

# Build PyG data object
x = torch.tensor([e["features"] for e in entities], dtype=torch.float)
edge_src = [e[0] for e in edges] + [e[1] for e in edges]  # bidirectional
edge_dst = [e[1] for e in edges] + [e[0] for e in edges]
edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)

le = LabelEncoder()
labels = le.fit_transform([e["type"] for e in entities])
y = torch.tensor(labels, dtype=torch.long)

data = Data(x=x, edge_index=edge_index, y=y)

# Train/test split masks
data.train_mask = torch.tensor([True, True, True, True, True, False, False, False])
data.test_mask = torch.tensor([False, False, False, False, False, True, True, True])

print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
print(f"Features: {data.num_node_features}")
print(f"Classes: {len(le.classes_)} ({list(le.classes_)})")
```

---

## GCN for Entity Type Prediction

### Model Definition

```python
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Layer 1: aggregate + transform + ReLU
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Layer 2: aggregate + transform (logits)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

num_classes = len(le.classes_)
model = GCN(
    in_channels=data.num_node_features,
    hidden_channels=16,
    out_channels=num_classes,
)
print(model)
```

### Training Loop

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate():
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)

        # Training accuracy
        train_correct = (pred[data.train_mask] == data.y[data.train_mask]).sum()
        train_acc = int(train_correct) / int(data.train_mask.sum())

        # Test accuracy
        test_correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        test_acc = int(test_correct) / int(data.test_mask.sum())

    return train_acc, test_acc

# Training
for epoch in range(1, 201):
    loss = train()
    if epoch % 20 == 0:
        train_acc, test_acc = evaluate()
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

---

## Using a Real Dataset: Cora

Cora is a citation network commonly used for node classification benchmarks:

```python
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root="/tmp/Cora", name="Cora")
data = dataset[0]

print(f"Nodes: {data.num_nodes}")           # 2708
print(f"Edges: {data.num_edges}")           # 10556
print(f"Features: {data.num_node_features}")  # 1433
print(f"Classes: {dataset.num_classes}")     # 7
print(f"Training nodes: {data.train_mask.sum()}")  # 140

# Train GCN on Cora
model = GCN(
    in_channels=dataset.num_node_features,
    hidden_channels=64,
    out_channels=dataset.num_classes,
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(1, 201):
    loss = train()

train_acc, test_acc = evaluate()
print(f"Final Test Accuracy: {test_acc:.4f}")
# Expected: ~0.81 for 2-layer GCN
```

---

## GAT for Node Classification

Replacing GCN with GAT often improves performance:

```python
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

model = GAT(
    in_channels=dataset.num_node_features,
    hidden_channels=8,
    out_channels=dataset.num_classes,
    heads=8,
)
# Expected Cora accuracy: ~0.83
```

---

## GraphSAGE for Scalable Classification

GraphSAGE uses neighbor sampling for mini-batch training on large graphs:

```python
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Mini-batch training with neighbor sampling
train_loader = NeighborLoader(
    data,
    num_neighbors=[10, 5],  # sample 10 neighbors at hop 1, 5 at hop 2
    batch_size=128,
    input_nodes=data.train_mask,
)

model = GraphSAGE(dataset.num_node_features, 64, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1, 51):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = F.nll_loss(out[:batch.batch_size], batch.y[:batch.batch_size])
        loss.backward()
        optimizer.step()
```

---

## Evaluation: Accuracy and F1

```python
from sklearn.metrics import classification_report, f1_score

model.eval()
with torch.no_grad():
    out = model(data)
    pred = out.argmax(dim=1)

    y_true = data.y[data.test_mask].cpu().numpy()
    y_pred = pred[data.test_mask].cpu().numpy()

# Detailed classification report
print(classification_report(y_true, y_pred, target_names=le.classes_))

# Macro F1 (treats all classes equally)
macro_f1 = f1_score(y_true, y_pred, average="macro")
print(f"Macro F1: {macro_f1:.4f}")

# Micro F1 (equivalent to accuracy for single-label)
micro_f1 = f1_score(y_true, y_pred, average="micro")
print(f"Micro F1: {micro_f1:.4f}")
```

---

## R-GCN for Multi-Relational Node Classification

For KGs with typed edges, use R-GCN:

```python
from torch_geometric.nn import RGCNConv

class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations):
        super().__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations,
                              num_bases=30)  # basis decomposition
        self.conv2 = RGCNConv(hidden_channels, out_channels, num_relations,
                              num_bases=30)

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return F.log_softmax(x, dim=1)

# edge_type is a tensor of relation IDs for each edge
model = RGCN(
    in_channels=data.num_node_features,
    hidden_channels=64,
    out_channels=num_classes,
    num_relations=num_relation_types,
)
```

---

## Complete End-to-End Example

Putting it all together with a clean workflow:

```python
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from sklearn.metrics import f1_score

# 1. Load data
dataset = Planetoid(root="/tmp/Cora", name="Cora")
data = dataset[0]

# 2. Define model
class GCN(torch.nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hid_ch)
        self.conv2 = GCNConv(hid_ch, out_ch)

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=1)

model = GCN(dataset.num_node_features, 64, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# 3. Train
best_val_acc = 0
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        pred = model(data).argmax(dim=1)
        val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")

# 4. Test
model.load_state_dict(torch.load("best_model.pt"))
model.eval()
with torch.no_grad():
    pred = model(data).argmax(dim=1)
    test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
    test_f1 = f1_score(
        data.y[data.test_mask].cpu(), pred[data.test_mask].cpu(), average="macro"
    )

print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Macro-F1: {test_f1:.4f}")
```

---

## Tips and Best Practices

1. **Start with GCN** as your baseline, then try GAT for attention and GraphSAGE for scalability
2. **Use 2 layers** unless you have evidence that deeper is better for your specific graph
3. **Hidden dimension**: 64-256 is typical; larger KGs may benefit from larger hidden dims
4. **Dropout**: 0.5-0.6 on both features and between layers
5. **Learning rate**: 0.01 with Adam is a reliable default
6. **Weight decay**: 5e-4 for L2 regularization
7. **Neighbor sampling** (GraphSAGE/NeighborLoader) is essential for graphs with >100K nodes
8. **Use R-GCN** when edge types carry important semantic information

---

## References

- PyTorch Geometric documentation: https://pytorch-geometric.readthedocs.io/
- PyG examples: https://github.com/pyg-team/pytorch_geometric/tree/master/examples
- Kipf & Welling (2017). "Semi-Supervised Classification with Graph Convolutional Networks."
- Hamilton et al. (2017). "Inductive Representation Learning on Large Graphs."
