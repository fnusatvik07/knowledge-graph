# Project 15: GNN Node Classification on Knowledge Graphs

## Overview

Train Graph Neural Networks (GCN, GraphSAGE, GAT) for **node classification** on a
knowledge graph. Given a KG where some entity types are known (PERSON, ORGANIZATION,
TECHNOLOGY, CONCEPT), predict the type of entities with missing labels.

This is the practical bridge between the 600+ lines of GNN theory in the Knowledge Base
and actual working code. We go from JSON knowledge graph to trained PyTorch Geometric
models to predictions on unknown entities.

## What You'll Build

| Script | Purpose |
|--------|---------|
| `01_prepare_data.py` | Convert KG JSON to PyTorch Geometric `Data` object |
| `02_train_gcn.py` | Train a 2-layer GCN (Graph Convolutional Network) |
| `03_train_graphsage.py` | Train GraphSAGE with neighborhood sampling |
| `04_train_gat.py` | Train GAT with multi-head attention |
| `05_model_comparison.py` | Compare all 3 models side by side |
| `06_predict_unknown.py` | Predict types for unlabeled entities |
| `07_visualize_embeddings.py` | t-SNE visualization of learned embeddings |

## Architecture

```
kg_dataset.json
      |
      v
01_prepare_data.py  -->  kg_data.pt (PyTorch Geometric Data)
      |
      v
02/03/04_train_*.py -->  models/ + metrics/ (saved checkpoints + JSON metrics)
      |
      v
05_model_comparison.py --> comparison chart (matplotlib)
      |
      v
06_predict_unknown.py  --> predictions with confidence scores
      |
      v
07_visualize_embeddings.py --> t-SNE plot colored by entity type
```

## Prerequisites

```bash
pip install torch torch_geometric matplotlib scikit-learn
```

**PyTorch Geometric installation** can be tricky depending on your CUDA version.
See: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

For CPU-only (simplest):
```bash
pip install torch torchvision torchaudio
pip install torch_geometric
```

## Dataset

`data/kg_dataset.json` contains ~60 entities and ~80 relationships forming a knowledge
graph about AI/ML topics. Entity types serve as classification labels:

- **PERSON** — Researchers, engineers, founders
- **ORGANIZATION** — Companies, universities, labs
- **TECHNOLOGY** — Frameworks, models, algorithms
- **CONCEPT** — Abstract ideas, methodologies, fields

## Key Concepts

- **Node features**: Either one-hot encoded entity names or LLM-generated text embeddings
  of entity descriptions (via shared LLM layer)
- **Message passing**: Each GNN layer aggregates neighbor information — after 2 layers,
  each node "sees" its 2-hop neighborhood
- **Transductive learning**: All nodes (including test) are in the graph during training,
  but test labels are masked

## Usage

Run scripts in order:
```bash
python src/01_prepare_data.py
python src/02_train_gcn.py
python src/03_train_graphsage.py
python src/04_train_gat.py
python src/05_model_comparison.py
python src/06_predict_unknown.py
python src/07_visualize_embeddings.py
```

## Connection to Knowledge Base

- **KB Section 08**: GNN foundations, message passing, spectral vs. spatial
- **KB Section 09**: Graph embeddings (node2vec, TransE) — compare with GNN-learned embeddings
- **KB Section 16**: Agentic patterns — GNN predictions can feed agent decisions
