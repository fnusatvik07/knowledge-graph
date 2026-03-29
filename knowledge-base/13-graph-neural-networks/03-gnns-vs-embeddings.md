# GNNs vs KG Embeddings: When to Use What

## Overview

Graph Neural Networks (GNNs) and Knowledge Graph embedding methods (TransE, RotatE, ComplEx, DistMult) are both approaches for learning representations of entities in a knowledge graph. They have different strengths, different assumptions, and excel at different tasks. This guide helps you choose.

---

## Fundamental Differences

### KG Embeddings (TransE, RotatE, ComplEx, DistMult)

- **Learn**: A fixed embedding vector for each entity and relation
- **Paradigm**: Score triples `f(h, r, t)` using algebraic operations on embeddings
- **Training signal**: Positive and negative triples only (no node features)
- **Prediction**: Score unseen triples for link prediction
- **Limitation**: Transductive -- cannot handle entities not seen during training

### GNNs (GCN, GraphSAGE, GAT, R-GCN)

- **Learn**: A function that computes embeddings from local neighborhood structure + node features
- **Paradigm**: Message passing -- aggregate neighbor information iteratively
- **Training signal**: Graph structure + node features + (optionally) edge features
- **Prediction**: Node classification, link prediction, graph classification
- **Advantage**: Inductive -- can generalize to unseen nodes and graphs

---

## Comparison Table

| Dimension | KG Embeddings | GNNs |
|-----------|--------------|------|
| **Representation** | Lookup table (one vector per entity) | Computed from neighborhood |
| **Node features** | Not used | Core input |
| **Relation types** | Natively handled | Requires R-GCN extension |
| **Inductive ability** | No (transductive only) | Yes (generalizes to new nodes) |
| **Primary task** | Link prediction | Node/graph classification + link prediction |
| **Scalability** | Scales to millions of entities | Memory-bound (sampling helps) |
| **Training speed** | Fast (simple operations) | Slower (neighborhood aggregation) |
| **Interpretability** | Limited (examine distances) | Moderate (attention weights in GAT) |
| **Theoretical grounding** | Translation/rotation geometry | Spectral graph theory, WL test |

---

## Task-Based Decision Guide

### Link Prediction

**Predicting missing edges: (h, r, ?) or (?, r, t)**

- **KG Embeddings win** when:
  - All entities are known at training time
  - Many relation types with diverse patterns (symmetric, composition, etc.)
  - KG has millions of entities (embeddings scale better)
  - You want fast inference (simple dot product / distance)

- **GNNs win** when:
  - New entities appear at inference time (inductive setting)
  - Entities have rich features (text descriptions, numerical attributes)
  - You want to leverage multi-hop neighborhood patterns

### Node Classification

**Predicting entity types, categories, or properties**

- **GNNs are the clear winner** for this task
- KG embeddings are not designed for node classification
- Semi-supervised GNNs can classify with very few labeled examples (leveraging graph structure)

### Entity Clustering / Similarity

- Both work well
- **KG embeddings**: cluster in the learned embedding space
- **GNNs**: use learned node representations for clustering
- GNNs are better when node features exist; embeddings are better for purely structural similarity

### Relation Prediction

**Given (h, ?, t), predict the relation type**

- **KG embeddings** handle this natively (score all relation types)
- GNNs require a specific decoder design for this task

---

## Decision Flowchart

```
                    START
                      |
                      v
        Do new entities appear at test time?
                /              \
              YES               NO
               |                 |
               v                 v
          Use GNNs         Is link prediction
      (inductive)          the primary task?
               |            /          \
               |          YES           NO
               |           |             |
               |           v             v
               |    Do you have      Node/Graph
               |    many relation    classification?
               |    types (>50)?         |
               |      /      \          YES
               |    YES       NO         |
               |     |         |         v
               |     v         v      Use GNNs
               |  Use KG     Either
               |  Embeddings  works
               |     |         |
               |     v         v
               |  (TransE/   Try both,
               |   RotatE)   benchmark
               |
               v
        Do entities have
        rich features?
          /          \
        YES           NO
         |             |
         v             v
     Use GNNs      Use GNNs
   (GAT/SAGE)    (with identity
                  features or
                  degree features)
```

---

## Hybrid Approaches: GNN Encoder + Embedding Decoder

The best of both worlds: use a GNN to compute entity representations, then score triples using an embedding-style decoder.

### Architecture

```
                     GNN Encoder              Embedding Decoder
                  (neighborhood-aware)        (relation-aware)

Node Features       R-GCN / GAT              TransE / DistMult
    |                   |                          |
    v                   v                          v
  [x_h] ---> GNN ---> [z_h]  ---+
                                 +---> f(z_h, r, z_t) ---> score
  [x_t] ---> GNN ---> [z_t]  ---+
                                 |
                            [r] (learned)
```

### Implementation Sketch

```python
import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv

class GNNLinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden, num_relations):
        super().__init__()
        # GNN encoder
        self.conv1 = RGCNConv(in_channels, hidden, num_relations)
        self.conv2 = RGCNConv(hidden, hidden, num_relations)

        # DistMult-style decoder
        self.rel_emb = nn.Embedding(num_relations, hidden)

    def encode(self, x, edge_index, edge_type):
        x = torch.relu(self.conv1(x, edge_index, edge_type))
        x = self.conv2(x, edge_index, edge_type)
        return x

    def decode(self, z_h, rel_id, z_t):
        # DistMult scoring: sum(h * r * t)
        r = self.rel_emb(rel_id)
        return (z_h * r * z_t).sum(dim=-1)

    def forward(self, x, edge_index, edge_type, head_idx, rel_idx, tail_idx):
        z = self.encode(x, edge_index, edge_type)
        return self.decode(z[head_idx], rel_idx, z[tail_idx])
```

### When to Use Hybrids

- You need both inductive capability AND relation-aware scoring
- You have node features but also care about diverse relation patterns
- State-of-the-art results on KG completion benchmarks increasingly use hybrid methods

---

## Computational Cost Comparison

### Training Time (approximate, 100K entities, 500K triples)

| Method | Time per Epoch | Memory | GPU Required? |
|--------|---------------|--------|---------------|
| TransE | ~10 seconds | ~1 GB | No (but helps) |
| RotatE | ~20 seconds | ~2 GB | Recommended |
| DistMult | ~8 seconds | ~1 GB | No |
| ComplEx | ~15 seconds | ~2 GB | Recommended |
| GCN (2-layer) | ~30 seconds | ~4 GB | Yes |
| R-GCN (2-layer) | ~60 seconds | ~8 GB | Yes |
| GAT (2-layer, 8 heads) | ~45 seconds | ~6 GB | Yes |
| GraphSAGE (mini-batch) | ~40 seconds | ~2 GB | Recommended |

### Scaling Behavior

```
Entities:     10K      100K      1M       10M
             ----     -----    -----    ------
TransE:       OK        OK       OK       OK
RotatE:       OK        OK       OK     Tight
GCN:          OK        OK     Hard    Infeasible
GraphSAGE:    OK        OK       OK     Possible
R-GCN:        OK      Tight    Hard    Infeasible
```

- **KG embeddings scale linearly** with entity count (just more rows in the embedding table)
- **Full-batch GNNs scale quadratically** in worst case (neighbor explosion)
- **Sampling-based GNNs (GraphSAGE)** scale much better via mini-batching

---

## Latest Trends

### Foundation Models for Graphs

Large pre-trained graph models are emerging, analogous to LLMs for text:

- **Graph-GPT**: Pre-train a graph transformer on many graphs, fine-tune on specific KGs
- **ULTRA**: Universal link prediction -- train once, predict on any KG (zero-shot)
- **GraphMAE**: Masked autoencoding for graph pre-training

### LLMs Meet KGs

- **LLMs as KG reasoners**: Use GPT-4 / Claude to perform multi-hop reasoning over KG triples
- **KGs as LLM grounding**: Retrieve structured facts from KGs to reduce hallucination
- **LLM-generated embeddings**: Use LLM text embeddings of entity descriptions as node features for GNNs

### GNN + Retrieval

- **Graph retrieval-augmented generation**: Use GNN-computed embeddings to retrieve relevant subgraphs for RAG
- This combines the structural awareness of GNNs with the generation capability of LLMs

---

## Practical Recommendations

### Start Here

1. **For link prediction on a static KG**: Start with RotatE (via PyKEEN). It handles most relation patterns and is well-understood
2. **For node classification**: Start with a 2-layer GCN (via PyG). Graduate to GAT if you need attention
3. **For evolving KGs with new entities**: Use GraphSAGE -- it generalizes inductively
4. **For production RAG systems**: Pre-compute RotatE embeddings for entity retrieval, use GNN only if you have node features

### Benchmark Both

When in doubt, try both approaches on your specific KG and task. The relative performance depends heavily on:
- KG size and density
- Number of relation types
- Availability of node features
- Whether new entities appear at inference time
- Computational budget

```python
# Quick comparison framework
from pykeen.pipeline import pipeline as pykeen_pipeline

# KG Embedding baseline
kg_result = pykeen_pipeline(
    training=your_triples,
    model="RotatE",
    training_kwargs=dict(num_epochs=100),
)
print(f"RotatE MRR: {kg_result.metric_results.get_metric('both.realistic.inverse_harmonic_mean_rank'):.4f}")

# GNN baseline (using PyG)
# ... train GCN/GAT and evaluate link prediction
# Compare MRR, Hits@10
```

---

## Summary

| If you need... | Use... |
|---------------|--------|
| Link prediction, static KG, many relations | KG Embeddings (RotatE) |
| Node classification | GNNs (GCN, GAT) |
| Inductive prediction (new entities) | GNNs (GraphSAGE) |
| Both structure and features | Hybrid (GNN encoder + embedding decoder) |
| Maximum scalability (>1M entities) | KG Embeddings or sampling-based GNNs |
| Interpretable predictions | GAT (attention weights) |
| Fast prototyping | KG Embeddings via PyKEEN |

---

## References

- Ali, M., et al. (2021). "PyKEEN 1.0." JMLR.
- Schlichtkrull, M., et al. (2018). "Modeling Relational Data with Graph Convolutional Networks." ESWC.
- Galkin, M., et al. (2023). "Towards Foundation Models for Knowledge Graph Reasoning." arXiv.
- PyKEEN: https://pykeen.readthedocs.io/
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
