# Graph Neural Networks for Knowledge Graphs: Fundamentals

## Overview

Graph Neural Networks (GNNs) are deep learning models designed to operate directly on graph-structured data. Unlike KG embedding methods (TransE, RotatE) that learn static embeddings, GNNs learn representations by iteratively aggregating information from a node's neighborhood. This makes them powerful for tasks where local structure and node features matter.

---

## Why GNNs for Knowledge Graphs?

Traditional KG embedding methods are **transductive**: they can only make predictions about entities seen during training. GNNs offer key advantages:

1. **Inductive learning**: GNNs can generalize to unseen nodes and even unseen graphs
2. **Node features**: GNNs can incorporate entity attributes (text descriptions, numerical properties)
3. **Rich aggregation**: Multi-hop neighborhood information is captured through message passing layers
4. **Flexibility**: Same architecture can handle node classification, link prediction, and graph classification

---

## The Message Passing Paradigm

All GNNs follow the same fundamental pattern: **message passing**. Each layer updates node representations by aggregating information from neighbors.

### Three Steps Per Layer

```
For each node v, at layer l:

1. AGGREGATE:  m_v^(l) = AGG({h_u^(l-1) : u in N(v)})
                          Collect messages from neighbors

2. UPDATE:     h_v^(l) = UPD(h_v^(l-1), m_v^(l))
                          Combine own representation with aggregated messages

3. READOUT:    z_G = READ({h_v^(L) : v in V})  (for graph-level tasks)
                          Pool all final node representations
```

### Intuition: Information Diffusion

Each message passing layer allows information to travel one hop. With L layers, each node's representation captures information from its L-hop neighborhood:

```
Layer 0: Node knows only itself
         [A]

Layer 1: Node knows its direct neighbors
         [B]--[A]--[C]

Layer 2: Node knows neighbors of neighbors
    [D]--[B]--[A]--[C]--[E]

Layer 3: Node knows 3-hop neighborhood
[F]--[D]--[B]--[A]--[C]--[E]--[G]
```

### The Over-Smoothing Problem

Too many layers cause all node representations to converge to the same value. In practice, 2-4 layers work best for most tasks. This is a fundamental limitation of GNNs.

---

## GCN: Graph Convolutional Networks

### Origin

GCN (Kipf & Welling, 2017) adapts convolutional neural networks to graphs using spectral graph theory.

### Formula

```
H^(l+1) = sigma(D_hat^(-1/2) * A_hat * D_hat^(-1/2) * H^(l) * W^(l))
```

where:
- `A_hat = A + I` (adjacency matrix with self-loops)
- `D_hat` is the degree matrix of `A_hat`
- `H^(l)` is the node feature matrix at layer l
- `W^(l)` is the learnable weight matrix
- `sigma` is an activation function (ReLU)

### Simplified Per-Node View

For a single node v:

```
h_v^(l+1) = sigma(W^(l) * SUM_{u in N(v) + {v}} (1 / sqrt(deg(u) * deg(v))) * h_u^(l))
```

Each neighbor's features are scaled by inverse degree (symmetric normalization), summed, then transformed.

### Architecture Diagram

```
Input Features    Layer 1          Layer 2         Output
                  (Aggregate +     (Aggregate +
                   Transform)       Transform)

  [x_A] ----+                +---+
             |  Aggregate    |   |  Aggregate
  [x_B] ----+---> [h_A^1] --+   +---> [h_A^2] ---> Prediction
             |               |   |
  [x_C] ----+               +---+
             |               |
  [x_D] ----+               |
                             |
  [x_B] ----+               |
             |  Aggregate    |
  [x_A] ----+---> [h_B^1] --+
             |               |
  [x_D] ----+               +---> [h_B^2] ---> Prediction
                             |
              ...            ...
```

### Strengths and Weaknesses

- **Strengths**: Simple, efficient, well-understood theoretically
- **Weaknesses**: Fixed aggregation weights (no attention), requires full graph in memory (transductive)

---

## GraphSAGE: Sampling-Based Inductive Learning

### Key Innovation

GraphSAGE (Hamilton et al., 2017) solves two problems with GCN:
1. **Scalability**: Instead of using the full graph, it samples a fixed number of neighbors
2. **Inductive**: Learns an aggregation function, not fixed embeddings -- works on unseen nodes

### Algorithm

```
For each node v, at layer l:

1. Sample K neighbors: N_sample(v) = SAMPLE(N(v), K)

2. Aggregate:   m_v^(l) = AGG({h_u^(l-1) : u in N_sample(v)})
                           (mean, max-pool, or LSTM aggregator)

3. Update:      h_v^(l) = sigma(W^(l) * CONCAT(h_v^(l-1), m_v^(l)))

4. Normalize:   h_v^(l) = h_v^(l) / ||h_v^(l)||
```

### Aggregator Options

```
Mean:      AGG = (1/|N|) * SUM h_u
Max-Pool:  AGG = MAX(sigma(W_pool * h_u + b))
LSTM:      AGG = LSTM(PERMUTE({h_u}))  (order-invariant via random permutation)
```

### Why GraphSAGE Works for KGs

In a KG with new entities appearing over time (e.g., new products, new users), GraphSAGE can generate embeddings for these entities without retraining, as long as they are connected to existing entities.

```
Training Graph:               Inference (new node X):

  A---B---C                     A---B---C
  |       |                     |       |
  D---E---F                     D---E---F
                                    |
                                    X  <-- NEW (no retraining needed)
```

---

## GAT: Graph Attention Networks

### Key Innovation

GAT (Velickovic et al., 2018) introduces **attention** to neighbor aggregation. Not all neighbors are equally important -- attention weights learn which neighbors matter most.

### Attention Mechanism

```
For each edge (v, u):

1. Compute attention coefficient:
   e_vu = LeakyReLU(a^T * CONCAT(W*h_v, W*h_u))

2. Normalize via softmax:
   alpha_vu = softmax_u(e_vu) = exp(e_vu) / SUM_{k in N(v)} exp(e_vk)

3. Aggregate with attention weights:
   h_v' = sigma(SUM_{u in N(v)} alpha_vu * W * h_u)
```

### Multi-Head Attention

Like Transformers, GAT uses multiple attention heads and concatenates (or averages) their outputs:

```
h_v' = CONCAT_{k=1}^{K} sigma(SUM_{u in N(v)} alpha_vu^k * W^k * h_u)
```

### Architecture Diagram (Single Node)

```
                    Attention
Neighbors           Weights        Weighted Sum
+---------+
| h_u1    | -----> alpha_1 = 0.4 -+
+---------+                        |
| h_u2    | -----> alpha_2 = 0.1 -+---> SUM(alpha_i * W*h_ui) ---> h_v'
+---------+                        |
| h_u3    | -----> alpha_3 = 0.3 -+
+---------+                        |
| h_u4    | -----> alpha_4 = 0.2 -+
+---------+

Multi-Head (K=3):
  Head 1: h_v'^1 --+
  Head 2: h_v'^2 --+--> CONCAT --> h_v_final
  Head 3: h_v'^3 --+
```

### When to Use GAT

- When some neighbors are more informative than others (common in KGs)
- When you want interpretable aggregation (attention weights show which relations matter)
- Higher computational cost than GCN but often better performance

---

## R-GCN: Relational Graph Convolutional Networks

### Why Standard GCNs Fail on KGs

Standard GCN treats all edges the same. In a KG, edges have types (relations), and different relations carry different semantics. R-GCN (Schlichtkrull et al., 2018) extends GCN to multi-relational graphs.

### Formula

```
h_v^(l+1) = sigma(SUM_{r in R} SUM_{u in N_r(v)} (1/c_{v,r}) * W_r^(l) * h_u^(l) + W_0^(l) * h_v^(l))
```

where:
- `R` is the set of relation types
- `N_r(v)` is the set of neighbors connected to v via relation r
- `W_r^(l)` is a relation-specific weight matrix
- `c_{v,r}` is a normalization constant (e.g., `|N_r(v)|`)
- `W_0^(l)` is the self-loop weight matrix

### Parameter Explosion Problem

With many relation types, each needing its own `W_r`, parameters explode. Two solutions:

**Basis Decomposition**:
```
W_r = SUM_{b=1}^{B} a_{rb} * V_b
```
Share B basis matrices across all relations; each relation is a weighted combination.

**Block Diagonal Decomposition**:
```
W_r = diag(W_r^1, W_r^2, ..., W_r^B)
```
Each relation matrix is block-diagonal, reducing parameters.

### Architecture for KG Tasks

```
Entity Features     R-GCN Layer 1      R-GCN Layer 2      Task Head
                   (per-relation       (per-relation
                    aggregation)        aggregation)

  [x_A] ---+                    +---+
  "works_at"|  W_works_at *     |   |
  [x_B] ---+---> h_A^1 --------+   +---> h_A^2 ---> [Node Classification]
  "knows"  |                    |   |                  or
  [x_C] ---+  W_knows *        |   |                 [Link Prediction]
             aggregate          +---+
                                  ^
                         Each relation type
                         has its own W_r
```

---

## When GNNs Beat Embedding Methods

| Scenario | KG Embeddings | GNNs | Winner |
|----------|--------------|------|--------|
| Transductive link prediction | Strong | Good | Embeddings |
| Inductive (new entities at test time) | Cannot handle | Strong | GNNs |
| Rich node features available | Ignores them | Uses them | GNNs |
| Node classification | Not designed for this | Strong | GNNs |
| Graph classification | Not applicable | Strong | GNNs |
| Very large KG (>10M entities) | Scales well | Memory issues | Embeddings |
| Few relation types | Overkill | Good | GNNs |
| Many relation types (>1000) | Good | R-GCN struggles | Embeddings |
| Temporal/dynamic KGs | Requires retraining | Mini-batch update | GNNs |

### Rule of Thumb

- **Use KG Embeddings** when: your task is link prediction, your KG is large, you have many relation types, and all entities are known at training time
- **Use GNNs** when: you need inductive predictions, you have node features, you need node/graph classification, or your graph evolves frequently

---

## Summary of GNN Architectures

```
+------------------+------------------+------------------+------------------+
|                  |      GCN         |    GraphSAGE     |       GAT        |
+------------------+------------------+------------------+------------------+
| Aggregation      | Weighted sum     | Sample + pool    | Attention-       |
|                  | (degree-norm)    | (mean/max/LSTM)  | weighted sum     |
+------------------+------------------+------------------+------------------+
| Inductive?       | No (full graph)  | Yes (sampling)   | Yes (attention   |
|                  |                  |                  | is param-based)  |
+------------------+------------------+------------------+------------------+
| Scalability      | Full batch       | Mini-batch       | Full batch       |
|                  | (memory-bound)   | (scalable)       | (memory-bound)   |
+------------------+------------------+------------------+------------------+
| Multi-relational | No (use R-GCN)   | Not standard     | Not standard     |
|                  |                  | (can extend)     | (can extend)     |
+------------------+------------------+------------------+------------------+
| Interpretability | Low              | Low              | High (attention  |
|                  |                  |                  | weights)         |
+------------------+------------------+------------------+------------------+
```

---

## Key Takeaways

1. **Message passing is the core paradigm** -- all GNNs aggregate neighbor information iteratively
2. **2-3 layers suffice** for most tasks; more layers cause over-smoothing
3. **GCN is the baseline** -- start here, then try GAT or GraphSAGE if you need attention or scalability
4. **R-GCN is essential for KGs** -- standard GNNs ignore relation types
5. **GNNs complement KG embeddings** -- they excel in different scenarios, and hybrid approaches often work best

---

## References

- Kipf, T. & Welling, M. (2017). "Semi-Supervised Classification with Graph Convolutional Networks." ICLR.
- Hamilton, W., et al. (2017). "Inductive Representation Learning on Large Graphs." NeurIPS.
- Velickovic, P., et al. (2018). "Graph Attention Networks." ICLR.
- Schlichtkrull, M., et al. (2018). "Modeling Relational Data with Graph Convolutional Networks." ESWC.
- Wu, Z., et al. (2020). "A Comprehensive Survey on Graph Neural Networks." IEEE TNNLS.
