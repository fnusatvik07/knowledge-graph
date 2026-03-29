# Graph Embeddings for Knowledge Graphs

Graph embeddings map the discrete structure of a knowledge graph -- its nodes, edges, and relationships -- into continuous vector spaces. Once entities and relations live as vectors, you can apply the same mathematical operations used in NLP and computer vision: measuring similarity, clustering, and feeding them into downstream models.

## What Are Graph Embeddings?

A graph embedding is a learned function that maps each node (and optionally each edge type) to a dense, low-dimensional vector:

```
f: Node → ℝ^d
g: Relation → ℝ^d
```

The key constraint: embeddings must **preserve graph structure**. Nodes connected by edges should be closer in the vector space than unconnected nodes, and specific relationship patterns should correspond to geometric operations.

```
# Conceptual example
embedding("Albert Einstein") ≈ embedding("Max Planck")   # both physicists
embedding("Einstein") + embedding("developed") ≈ embedding("Relativity")
```

## Why Graph Embeddings Matter for KGs

| Challenge | How Embeddings Help |
|-----------|-------------------|
| KG completion | Predict missing links by checking if head + relation ≈ tail |
| Entity resolution | Detect duplicate entities by comparing vectors |
| Retrieval | Find relevant subgraphs via nearest-neighbor search |
| Clustering | Group related entities without manual categorization |
| Transfer learning | Pre-trained KG embeddings boost downstream tasks |

## Major Embedding Methods

### Translation-Based Models

**TransE** (2013) -- the foundational method. Models each relation as a translation in embedding space:

```
head + relation ≈ tail
```

For a valid triple (Einstein, developed, Relativity), the vectors satisfy:
```
v_Einstein + v_developed ≈ v_Relativity
```

TransE is fast and simple but struggles with one-to-many relations (one person develops multiple theories).

**RotatE** (2019) -- models relations as rotations in complex vector space. Each relation is a rotation angle, and:

```
head ∘ relation ≈ tail   (element-wise complex multiplication)
```

RotatE handles symmetry, antisymmetry, inversion, and composition patterns that TransE cannot.

### Random Walk Methods

**node2vec** generates embeddings by simulating biased random walks on the graph and feeding the resulting "sentences" of nodes into a Word2Vec-style model.

Two parameters control the walk behavior:
- **p** (return parameter): likelihood of revisiting the previous node (low p = depth-first / structural)
- **q** (in-out parameter): likelihood of exploring outward (low q = breadth-first / community-aware)

### Graph Neural Networks (GNNs)

GNNs learn embeddings by **message passing**: each node aggregates information from its neighbors over multiple layers.

```
# Conceptual GNN layer
h_v^(k+1) = UPDATE(h_v^(k), AGGREGATE({h_u^(k) : u ∈ neighbors(v)}))
```

Key architectures:
- **GCN** (Graph Convolutional Network): weighted mean aggregation
- **GraphSAGE**: sampling + aggregation for large graphs
- **GAT** (Graph Attention Network): attention-weighted neighbor aggregation
- **R-GCN** (Relational GCN): separate weight matrices per relation type -- designed for KGs

## Code Example: node2vec with NetworkX

```python
import networkx as nx
from node2vec import Node2Vec
import numpy as np

# Build a small knowledge graph
G = nx.DiGraph()

# Add entities and relationships
edges = [
    ("Einstein", "Relativity"),
    ("Einstein", "Photoelectric_Effect"),
    ("Einstein", "Princeton"),
    ("Bohr", "Atomic_Model"),
    ("Bohr", "Copenhagen"),
    ("Relativity", "Physics"),
    ("Atomic_Model", "Physics"),
    ("Photoelectric_Effect", "Physics"),
    ("Princeton", "New_Jersey"),
    ("Copenhagen", "Denmark"),
]
G.add_edges_from(edges)

# Generate embeddings with node2vec
node2vec = Node2Vec(
    G,
    dimensions=64,      # embedding size
    walk_length=20,      # nodes per random walk
    num_walks=100,       # walks per node
    p=1.0,               # return parameter
    q=2.0,               # in-out parameter (BFS-like)
    workers=4,
)

# Train the model (uses gensim Word2Vec under the hood)
model = node2vec.fit(
    window=5,
    min_count=1,
    batch_words=4,
)

# Find most similar entities to Einstein
similar = model.wv.most_similar("Einstein", topn=5)
print("Most similar to Einstein:")
for node, score in similar:
    print(f"  {node}: {score:.3f}")

# Check similarity between two physicists
sim = model.wv.similarity("Einstein", "Bohr")
print(f"\nEinstein-Bohr similarity: {sim:.3f}")

# Get the raw embedding vector
einstein_vec = model.wv["Einstein"]
print(f"\nEinstein embedding shape: {einstein_vec.shape}")
print(f"First 5 dimensions: {einstein_vec[:5]}")
```

### Using Embeddings for Link Prediction

```python
from itertools import product

def predict_missing_links(model, G, threshold=0.7):
    """Predict missing edges based on embedding similarity."""
    nodes = list(G.nodes())
    predictions = []

    for u, v in product(nodes, repeat=2):
        if u != v and not G.has_edge(u, v):
            sim = model.wv.similarity(u, v)
            if sim > threshold:
                predictions.append((u, v, sim))

    predictions.sort(key=lambda x: x[2], reverse=True)
    return predictions

missing_links = predict_missing_links(model, G, threshold=0.5)
print("Predicted missing links:")
for src, dst, score in missing_links[:10]:
    print(f"  ({src}) --> ({dst})  score={score:.3f}")
```

## Relationship to Text Embeddings

Text embeddings (from models like `text-embedding-3-small`) and graph embeddings serve different purposes:

| Aspect | Text Embeddings | Graph Embeddings |
|--------|----------------|-----------------|
| Input | Raw text / sentences | Graph structure |
| Captures | Semantic meaning | Structural relationships |
| Similarity | "Means the same thing" | "Connected in the graph" |
| Weakness | Misses structural context | Misses semantic nuance |

In practice, the most effective systems use **both**: text embeddings for semantic similarity and graph embeddings for structural awareness.

## Cross-Modal Embeddings

Cross-modal embeddings unify text and graph representations into a **shared vector space**, enabling queries like "find the graph entity most relevant to this sentence."

Architecture pattern:
```
Text Encoder (e.g., BERT)    →  shared space  ←  Graph Encoder (e.g., R-GCN)
"physicist who developed       ℝ^d               node: Einstein
 general relativity"                              + neighbors + relations
```

Approaches:
1. **Joint training**: train text and graph encoders simultaneously with a contrastive loss
2. **Alignment**: learn a projection from one space to the other using anchor pairs
3. **Hybrid models**: encode node descriptions with a text model, then refine with graph structure via GNN layers

Cross-modal embeddings power modern Graph RAG systems where a natural-language query must retrieve structured graph context.

## KG Completion with Embeddings

KG completion -- predicting missing triples -- is the flagship application of graph embeddings.

```
Known:   (Einstein, born_in, Ulm), (Ulm, located_in, Germany)
Missing: (Einstein, nationality, ?) → predict "German"
```

Pipeline:
1. Train embeddings on the existing KG
2. For a candidate triple (h, r, ?), compute `h + r` and find nearest entity
3. Rank candidates by distance; apply a threshold or top-k filter
4. Optionally validate with an LLM before inserting into the KG

## Practical Considerations

- **Dimensionality**: 64-256 dimensions is typical; higher is not always better
- **Training data**: embeddings are only as good as the graph -- noisy or incomplete KGs produce poor embeddings
- **Scalability**: TransE and node2vec scale to millions of nodes; GNNs require more compute but capture richer patterns
- **Dynamic graphs**: if the KG changes frequently, embeddings need periodic retraining or incremental update strategies
- **Evaluation**: use link prediction benchmarks (MRR, Hits@10) on held-out triples

## Key Takeaways

- Graph embeddings map KG structure into vector spaces, enabling similarity search, link prediction, and clustering
- TransE and RotatE model relations as geometric operations on entity vectors
- node2vec uses random walks to learn structure-aware embeddings and integrates easily with NetworkX
- GNNs learn embeddings through neighborhood aggregation and are the most expressive approach
- Cross-modal embeddings bridge text and graph representations for Graph RAG retrieval
- In production, combining text embeddings with graph embeddings yields the best retrieval quality
