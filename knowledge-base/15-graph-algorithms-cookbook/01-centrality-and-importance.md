# Centrality and Importance Algorithms for Knowledge Graphs

## Overview

Centrality algorithms identify the most important or influential nodes in a graph. In a knowledge graph, this translates to finding the most significant entities -- the people, concepts, or organizations that play central roles in the knowledge structure.

This guide covers five major centrality metrics with NetworkX code examples applied to knowledge graphs.

---

## Setup: Building a Sample KG

```python
import networkx as nx
import matplotlib.pyplot as plt

# Build a small knowledge graph as a NetworkX graph
G = nx.DiGraph()

# Entities and relationships
edges = [
    ("Albert Einstein", "University of Zurich", "educated_at"),
    ("Albert Einstein", "Physics", "field"),
    ("Albert Einstein", "Nobel Prize", "award"),
    ("Albert Einstein", "Max Planck", "collaborated_with"),
    ("Albert Einstein", "Relativity", "discovered"),
    ("Max Planck", "Physics", "field"),
    ("Max Planck", "Nobel Prize", "award"),
    ("Max Planck", "Quantum Mechanics", "contributed_to"),
    ("Niels Bohr", "Physics", "field"),
    ("Niels Bohr", "Nobel Prize", "award"),
    ("Niels Bohr", "Quantum Mechanics", "contributed_to"),
    ("Niels Bohr", "Max Planck", "influenced_by"),
    ("Werner Heisenberg", "Physics", "field"),
    ("Werner Heisenberg", "Nobel Prize", "award"),
    ("Werner Heisenberg", "Quantum Mechanics", "contributed_to"),
    ("Werner Heisenberg", "Niels Bohr", "student_of"),
    ("Physics", "Natural Science", "subfield_of"),
    ("Quantum Mechanics", "Physics", "subfield_of"),
    ("Relativity", "Physics", "subfield_of"),
    ("University of Zurich", "Switzerland", "located_in"),
]

for src, tgt, rel in edges:
    G.add_edge(src, tgt, relation=rel)

# For undirected centrality metrics
G_undirected = G.to_undirected()

print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
```

---

## Degree Centrality

### What It Measures

Degree centrality counts the number of direct connections a node has. In a KG, entities with high degree centrality participate in many relationships -- they are the "hubs."

### Formula

```
C_D(v) = deg(v) / (n - 1)
```

where `deg(v)` is the number of edges incident to v and n is the total number of nodes.

For directed graphs, we distinguish in-degree (incoming edges) and out-degree (outgoing edges).

### Code

```python
# Undirected degree centrality
degree_cent = nx.degree_centrality(G_undirected)
sorted_degree = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)

print("=== Degree Centrality (Top 10) ===")
for node, score in sorted_degree[:10]:
    print(f"  {node:30s}  {score:.4f}  (degree={G_undirected.degree(node)})")

# Directed: in-degree vs out-degree
in_degree = nx.in_degree_centrality(G)
out_degree = nx.out_degree_centrality(G)

print("\n=== In-Degree Centrality (most pointed to) ===")
for node, score in sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {node:30s}  {score:.4f}")

print("\n=== Out-Degree Centrality (most connections from) ===")
for node, score in sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {node:30s}  {score:.4f}")
```

### Interpretation in KGs

- **High in-degree**: Entities that many others reference (e.g., "Physics", "Nobel Prize") -- these are common targets or categories
- **High out-degree**: Entities with many described relationships (e.g., "Albert Einstein") -- these are well-documented entities

---

## Betweenness Centrality

### What It Measures

Betweenness centrality identifies nodes that serve as bridges between different parts of the graph. A node has high betweenness if many shortest paths between other nodes pass through it.

### Formula

```
C_B(v) = SUM_{s != v != t} (sigma_st(v) / sigma_st)
```

where `sigma_st` is the total number of shortest paths from s to t, and `sigma_st(v)` is the number of those paths passing through v.

### Code

```python
betweenness = nx.betweenness_centrality(G_undirected, normalized=True)
sorted_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)

print("=== Betweenness Centrality ===")
for node, score in sorted_betweenness[:10]:
    print(f"  {node:30s}  {score:.4f}")
```

### Interpretation in KGs

- **High betweenness**: Entities that connect different domains or communities
- Example: "Physics" might have high betweenness because it connects scientists to concepts
- Useful for finding **bridge concepts** that link otherwise separate areas of knowledge

### Use Case: Finding Bridge Entities

```python
# Find entities that bridge different communities
def find_bridges(G, top_k=5):
    betweenness = nx.betweenness_centrality(G, normalized=True)
    bridges = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:top_k]
    for node, score in bridges:
        neighbors = list(G.neighbors(node))
        print(f"  Bridge: {node} (score={score:.4f})")
        print(f"    Connects: {', '.join(neighbors[:5])}")
    return bridges

find_bridges(G_undirected)
```

---

## Closeness Centrality

### What It Measures

Closeness centrality measures how quickly a node can reach all other nodes. Nodes with high closeness are "close" to the rest of the network.

### Formula

```
C_C(v) = (n - 1) / SUM_{u != v} d(v, u)
```

where `d(v, u)` is the shortest path length between v and u.

### Code

```python
closeness = nx.closeness_centrality(G_undirected)
sorted_closeness = sorted(closeness.items(), key=lambda x: x[1], reverse=True)

print("=== Closeness Centrality ===")
for node, score in sorted_closeness[:10]:
    print(f"  {node:30s}  {score:.4f}")
```

### Interpretation in KGs

- **High closeness**: Entities that are few hops from everything else -- central concepts
- Useful for finding the most "accessible" entities for navigation or exploration
- Low closeness entities are peripheral or isolated

---

## PageRank

### What It Measures

PageRank (Page et al., 1998) computes recursive importance: a node is important if important nodes point to it. Originally designed for ranking web pages, it works beautifully on KGs.

### Formula

```
PR(v) = (1 - d) / n + d * SUM_{u in In(v)} PR(u) / Out(u)
```

where:
- `d` is the damping factor (typically 0.85)
- `In(v)` is the set of nodes pointing to v
- `Out(u)` is the out-degree of u
- `n` is the total number of nodes

### Code

```python
pagerank = nx.pagerank(G, alpha=0.85)
sorted_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)

print("=== PageRank ===")
for node, score in sorted_pagerank[:10]:
    print(f"  {node:30s}  {score:.6f}")
```

### Tuning the Damping Factor

```python
# Compare different damping factors
for alpha in [0.5, 0.7, 0.85, 0.95]:
    pr = nx.pagerank(G, alpha=alpha)
    top = sorted(pr.items(), key=lambda x: x[1], reverse=True)[0]
    print(f"  alpha={alpha}: top entity = {top[0]} ({top[1]:.6f})")
```

### Personalized PageRank

Focus the ranking around specific entities:

```python
# Personalized PageRank: biased toward "Albert Einstein"
personalization = {node: 0 for node in G.nodes()}
personalization["Albert Einstein"] = 1.0

ppr = nx.pagerank(G, alpha=0.85, personalization=personalization)
sorted_ppr = sorted(ppr.items(), key=lambda x: x[1], reverse=True)

print("=== Personalized PageRank (biased toward Einstein) ===")
for node, score in sorted_ppr[:10]:
    print(f"  {node:30s}  {score:.6f}")
```

### Interpretation in KGs

- **High PageRank**: Entities pointed to by other important entities
- More nuanced than degree centrality: an entity with few connections can rank high if those connections come from important entities
- **Personalized PageRank** is excellent for finding entities relevant to a specific context

---

## Eigenvector Centrality

### What It Measures

Eigenvector centrality is the theoretical foundation of PageRank. A node is important if its neighbors are important (recursive definition). The centrality scores are the components of the dominant eigenvector of the adjacency matrix.

### Formula

```
C_E(v) = (1 / lambda) * SUM_{u in N(v)} C_E(u)
```

where lambda is the largest eigenvalue of the adjacency matrix.

### Code

```python
try:
    eigenvector = nx.eigenvector_centrality(G_undirected, max_iter=1000)
    sorted_eigen = sorted(eigenvector.items(), key=lambda x: x[1], reverse=True)

    print("=== Eigenvector Centrality ===")
    for node, score in sorted_eigen[:10]:
        print(f"  {node:30s}  {score:.6f}")
except nx.PowerIterationFailedConvergence:
    print("Eigenvector centrality did not converge -- try increasing max_iter")
```

### Eigenvector vs PageRank

- Eigenvector centrality works on undirected graphs; PageRank is designed for directed graphs
- PageRank adds the damping factor to handle dangling nodes and guarantee convergence
- For KGs, PageRank is usually preferred (directed relationships are meaningful)

---

## Comparison: When to Use Which Metric

| Metric | Question It Answers | Best For |
|--------|-------------------|----------|
| **Degree** | Who has the most connections? | Finding hubs, data quality audit |
| **Betweenness** | Who bridges communities? | Finding connectors, bottleneck analysis |
| **Closeness** | Who can reach everyone fastest? | Finding central concepts, navigation |
| **PageRank** | Who is recursively important? | Entity ranking, search results |
| **Eigenvector** | Who is connected to important nodes? | Influence analysis |

### Complete Comparison Code

```python
import pandas as pd

metrics = {
    "Degree": nx.degree_centrality(G_undirected),
    "Betweenness": nx.betweenness_centrality(G_undirected),
    "Closeness": nx.closeness_centrality(G_undirected),
    "PageRank": nx.pagerank(G),
}

try:
    metrics["Eigenvector"] = nx.eigenvector_centrality(G_undirected, max_iter=1000)
except:
    pass

df = pd.DataFrame(metrics)
df = df.round(4)

# Rank each metric
for col in df.columns:
    df[f"{col}_rank"] = df[col].rank(ascending=False).astype(int)

print(df.sort_values("PageRank", ascending=False).head(10))
```

---

## Use Case: Finding the Most Important Entities

Combine multiple centrality measures for a robust importance score:

```python
def composite_importance(G, weights=None):
    """Compute a weighted composite importance score."""
    if weights is None:
        weights = {"degree": 0.2, "betweenness": 0.2, "closeness": 0.2, "pagerank": 0.4}

    G_u = G.to_undirected() if G.is_directed() else G

    scores = {
        "degree": nx.degree_centrality(G_u),
        "betweenness": nx.betweenness_centrality(G_u),
        "closeness": nx.closeness_centrality(G_u),
        "pagerank": nx.pagerank(G),
    }

    # Normalize each metric to [0, 1]
    for metric in scores:
        max_val = max(scores[metric].values()) or 1
        scores[metric] = {k: v / max_val for k, v in scores[metric].items()}

    # Weighted combination
    composite = {}
    for node in G.nodes():
        composite[node] = sum(
            weights.get(metric, 0) * scores[metric].get(node, 0)
            for metric in scores
        )

    return sorted(composite.items(), key=lambda x: x[1], reverse=True)

print("=== Composite Importance ===")
for node, score in composite_importance(G)[:10]:
    print(f"  {node:30s}  {score:.4f}")
```

---

## Visualization

```python
def visualize_centrality(G, centrality_dict, title="Centrality"):
    """Visualize a graph with node sizes proportional to centrality."""
    pos = nx.spring_layout(G, seed=42, k=2)

    # Node sizes proportional to centrality
    max_cent = max(centrality_dict.values()) or 1
    node_sizes = [3000 * centrality_dict.get(n, 0) / max_cent + 100 for n in G.nodes()]

    plt.figure(figsize=(14, 10))
    nx.draw(G, pos,
            node_size=node_sizes,
            node_color=list(centrality_dict.values()),
            cmap=plt.cm.YlOrRd,
            with_labels=True,
            font_size=8,
            edge_color="gray",
            alpha=0.8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png", dpi=150)
    plt.show()

visualize_centrality(G_undirected, pagerank, "PageRank Centrality")
```

---

## Key Takeaways

1. **Degree centrality is the simplest** -- start here for a quick overview of hub entities
2. **PageRank is the most useful** for KGs -- it captures recursive importance in directed graphs
3. **Betweenness reveals bridge entities** that connect different knowledge domains
4. **Personalized PageRank** is powerful for context-specific entity ranking (e.g., in RAG)
5. **Combine multiple metrics** for a robust, well-rounded importance score
6. **All metrics are available in NetworkX** with simple one-line function calls

---

## References

- Page, L., et al. (1998). "The PageRank Citation Ranking: Bringing Order to the Web." Stanford Technical Report.
- Freeman, L. (1978). "Centrality in Social Networks: Conceptual Clarification." Social Networks.
- NetworkX documentation: https://networkx.org/documentation/stable/reference/algorithms/centrality.html
