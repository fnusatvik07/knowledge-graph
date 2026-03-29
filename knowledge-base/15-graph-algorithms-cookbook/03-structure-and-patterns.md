# Structural Analysis and Pattern Detection in Knowledge Graphs

## Overview

Understanding the structural properties of a knowledge graph reveals important insights: Which parts of the graph are disconnected? Where are the dense clusters? What recurring patterns exist? This guide covers structural analysis algorithms with practical NetworkX code examples.

---

## Setup

```python
import networkx as nx
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# Build a knowledge graph with community structure
G = nx.Graph()

# Community 1: Machine Learning researchers
ml_edges = [
    ("Hinton", "LeCun"), ("Hinton", "Bengio"),
    ("LeCun", "Bengio"), ("LeCun", "Goodfellow"),
    ("Bengio", "Goodfellow"), ("Hinton", "Goodfellow"),
    ("Hinton", "Deep Learning"), ("LeCun", "CNN"),
    ("Bengio", "NLP"), ("Goodfellow", "GAN"),
]

# Community 2: Graph researchers
graph_edges = [
    ("Kipf", "Welling"), ("Kipf", "GCN"),
    ("Welling", "VAE"), ("Kipf", "Hamilton"),
    ("Hamilton", "GraphSAGE"), ("Hamilton", "Welling"),
    ("Velickovic", "GAT"), ("Velickovic", "Kipf"),
    ("Velickovic", "Hamilton"),
]

# Community 3: NLP researchers
nlp_edges = [
    ("Vaswani", "Transformer"), ("Vaswani", "Shazeer"),
    ("Shazeer", "Transformer"), ("Devlin", "BERT"),
    ("Devlin", "Vaswani"), ("Radford", "GPT"),
    ("Radford", "Transformer"), ("Devlin", "Radford"),
]

# Bridge edges connecting communities
bridge_edges = [
    ("Bengio", "Kipf"),       # ML <-> Graph
    ("Bengio", "Devlin"),     # ML <-> NLP
    ("Hamilton", "Vaswani"),  # Graph <-> NLP
]

G.add_edges_from(ml_edges)
G.add_edges_from(graph_edges)
G.add_edges_from(nlp_edges)
G.add_edges_from(bridge_edges)

print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
```

---

## Connected Components

### Finding Isolated Subgraphs

Connected components identify groups of nodes that can reach each other but are disconnected from other groups.

```python
# For undirected graphs
components = list(nx.connected_components(G))
print(f"Number of connected components: {len(components)}")
for i, comp in enumerate(components):
    print(f"  Component {i+1} ({len(comp)} nodes): {comp}")

# Largest connected component
largest_cc = max(nx.connected_components(G), key=len)
G_largest = G.subgraph(largest_cc).copy()
print(f"Largest component: {len(largest_cc)} nodes")
```

### Directed Graphs: Strongly vs Weakly Connected

```python
G_directed = nx.DiGraph()
G_directed.add_edges_from([
    ("A", "B"), ("B", "C"), ("C", "A"),  # strongly connected cycle
    ("C", "D"), ("D", "E"),               # one-way path
    ("F", "G"),                           # separate component
])

# Weakly connected: ignoring edge direction
weak = list(nx.weakly_connected_components(G_directed))
print(f"Weakly connected components: {len(weak)}")

# Strongly connected: respecting edge direction
strong = list(nx.strongly_connected_components(G_directed))
print(f"Strongly connected components: {len(strong)}")
for i, comp in enumerate(strong):
    print(f"  SCC {i+1}: {comp}")
```

### Use Case: KG Quality Audit

```python
def audit_connectivity(G):
    """Check if KG has isolated subgraphs (potential data quality issue)."""
    G_undirected = G.to_undirected() if G.is_directed() else G
    components = list(nx.connected_components(G_undirected))

    if len(components) == 1:
        print("Graph is fully connected.")
    else:
        print(f"WARNING: {len(components)} disconnected components found!")
        for i, comp in enumerate(sorted(components, key=len, reverse=True)):
            print(f"  Component {i+1}: {len(comp)} nodes")
            if len(comp) <= 5:
                print(f"    Nodes: {comp}")

audit_connectivity(G)
```

---

## K-Core Decomposition

### What It Is

A k-core is the maximal subgraph where every node has degree >= k. K-core decomposition reveals the dense "core" of the graph versus the sparse "periphery."

```python
# Core number for each node
core_numbers = nx.core_number(G)
print("=== Core Numbers ===")
for node, k in sorted(core_numbers.items(), key=lambda x: x[1], reverse=True):
    print(f"  {node:20s}  core={k}")

# Extract the 3-core (dense core)
k = 3
k_core = nx.k_core(G, k=k)
print(f"\n{k}-core: {list(k_core.nodes())}")

# Extract the periphery (shell)
k_shell = nx.k_shell(G, k=1)
print(f"1-shell (periphery): {list(k_shell.nodes())}")
```

### Hierarchical Core Structure

```python
# All cores from 1 to max
max_core = max(core_numbers.values())
print(f"Maximum core number: {max_core}")

for k in range(1, max_core + 1):
    k_core_nodes = nx.k_core(G, k=k).nodes()
    print(f"  {k}-core: {len(k_core_nodes)} nodes - {set(k_core_nodes)}")
```

### Use Case: Finding the Knowledge Core

```python
def find_knowledge_core(G, min_k=3):
    """Find the densely connected core entities of a KG."""
    core_numbers = nx.core_number(G)
    core_entities = {n for n, k in core_numbers.items() if k >= min_k}

    print(f"Knowledge core ({min_k}-core): {len(core_entities)} entities")
    for entity in sorted(core_entities):
        print(f"  - {entity} (core={core_numbers[entity]})")
    return core_entities

core = find_knowledge_core(G, min_k=3)
```

---

## Clique Detection

### What It Is

A clique is a subset of nodes where every pair is connected (a fully connected subgraph). Cliques represent tightly coupled groups of entities.

```python
# Find all maximal cliques
cliques = list(nx.find_cliques(G))
print(f"Number of maximal cliques: {len(cliques)}")

for i, clique in enumerate(sorted(cliques, key=len, reverse=True)):
    print(f"  Clique {i+1} (size={len(clique)}): {clique}")

# Largest clique
max_clique = max(cliques, key=len)
print(f"\nLargest clique: {max_clique}")

# Clique number (size of largest clique)
clique_number = nx.graph_clique_number(G)
print(f"Clique number: {clique_number}")
```

### Cliques Containing a Specific Node

```python
# All cliques containing "Hinton"
hinton_cliques = [c for c in nx.find_cliques(G) if "Hinton" in c]
print(f"Cliques containing Hinton: {len(hinton_cliques)}")
for clique in hinton_cliques:
    print(f"  {clique}")
```

### Use Case: Finding Tightly Coupled Entity Groups

```python
def find_entity_clusters(G, min_size=3):
    """Find groups of entities that are all mutually connected."""
    cliques = [c for c in nx.find_cliques(G) if len(c) >= min_size]
    cliques.sort(key=len, reverse=True)

    print(f"Found {len(cliques)} clusters of size >= {min_size}:")
    for i, clique in enumerate(cliques):
        print(f"  Cluster {i+1}: {clique}")
        # Check what they have in common
        common_neighbors = set(G.neighbors(clique[0]))
        for node in clique[1:]:
            common_neighbors &= set(G.neighbors(node))
        if common_neighbors - set(clique):
            print(f"    Shared connections: {common_neighbors - set(clique)}")

find_entity_clusters(G, min_size=3)
```

---

## Motif Detection

### What It Is

Motifs are recurring subgraph patterns (e.g., triangles, stars, chains). They reveal the structural building blocks of a KG.

### Triangle Counting

```python
# Number of triangles per node
triangles = nx.triangles(G)
print("=== Triangles per Node ===")
for node, count in sorted(triangles.items(), key=lambda x: x[1], reverse=True):
    if count > 0:
        print(f"  {node:20s}  {count} triangles")

# Total triangles in the graph
total_triangles = sum(triangles.values()) // 3  # each triangle counted 3 times
print(f"\nTotal triangles: {total_triangles}")

# Transitivity (ratio of triangles to possible triangles)
transitivity = nx.transitivity(G)
print(f"Transitivity: {transitivity:.4f}")
```

### Clustering Coefficient

The clustering coefficient measures how clustered a node's neighborhood is:

```python
# Per-node clustering coefficient
clustering = nx.clustering(G)
print("=== Clustering Coefficient ===")
for node, cc in sorted(clustering.items(), key=lambda x: x[1], reverse=True):
    print(f"  {node:20s}  {cc:.4f}")

# Average clustering coefficient
avg_clustering = nx.average_clustering(G)
print(f"\nAverage clustering: {avg_clustering:.4f}")
```

### Star Pattern Detection

```python
def find_star_patterns(G, min_leaves=3):
    """Find star patterns: a hub connected to many leaf nodes."""
    stars = []
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        # A leaf has degree 1
        leaves = [n for n in neighbors if G.degree(n) == 1]
        if len(leaves) >= min_leaves:
            stars.append((node, leaves))

    for hub, leaves in stars:
        print(f"  Star center: {hub}, leaves: {leaves}")
    return stars

print("=== Star Patterns ===")
find_star_patterns(G, min_leaves=1)
```

### Chain Pattern Detection

```python
def find_chains(G, min_length=3):
    """Find linear chains (paths where internal nodes have degree 2)."""
    chains = []
    visited = set()

    for node in G.nodes():
        if G.degree(node) == 1 and node not in visited:
            # Start of a potential chain
            chain = [node]
            current = node
            prev = None

            while True:
                neighbors = [n for n in G.neighbors(current) if n != prev]
                if len(neighbors) != 1 or G.degree(neighbors[0]) > 2:
                    if len(neighbors) == 1:
                        chain.append(neighbors[0])
                    break
                prev = current
                current = neighbors[0]
                chain.append(current)

            if len(chain) >= min_length:
                chains.append(chain)
                visited.update(chain)

    for chain in chains:
        print(f"  Chain: {' -> '.join(chain)}")
    return chains

print("=== Chain Patterns ===")
find_chains(G, min_length=2)
```

---

## Graph Density and Diameter

### Density

```python
density = nx.density(G)
print(f"Graph density: {density:.4f}")
# 0 = no edges, 1 = fully connected (complete graph)
# KGs typically have very low density (0.001 - 0.01)

# Interpret
n = G.number_of_nodes()
max_edges = n * (n - 1) / 2  # for undirected
actual_edges = G.number_of_edges()
print(f"Actual edges: {actual_edges} / {max_edges:.0f} possible ({density*100:.1f}%)")
```

### Diameter and Radius

```python
if nx.is_connected(G):
    diameter = nx.diameter(G)
    radius = nx.radius(G)
    center = nx.center(G)
    periphery = nx.periphery(G)

    print(f"Diameter: {diameter} (longest shortest path)")
    print(f"Radius: {radius} (minimum eccentricity)")
    print(f"Center nodes: {center}")
    print(f"Periphery nodes: {periphery}")

    # Eccentricity of each node
    eccentricity = nx.eccentricity(G)
    for node, ecc in sorted(eccentricity.items(), key=lambda x: x[1]):
        print(f"  {node:20s}  eccentricity={ecc}")
```

### Average Path Length

```python
if nx.is_connected(G):
    avg_path = nx.average_shortest_path_length(G)
    print(f"Average shortest path length: {avg_path:.3f}")
```

---

## Degree Distribution Analysis

### Power Law in KGs

Real-world KGs typically follow a power-law degree distribution: most entities have few connections, while a few entities (hubs) have many.

```python
# Degree distribution
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
degree_count = Counter(degree_sequence)

print("=== Degree Distribution ===")
for degree, count in sorted(degree_count.items()):
    bar = "#" * count
    print(f"  Degree {degree:2d}: {count:3d} nodes  {bar}")

# Basic statistics
degrees = [d for _, d in G.degree()]
print(f"\nMin degree:    {min(degrees)}")
print(f"Max degree:    {max(degrees)}")
print(f"Mean degree:   {np.mean(degrees):.2f}")
print(f"Median degree: {np.median(degrees):.1f}")
print(f"Std degree:    {np.std(degrees):.2f}")
```

### Plotting Degree Distribution

```python
def plot_degree_distribution(G, title="Degree Distribution"):
    """Plot degree distribution on log-log scale to check for power law."""
    degrees = [d for _, d in G.degree()]
    degree_count = Counter(degrees)

    x = sorted(degree_count.keys())
    y = [degree_count[k] for k in x]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Linear scale
    axes[0].bar(x, y, color="steelblue")
    axes[0].set_xlabel("Degree")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"{title} (Linear)")

    # Log-log scale (power law appears as straight line)
    axes[1].scatter(x, y, color="steelblue", s=50)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Degree (log)")
    axes[1].set_ylabel("Count (log)")
    axes[1].set_title(f"{title} (Log-Log)")

    plt.tight_layout()
    plt.savefig("degree_distribution.png", dpi=150)
    plt.show()

plot_degree_distribution(G)
```

### Power Law Fit

```python
def check_power_law(G):
    """Check if degree distribution follows a power law."""
    degrees = [d for _, d in G.degree() if d > 0]

    # Simple power law exponent estimation (maximum likelihood)
    d_min = min(degrees)
    n = len(degrees)
    alpha = 1 + n / sum(np.log(np.array(degrees) / (d_min - 0.5)))

    print(f"Estimated power law exponent (alpha): {alpha:.2f}")
    print(f"  alpha ~ 2-3 is typical for real-world networks")
    print(f"  alpha < 2: very heavy tail (super hubs)")
    print(f"  alpha > 3: closer to random graph")

    return alpha

check_power_law(G)
```

---

## Triadic Closure

### What It Is

Triadic closure is the principle that if A knows B and A knows C, then B and C are likely to know each other. This is used for **link prediction** in KGs.

```python
def predict_edges_triadic_closure(G, top_k=10):
    """Predict missing edges based on triadic closure (common neighbors)."""
    predictions = []

    nodes = list(G.nodes())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            if not G.has_edge(u, v):
                # Count common neighbors
                common = len(list(nx.common_neighbors(G, u, v)))
                if common > 0:
                    predictions.append((u, v, common))

    predictions.sort(key=lambda x: x[2], reverse=True)

    print(f"=== Predicted Edges (Triadic Closure) ===")
    for u, v, score in predictions[:top_k]:
        common_neighbors = list(nx.common_neighbors(G, u, v))
        print(f"  {u} <-> {v}  (common neighbors={score}: {common_neighbors})")

    return predictions[:top_k]

predict_edges_triadic_closure(G)
```

### Jaccard Coefficient

A normalized version of common neighbors:

```python
# Jaccard coefficient for all non-existing edges
preds = nx.jaccard_coefficient(G)
jaccard_scores = [(u, v, score) for u, v, score in preds if score > 0]
jaccard_scores.sort(key=lambda x: x[2], reverse=True)

print("=== Link Prediction (Jaccard Coefficient) ===")
for u, v, score in jaccard_scores[:10]:
    print(f"  {u} <-> {v}  jaccard={score:.4f}")
```

### Adamic-Adar Index

Weights common neighbors by inverse log of their degree (rarer common neighbors count more):

```python
preds = nx.adamic_adar_index(G)
aa_scores = [(u, v, score) for u, v, score in preds if score > 0]
aa_scores.sort(key=lambda x: x[2], reverse=True)

print("=== Link Prediction (Adamic-Adar) ===")
for u, v, score in aa_scores[:10]:
    print(f"  {u} <-> {v}  adamic_adar={score:.4f}")
```

---

## Community Detection

Finding groups of densely connected entities:

```python
# Louvain community detection
from networkx.algorithms.community import louvain_communities

communities = louvain_communities(G, seed=42)
print(f"Found {len(communities)} communities:")
for i, comm in enumerate(communities):
    print(f"  Community {i+1} ({len(comm)} members): {comm}")

# Modularity score
from networkx.algorithms.community import modularity
mod = modularity(G, communities)
print(f"Modularity: {mod:.4f}")
```

### Girvan-Newman (Edge Betweenness)

```python
from networkx.algorithms.community import girvan_newman

# Iteratively remove highest-betweenness edges
comp = girvan_newman(G)
# Get partition into k communities
k = 3
for communities in comp:
    if len(communities) >= k:
        print(f"\n{len(communities)} communities (Girvan-Newman):")
        for i, comm in enumerate(communities):
            print(f"  Community {i+1}: {sorted(comm)}")
        break
```

---

## Complete Structural Analysis Report

```python
def structural_report(G):
    """Generate a complete structural analysis of a knowledge graph."""
    G_u = G.to_undirected() if G.is_directed() else G

    print("=" * 60)
    print("KNOWLEDGE GRAPH STRUCTURAL ANALYSIS")
    print("=" * 60)

    # Basic stats
    print(f"\nNodes:     {G.number_of_nodes()}")
    print(f"Edges:     {G.number_of_edges()}")
    print(f"Density:   {nx.density(G_u):.4f}")

    # Connectivity
    components = list(nx.connected_components(G_u))
    print(f"Connected components: {len(components)}")

    if nx.is_connected(G_u):
        print(f"Diameter:  {nx.diameter(G_u)}")
        print(f"Radius:    {nx.radius(G_u)}")
        print(f"Avg path:  {nx.average_shortest_path_length(G_u):.3f}")

    # Degree stats
    degrees = [d for _, d in G_u.degree()]
    print(f"\nDegree - min: {min(degrees)}, max: {max(degrees)}, "
          f"mean: {np.mean(degrees):.1f}, median: {np.median(degrees):.0f}")

    # Clustering
    print(f"Avg clustering: {nx.average_clustering(G_u):.4f}")
    print(f"Transitivity:   {nx.transitivity(G_u):.4f}")
    print(f"Triangles:      {sum(nx.triangles(G_u).values()) // 3}")

    # Core decomposition
    core_numbers = nx.core_number(G_u)
    print(f"Max core number: {max(core_numbers.values())}")

    # Cliques
    cliques = list(nx.find_cliques(G_u))
    print(f"Maximal cliques: {len(cliques)}")
    print(f"Largest clique:  {max(len(c) for c in cliques)} nodes")

    print("=" * 60)

structural_report(G)
```

---

## Key Takeaways

1. **Connected components** reveal data quality issues -- disconnected subgraphs may indicate missing links
2. **K-core decomposition** separates core knowledge from peripheral facts
3. **Cliques** identify tightly coupled entity groups (e.g., research collaborators, product bundles)
4. **Triangles and clustering** measure local cohesion -- high clustering suggests redundant but reliable knowledge
5. **Degree distribution** following a power law is a hallmark of real-world KGs
6. **Triadic closure** is a simple but effective link prediction heuristic
7. **Community detection** groups related entities without manual categorization

---

## References

- Batagelj, V. & Zaversnik, M. (2003). "An O(m) Algorithm for Cores Decomposition of Networks."
- Watts, D. & Strogatz, S. (1998). "Collective Dynamics of Small-World Networks." Nature.
- Barabasi, A. & Albert, R. (1999). "Emergence of Scaling in Random Networks." Science.
- NetworkX documentation: https://networkx.org/documentation/stable/reference/algorithms/
- Blondel, V., et al. (2008). "Fast Unfolding of Communities in Large Networks." JSTAT.
