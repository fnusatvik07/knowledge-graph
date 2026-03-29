# Path and Traversal Algorithms for Knowledge Graphs

## Overview

Path and traversal algorithms are fundamental to knowledge graph reasoning. Finding how two entities are connected, discovering the shortest explanation path, or traversing the graph systematically are core operations for QA systems, recommendation engines, and explainable AI.

---

## Setup

```python
import networkx as nx
from collections import defaultdict

# Build a knowledge graph
G = nx.DiGraph()

edges = [
    ("Alice", "Bob", {"relation": "knows", "weight": 0.9}),
    ("Alice", "Carol", {"relation": "knows", "weight": 0.7}),
    ("Bob", "David", {"relation": "works_with", "weight": 0.8}),
    ("Bob", "Eve", {"relation": "knows", "weight": 0.6}),
    ("Carol", "David", {"relation": "knows", "weight": 0.5}),
    ("Carol", "Frank", {"relation": "manages", "weight": 0.9}),
    ("David", "Eve", {"relation": "knows", "weight": 0.7}),
    ("David", "Grace", {"relation": "works_with", "weight": 0.8}),
    ("Eve", "Grace", {"relation": "knows", "weight": 0.4}),
    ("Frank", "Grace", {"relation": "knows", "weight": 0.6}),
    ("Grace", "Heidi", {"relation": "manages", "weight": 0.9}),
    ("Eve", "Heidi", {"relation": "knows", "weight": 0.3}),
]

G.add_edges_from(edges)
print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
```

---

## Shortest Path

### Unweighted Shortest Path (BFS-based)

Finds the path with the fewest hops:

```python
# Shortest path between two entities
path = nx.shortest_path(G, source="Alice", target="Heidi")
print(f"Shortest path: {' -> '.join(path)}")
print(f"Length: {len(path) - 1} hops")

# Shortest path length only
length = nx.shortest_path_length(G, source="Alice", target="Heidi")
print(f"Shortest distance: {length}")
```

### All Shortest Paths from One Source

```python
# All shortest paths from Alice
all_paths = nx.shortest_path(G, source="Alice")
all_lengths = nx.shortest_path_length(G, source="Alice")

for target, path in all_paths.items():
    print(f"  Alice -> {target}: {' -> '.join(path)} (length={all_lengths[target]})")
```

### Dijkstra's Algorithm (Weighted)

When edges have weights (e.g., relationship confidence), find the path that minimizes total weight:

```python
# Shortest weighted path (minimizes sum of weights)
# For confidence scores, we want to MAXIMIZE, so use 1-weight as cost
for u, v, d in G.edges(data=True):
    d["cost"] = 1 - d["weight"]  # high confidence = low cost

path = nx.dijkstra_path(G, source="Alice", target="Heidi", weight="cost")
cost = nx.dijkstra_path_length(G, source="Alice", target="Heidi", weight="cost")

print(f"Optimal path: {' -> '.join(path)}")
print(f"Total cost: {cost:.3f}")

# Show edge weights along the path
for i in range(len(path) - 1):
    edge_data = G[path[i]][path[i+1]]
    print(f"  {path[i]} --[{edge_data['relation']}, conf={edge_data['weight']}]--> {path[i+1]}")
```

### A* Algorithm

A* uses a heuristic to speed up pathfinding. For KGs, a simple heuristic might estimate distance based on graph properties:

```python
# A* requires a heuristic function
# For demonstration, use a simple heuristic (always returns 0 = degenerates to Dijkstra)
def heuristic(u, v):
    return 0  # admissible heuristic

path = nx.astar_path(G, source="Alice", target="Heidi", heuristic=heuristic, weight="cost")
print(f"A* path: {' -> '.join(path)}")
```

---

## All Paths Between Two Entities

### All Simple Paths

Find every possible path (no repeated nodes):

```python
all_paths = list(nx.all_simple_paths(G, source="Alice", target="Heidi"))

print(f"Found {len(all_paths)} paths from Alice to Heidi:")
for i, path in enumerate(all_paths):
    print(f"  Path {i+1}: {' -> '.join(path)} (length={len(path)-1})")
```

### Paths with Cutoff

Limit maximum path length to avoid combinatorial explosion:

```python
# Only paths with at most 4 hops
short_paths = list(nx.all_simple_paths(G, source="Alice", target="Heidi", cutoff=4))

print(f"Paths with <= 4 hops: {len(short_paths)}")
for path in short_paths:
    print(f"  {' -> '.join(path)}")
```

### Paths with Relationship Details

```python
def path_with_relations(G, path):
    """Show a path with relationship labels."""
    segments = []
    for i in range(len(path) - 1):
        rel = G[path[i]][path[i+1]].get("relation", "?")
        segments.append(f"{path[i]} --[{rel}]--> {path[i+1]}")
    return "\n    ".join(segments)

for path in nx.all_simple_paths(G, "Alice", "Grace", cutoff=3):
    print(f"  Path:")
    print(f"    {path_with_relations(G, path)}")
    print()
```

---

## Breadth-First Traversal (BFS)

BFS explores the graph level by level. It is the basis for shortest path in unweighted graphs.

```python
# BFS from Alice
print("=== BFS from Alice ===")
bfs_edges = list(nx.bfs_edges(G, source="Alice"))
for u, v in bfs_edges:
    print(f"  {u} -> {v}")

# BFS tree
bfs_tree = nx.bfs_tree(G, source="Alice")
print(f"\nBFS tree has {bfs_tree.number_of_nodes()} nodes, {bfs_tree.number_of_edges()} edges")

# BFS layers (distance from source)
layers = dict(enumerate(nx.bfs_layers(G, "Alice")))
for depth, nodes in layers.items():
    print(f"  Depth {depth}: {nodes}")
```

### BFS with Depth Limit

```python
# Only explore up to 2 hops from Alice
bfs_limited = nx.bfs_tree(G, source="Alice", depth_limit=2)
print(f"2-hop neighborhood of Alice: {list(bfs_limited.nodes())}")
```

---

## Depth-First Traversal (DFS)

DFS explores as deep as possible before backtracking. Useful for finding long paths and detecting cycles.

```python
# DFS from Alice
print("=== DFS from Alice ===")
dfs_edges = list(nx.dfs_edges(G, source="Alice"))
for u, v in dfs_edges:
    print(f"  {u} -> {v}")

# DFS tree
dfs_tree = nx.dfs_tree(G, source="Alice")

# DFS with depth limit
dfs_limited = nx.dfs_tree(G, source="Alice", depth_limit=2)

# Pre-order and post-order
preorder = list(nx.dfs_preorder_nodes(G, source="Alice"))
postorder = list(nx.dfs_postorder_nodes(G, source="Alice"))
print(f"DFS pre-order:  {preorder}")
print(f"DFS post-order: {postorder}")
```

---

## Weighted Path Finding

### Using Relationship Confidence

In a KG, edges often have confidence scores. We can find the most reliable path (highest minimum confidence along the path).

```python
def most_reliable_path(G, source, target, weight_key="weight"):
    """Find the path that maximizes the minimum edge weight (bottleneck path)."""
    all_paths = nx.all_simple_paths(G, source, target, cutoff=6)

    best_path = None
    best_min_weight = -1

    for path in all_paths:
        min_weight = float("inf")
        for i in range(len(path) - 1):
            w = G[path[i]][path[i+1]].get(weight_key, 0)
            min_weight = min(min_weight, w)

        if min_weight > best_min_weight:
            best_min_weight = min_weight
            best_path = path

    return best_path, best_min_weight

path, confidence = most_reliable_path(G, "Alice", "Heidi")
print(f"Most reliable path: {' -> '.join(path)}")
print(f"Bottleneck confidence: {confidence}")
```

### Path Score as Product of Confidences

```python
def highest_confidence_path(G, source, target, weight_key="weight"):
    """Find path with highest product of edge weights."""
    import math

    # Transform: maximize product = minimize sum of -log(weight)
    for u, v, d in G.edges(data=True):
        w = d.get(weight_key, 0.5)
        d["neg_log_weight"] = -math.log(max(w, 1e-10))

    try:
        path = nx.dijkstra_path(G, source, target, weight="neg_log_weight")
        # Compute actual confidence (product)
        confidence = 1.0
        for i in range(len(path) - 1):
            confidence *= G[path[i]][path[i+1]].get(weight_key, 0.5)
        return path, confidence
    except nx.NetworkXNoPath:
        return None, 0

path, conf = highest_confidence_path(G, "Alice", "Heidi")
print(f"Highest confidence path: {' -> '.join(path)} (conf={conf:.4f})")
```

---

## K-Shortest Paths

Find multiple alternative paths for diverse explanations:

```python
def k_shortest_paths(G, source, target, k=5, weight=None):
    """Find the K shortest simple paths."""
    return list(nx.shortest_simple_paths(G, source, target, weight=weight))[:k]

# Top 3 shortest paths
paths = k_shortest_paths(G, "Alice", "Heidi", k=3)
for i, path in enumerate(paths):
    print(f"  Path {i+1} (length={len(path)-1}): {' -> '.join(path)}")
```

### Use Case: Alternative Explanations

```python
def explain_connection(G, entity1, entity2, max_explanations=3):
    """Generate multiple explanation paths between two entities."""
    paths = k_shortest_paths(G, entity1, entity2, k=max_explanations)

    print(f"How is {entity1} connected to {entity2}?")
    for i, path in enumerate(paths):
        print(f"\n  Explanation {i+1}:")
        for j in range(len(path) - 1):
            rel = G[path[j]][path[j+1]].get("relation", "related_to")
            print(f"    {path[j]} --[{rel}]--> {path[j+1]}")

explain_connection(G, "Alice", "Grace")
```

---

## Multi-Hop Reasoning Paths

### Reasoning Chains

Extract reasoning chains for question answering:

```python
def find_reasoning_chain(G, start, end, max_hops=4):
    """Find a reasoning chain with relationship explanations."""
    try:
        path = nx.shortest_path(G, start, end)
    except nx.NetworkXNoPath:
        return None

    if len(path) - 1 > max_hops:
        return None

    chain = []
    for i in range(len(path) - 1):
        edge = G[path[i]][path[i+1]]
        chain.append({
            "from": path[i],
            "to": path[i+1],
            "relation": edge.get("relation", "unknown"),
            "confidence": edge.get("weight", 1.0),
        })

    return chain

chain = find_reasoning_chain(G, "Alice", "Heidi")
if chain:
    print("Reasoning chain:")
    for step in chain:
        print(f"  {step['from']} --[{step['relation']}]--> {step['to']} "
              f"(conf={step['confidence']})")
```

### Multi-Hop Queries

```python
def multi_hop_query(G, start, relations, max_results=10):
    """
    Follow a sequence of relation types from a start entity.
    Example: ("Alice", ["knows", "works_with"]) = who do Alice's friends work with?
    """
    current_entities = {start}

    for rel in relations:
        next_entities = set()
        for entity in current_entities:
            for neighbor in G.successors(entity):
                if G[entity][neighbor].get("relation") == rel:
                    next_entities.add(neighbor)
        current_entities = next_entities

        if not current_entities:
            return set()

    return current_entities

# Who do Alice's friends work with?
result = multi_hop_query(G, "Alice", ["knows", "works_with"])
print(f"Alice's friends' coworkers: {result}")

# Who do Alice's friends' friends know?
result = multi_hop_query(G, "Alice", ["knows", "knows", "knows"])
print(f"3-hop knows from Alice: {result}")
```

---

## Cypher Equivalents

For Neo4j users, here are the Cypher equivalents of key operations:

### Shortest Path

```cypher
// Shortest path
MATCH p = shortestPath((a:Person {name: "Alice"})-[*]-(h:Person {name: "Heidi"}))
RETURN p, length(p)

// All shortest paths
MATCH p = allShortestPaths((a:Person {name: "Alice"})-[*]-(h:Person {name: "Heidi"}))
RETURN p
```

### Variable-Length Paths

```cypher
// Paths of 1-3 hops
MATCH p = (a:Person {name: "Alice"})-[*1..3]-(target)
RETURN target.name, length(p)

// With specific relationship type
MATCH p = (a:Person {name: "Alice"})-[:KNOWS*1..3]-(target)
RETURN target.name, length(p)
```

### Multi-Hop Reasoning

```cypher
// Who do Alice's friends work with?
MATCH (a:Person {name: "Alice"})-[:KNOWS]->(friend)-[:WORKS_WITH]->(coworker)
RETURN coworker.name

// 3-hop knows chain
MATCH (a:Person {name: "Alice"})-[:KNOWS*3]->(distant)
RETURN DISTINCT distant.name
```

### Weighted Shortest Path

```cypher
// Weighted shortest path (requires GDS library)
CALL gds.shortestPath.dijkstra.stream({
    nodeProjection: '*',
    relationshipProjection: {ALL: {properties: 'weight'}},
    sourceNode: a,
    targetNode: h,
    relationshipWeightProperty: 'weight'
})
YIELD path, totalCost
RETURN path, totalCost
```

---

## Performance Tips

1. **Set a cutoff** for `all_simple_paths` -- without it, the number of paths grows exponentially
2. **Use `shortest_path` for single queries**, `shortest_path_length` if you only need the distance
3. **BFS is optimal** for unweighted shortest paths; Dijkstra for weighted
4. **For large graphs**, consider approximate algorithms or limit exploration depth
5. **Cache path results** if the same queries are repeated (the graph is usually static)
6. **Use `nx.has_path(G, s, t)`** before computing paths to avoid exceptions

```python
# Check connectivity before pathfinding
if nx.has_path(G, "Alice", "Heidi"):
    path = nx.shortest_path(G, "Alice", "Heidi")
else:
    print("No path exists")
```

---

## Key Takeaways

1. **Shortest path is the workhorse** -- use it for finding connections and reasoning chains
2. **K-shortest paths provide diversity** -- essential for generating multiple explanations
3. **Weighted paths incorporate confidence** -- critical for real-world KGs with noisy data
4. **Multi-hop queries implement graph patterns** -- the basis for complex KG question answering
5. **BFS for exploration, DFS for deep analysis** -- both are useful for different KG tasks
6. **NetworkX for prototyping, Cypher for production** -- the algorithms are the same

---

## References

- Dijkstra, E. W. (1959). "A Note on Two Problems in Connexion with Graphs." Numerische Mathematik.
- NetworkX path algorithms: https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html
- Neo4j Cypher manual: https://neo4j.com/docs/cypher-manual/
