# NetworkX Quickstart for Knowledge Graphs

**NetworkX** is a Python library for creating, manipulating, and studying the structure of complex networks. It is the fastest way to prototype a knowledge graph without setting up any infrastructure.

## Why NetworkX for Knowledge Graphs?

- **Zero infrastructure** -- pure Python, no server or database required
- **Rich algorithm library** -- centrality, shortest path, community detection out of the box
- **Flexible data model** -- nodes and edges can carry arbitrary Python dictionaries as attributes
- **Excellent for prototyping** -- validate your ontology and extraction pipeline before committing to a production database

### When NOT to Use NetworkX

NetworkX stores everything in memory. Once your graph exceeds roughly 100K nodes or you need concurrent read/write access, move to a dedicated graph database such as Neo4j (covered in the next guide).

---

## Installation

```bash
pip install networkx
```

For visualization support:

```bash
pip install matplotlib pyvis
```

---

## Creating a Directed Knowledge Graph

Knowledge graphs are inherently directional -- "Einstein developed Relativity" is not the same as "Relativity developed Einstein." Use `nx.DiGraph`.

```python
import networkx as nx

kg = nx.DiGraph()
```

### Adding Nodes with Attributes

Each node represents an **entity** in your knowledge graph. Attach metadata via keyword arguments.

```python
# People
kg.add_node("Albert Einstein", type="Person", birth_year=1879, nationality="German")
kg.add_node("Niels Bohr", type="Person", birth_year=1885, nationality="Danish")
kg.add_node("Marie Curie", type="Person", birth_year=1867, nationality="Polish")

# Organizations
kg.add_node("ETH Zurich", type="Organization", founded=1855, country="Switzerland")
kg.add_node("University of Copenhagen", type="Organization", founded=1479, country="Denmark")
kg.add_node("Sorbonne", type="Organization", founded=1257, country="France")

# Concepts
kg.add_node("Theory of Relativity", type="Concept", field="Physics")
kg.add_node("Quantum Mechanics", type="Concept", field="Physics")
kg.add_node("Radioactivity", type="Concept", field="Chemistry")

# Awards
kg.add_node("Nobel Prize in Physics", type="Award")
kg.add_node("Nobel Prize in Chemistry", type="Award")
```

### Adding Edges with Properties

Edges represent **relationships**. Attach a `relation` property (and any other metadata) to each edge.

```python
# Employment / affiliation
kg.add_edge("Albert Einstein", "ETH Zurich", relation="affiliated_with", role="Professor", year_start=1912)
kg.add_edge("Niels Bohr", "University of Copenhagen", relation="affiliated_with", role="Professor")
kg.add_edge("Marie Curie", "Sorbonne", relation="affiliated_with", role="Professor")

# Scientific contributions
kg.add_edge("Albert Einstein", "Theory of Relativity", relation="developed", year=1905)
kg.add_edge("Albert Einstein", "Quantum Mechanics", relation="contributed_to")
kg.add_edge("Niels Bohr", "Quantum Mechanics", relation="developed", year=1913)
kg.add_edge("Marie Curie", "Radioactivity", relation="discovered")

# Awards
kg.add_edge("Albert Einstein", "Nobel Prize in Physics", relation="received", year=1921)
kg.add_edge("Niels Bohr", "Nobel Prize in Physics", relation="received", year=1922)
kg.add_edge("Marie Curie", "Nobel Prize in Physics", relation="received", year=1903)
kg.add_edge("Marie Curie", "Nobel Prize in Chemistry", relation="received", year=1911)

# Collaborations
kg.add_edge("Albert Einstein", "Niels Bohr", relation="debated_with", topic="Quantum Mechanics")
kg.add_edge("Niels Bohr", "Albert Einstein", relation="debated_with", topic="Quantum Mechanics")
```

### Inspecting the Graph

```python
print(f"Nodes: {kg.number_of_nodes()}")
print(f"Edges: {kg.number_of_edges()}")

# List all nodes of a given type
people = [n for n, d in kg.nodes(data=True) if d.get("type") == "Person"]
print(f"People: {people}")

# Get all attributes of a node
print(kg.nodes["Albert Einstein"])
# {'type': 'Person', 'birth_year': 1879, 'nationality': 'German'}

# Get all outgoing edges from a node
for _, target, data in kg.out_edges("Marie Curie", data=True):
    print(f"  Marie Curie --[{data['relation']}]--> {target}")
```

---

## Querying the Graph

### Find All Neighbors of an Entity

```python
# Direct successors (outgoing edges)
successors = list(kg.successors("Albert Einstein"))
print(successors)
# ['ETH Zurich', 'Theory of Relativity', 'Quantum Mechanics',
#  'Nobel Prize in Physics', 'Niels Bohr']

# Direct predecessors (incoming edges)
predecessors = list(kg.predecessors("Quantum Mechanics"))
print(predecessors)
# ['Albert Einstein', 'Niels Bohr']
```

### Filter Edges by Relationship Type

```python
def get_related(graph, entity, relation):
    """Return all targets connected to entity by a specific relation."""
    return [
        target
        for _, target, data in graph.out_edges(entity, data=True)
        if data.get("relation") == relation
    ]

awards = get_related(kg, "Marie Curie", "received")
print(awards)
# ['Nobel Prize in Physics', 'Nobel Prize in Chemistry']
```

### Multi-Hop Queries

"Which organizations are affiliated with people who contributed to Quantum Mechanics?"

```python
def multi_hop_query(graph, concept, inbound_relation, outbound_relation):
    """Two-hop traversal: concept <--[inbound]-- person --[outbound]--> target."""
    results = []
    # Hop 1: find people connected to the concept
    for person in graph.predecessors(concept):
        edge_data = graph.edges[person, concept]
        if edge_data.get("relation") == inbound_relation:
            # Hop 2: find targets of those people
            for _, target, data in graph.out_edges(person, data=True):
                if data.get("relation") == outbound_relation:
                    results.append((person, target))
    return results

orgs = multi_hop_query(kg, "Quantum Mechanics", "developed", "affiliated_with")
print(orgs)
# [('Niels Bohr', 'University of Copenhagen')]
```

---

## Graph Algorithms

NetworkX ships with dozens of algorithms that are useful for analyzing knowledge graphs.

### Degree Centrality

Identifies the most connected entities in the graph.

```python
centrality = nx.degree_centrality(kg)
sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

print("Top entities by degree centrality:")
for node, score in sorted_centrality[:5]:
    print(f"  {node}: {score:.3f}")
```

### Betweenness Centrality

Identifies entities that act as bridges between different parts of the graph.

```python
betweenness = nx.betweenness_centrality(kg)
sorted_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)

print("Top bridge entities:")
for node, score in sorted_betweenness[:5]:
    print(f"  {node}: {score:.3f}")
```

### Shortest Path

Find the shortest path between two entities. Useful for explaining how two concepts are connected.

```python
if nx.has_path(kg, "Marie Curie", "ETH Zurich"):
    path = nx.shortest_path(kg, "Marie Curie", "ETH Zurich")
    print(" -> ".join(path))

# All shortest paths
all_paths = list(nx.all_shortest_paths(kg, "Marie Curie", "ETH Zurich"))
for p in all_paths:
    print(" -> ".join(p))
```

### Connected Components

For undirected analysis, find clusters of related entities.

```python
# Convert to undirected for component analysis
undirected = kg.to_undirected()
components = list(nx.connected_components(undirected))
print(f"Number of connected components: {len(components)}")
for i, comp in enumerate(components):
    print(f"  Component {i}: {comp}")
```

### PageRank

Rank entities by importance, the same algorithm Google uses.

```python
pagerank = nx.pagerank(kg)
sorted_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)

print("PageRank scores:")
for node, score in sorted_pr[:5]:
    print(f"  {node}: {score:.4f}")
```

---

## Subgraph Extraction

Extract a portion of the graph around a specific entity -- useful for providing focused context to an LLM.

```python
def extract_subgraph(graph, center_node, max_hops=2):
    """Extract the ego graph (local neighborhood) around a node."""
    # Get all nodes within max_hops
    nodes = set()
    current_layer = {center_node}
    for _ in range(max_hops):
        next_layer = set()
        for node in current_layer:
            next_layer.update(graph.successors(node))
            next_layer.update(graph.predecessors(node))
        nodes.update(current_layer)
        current_layer = next_layer - nodes
    nodes.update(current_layer)

    return graph.subgraph(nodes).copy()

einstein_subgraph = extract_subgraph(kg, "Albert Einstein", max_hops=1)
print(f"Subgraph: {einstein_subgraph.number_of_nodes()} nodes, "
      f"{einstein_subgraph.number_of_edges()} edges")
```

---

## Serialization

### GraphML (XML-based, widely supported)

```python
# Write
nx.write_graphml(kg, "knowledge_graph.graphml")

# Read
kg_loaded = nx.read_graphml("knowledge_graph.graphml")
```

### JSON (node-link format)

```python
import json
from networkx.readwrite import json_graph

# Write
data = json_graph.node_link_data(kg)
with open("knowledge_graph.json", "w") as f:
    json.dump(data, f, indent=2)

# Read
with open("knowledge_graph.json", "r") as f:
    data = json.load(f)
kg_loaded = json_graph.node_link_graph(data, directed=True)
```

### GEXF (used by Gephi for visualization)

```python
nx.write_gexf(kg, "knowledge_graph.gexf")
```

### Pickle (fast, Python-only)

```python
import pickle

with open("knowledge_graph.pkl", "wb") as f:
    pickle.dump(kg, f)

with open("knowledge_graph.pkl", "rb") as f:
    kg_loaded = pickle.load(f)
```

---

## Visualization

### Quick Matplotlib Plot

```python
import matplotlib.pyplot as plt

pos = nx.spring_layout(kg, seed=42)
node_colors = ["#4CAF50" if kg.nodes[n].get("type") == "Person"
               else "#2196F3" if kg.nodes[n].get("type") == "Organization"
               else "#FF9800" for n in kg.nodes()]

nx.draw(kg, pos, with_labels=True, node_color=node_colors,
        node_size=2000, font_size=8, arrows=True)
plt.title("Knowledge Graph")
plt.tight_layout()
plt.savefig("kg_visualization.png", dpi=150)
plt.show()
```

### Interactive HTML with PyVis

```python
from pyvis.network import Network

net = Network(notebook=False, directed=True, height="600px", width="100%")
net.from_nx(kg)
net.show("kg_interactive.html")
```

---

## Converting NetworkX to Neo4j

When your prototype is ready for production, export to Cypher statements.

```python
def networkx_to_cypher(graph):
    """Generate Cypher CREATE statements from a NetworkX graph."""
    statements = []

    for node, attrs in graph.nodes(data=True):
        label = attrs.get("type", "Entity")
        props = {k: v for k, v in attrs.items() if k != "type"}
        prop_str = ", ".join(f'{k}: "{v}"' if isinstance(v, str) else f"{k}: {v}"
                            for k, v in props.items())
        safe_name = node.replace(" ", "_").replace("-", "_")
        statements.append(f'CREATE ({safe_name}:{label} {{name: "{node}", {prop_str}}})')

    for source, target, attrs in graph.edges(data=True):
        rel_type = attrs.get("relation", "RELATED_TO").upper()
        props = {k: v for k, v in attrs.items() if k != "relation"}
        prop_str = ", ".join(f'{k}: "{v}"' if isinstance(v, str) else f"{k}: {v}"
                            for k, v in props.items())
        safe_source = source.replace(" ", "_").replace("-", "_")
        safe_target = target.replace(" ", "_").replace("-", "_")
        if prop_str:
            statements.append(f'CREATE ({safe_source})-[:{rel_type} {{{prop_str}}}]->({safe_target})')
        else:
            statements.append(f'CREATE ({safe_source})-[:{rel_type}]->({safe_target})')

    return ";\n".join(statements) + ";"

print(networkx_to_cypher(kg))
```

---

## Summary

| Feature | Details |
|---------|---------|
| Best for | Prototyping, small graphs, algorithm exploration |
| Max practical size | ~100K nodes (memory-bound) |
| Setup | `pip install networkx` |
| Persistence | File-based (GraphML, JSON, Pickle) |
| Concurrency | None (single-process) |
| Algorithm library | 50+ built-in algorithms |

NetworkX is the ideal starting point for any knowledge graph project. Build your extraction pipeline, validate your ontology, and run structural analyses -- then graduate to Neo4j when you need persistence, scale, or concurrent access.
