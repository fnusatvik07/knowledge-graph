# Graph Terminology

Before diving into knowledge graphs and Graph RAG, let's establish the vocabulary used throughout this repository.

## Basic Graph Concepts

### Nodes (Vertices)
The fundamental units of a graph. In a knowledge graph, nodes represent **entities** — people, places, concepts, events, or any distinct thing.

```
Node: { id: "einstein", label: "Albert Einstein", type: "Person" }
```

### Edges (Links, Relationships)
Connections between nodes. In a knowledge graph, edges represent **relationships** and are typically **directed** (they have a source and a target).

```
Edge: (Einstein) --[DEVELOPED]--> (Theory of Relativity)
       source       relationship        target
```

### Properties (Attributes)
Key-value pairs attached to nodes or edges that store additional information.

```
Node: Albert Einstein
  Properties: { birth_year: 1879, nationality: "German" }

Edge: DEVELOPED
  Properties: { year: 1905, confidence: 0.95 }
```

## Graph Types

### Directed Graph (Digraph)
Edges have a direction — they go *from* one node *to* another. Most knowledge graphs are directed.

```
(A) --> (B)    # A points to B, but B does not point to A
```

### Undirected Graph
Edges have no direction — the relationship is mutual.

```
(A) --- (B)    # A and B are connected symmetrically
```

### Property Graph
A graph where both nodes and edges can have properties (key-value pairs). This is the model used by **Neo4j** and most modern graph databases.

```
(:Person {name: "Einstein"}) -[:DEVELOPED {year: 1905}]-> (:Theory {name: "Relativity"})
```

### RDF Graph (Resource Description Framework)
A graph based on subject-predicate-object triples. Used in semantic web and linked data. Each element is identified by a URI.

```
<http://dbpedia.org/resource/Einstein> <http://dbpedia.org/ontology/developed> <http://dbpedia.org/resource/Relativity>
```

## Knowledge Graph Specific Terms

### Triple
The atomic unit of a knowledge graph: `(subject, predicate, object)`. Also called a **fact** or **statement**.

### Entity
A real-world thing represented as a node — a person, organization, location, concept, etc.

### Relation Type
The category of relationship between entities: `WORKS_AT`, `BORN_IN`, `DEVELOPED`, `PART_OF`, etc.

### Ontology
A formal definition of the entity types and relationship types allowed in a knowledge graph. Think of it as the "schema" of the graph.

```
Ontology:
  Entity Types: Person, Organization, Location, Concept
  Relation Types:
    WORKS_AT: Person -> Organization
    BORN_IN: Person -> Location
    PART_OF: Concept -> Concept
```

### Community
A group of densely connected nodes within the graph. In Graph RAG, communities are detected using algorithms like **Leiden** and summarized to enable global reasoning.

## Graph Traversal Terms

### Neighbor
A node directly connected to a given node by a single edge.

### Degree
The number of edges connected to a node. In directed graphs:
- **In-degree**: number of incoming edges
- **Out-degree**: number of outgoing edges

### Path
A sequence of nodes connected by edges: `A → B → C → D`

### Hop
One step along an edge in a traversal. A "2-hop query" means following two edges from a starting node.

### Subgraph
A subset of nodes and edges from a larger graph. When querying a knowledge graph, you typically retrieve a **relevant subgraph** rather than the entire graph.

## Metrics and Analysis

### Centrality
How "important" a node is in the graph. Common measures:
- **Degree centrality**: nodes with many connections
- **Betweenness centrality**: nodes that sit on many shortest paths
- **PageRank**: nodes connected to other important nodes

### Clustering Coefficient
How connected a node's neighbors are to each other. High clustering = tightly knit community.

### Connected Component
A maximal set of nodes where every node can reach every other node via some path.

## Summary Table

| Term | Meaning | Example |
|------|---------|---------|
| Node | Entity in the graph | "Albert Einstein" |
| Edge | Relationship between entities | DEVELOPED |
| Triple | (Subject, Predicate, Object) | (Einstein, DEVELOPED, Relativity) |
| Property | Attribute on a node or edge | birth_year: 1879 |
| Ontology | Schema defining types and relations | Person → WORKS_AT → Organization |
| Community | Cluster of related nodes | "Physics researchers" cluster |
| Hop | One traversal step | Einstein → Relativity (1 hop) |
| Subgraph | Extracted portion of a graph | All nodes within 2 hops of Einstein |
