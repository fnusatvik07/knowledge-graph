# Graph Databases

Graph databases are purpose-built to store and query graph-structured data. Unlike relational databases that use joins to traverse relationships, graph databases use index-free adjacency, making relationship traversal constant-time regardless of data size.

## Why Graph Databases?

Traditional relational databases struggle with highly connected data. A query like "find friends of friends who share my interests" requires multiple expensive JOIN operations. In a graph database, this is a simple traversal.

Key advantages:
- **Performance**: Relationship queries are orders of magnitude faster
- **Flexibility**: Schema-free or schema-optional, easy to evolve
- **Intuitiveness**: The data model matches how we think about connected information

## Popular Graph Databases

### Neo4j
The most popular graph database. Uses the Cypher query language and the labeled property graph model. Excellent for [[knowledge_graphs]], fraud detection, and recommendation engines.

```cypher
MATCH (p:Person)-[:KNOWS]->(friend)-[:LIKES]->(movie:Movie)
WHERE p.name = "Alice"
RETURN movie.title, count(friend) as recommendations
ORDER BY recommendations DESC
```

### Amazon Neptune
Managed graph database supporting both property graphs (Gremlin) and RDF (SPARQL). Good for cloud-native applications.

### ArangoDB
Multi-model database supporting graphs, documents, and key-value. Uses AQL query language.

## Data Models

- **Property Graph**: Nodes and edges have labels and properties. Used by Neo4j, TigerGraph.
- **RDF (Resource Description Framework)**: Triple-based (subject, predicate, object). W3C standard for the semantic web. Queried with SPARQL.

## Graph Databases and Knowledge Graphs

[[knowledge_graphs]] are a natural fit for graph databases. The triple structure of KGs maps directly to the node-edge-node structure of graph databases, enabling efficient storage and querying of complex knowledge networks.

#graph-databases #neo4j #data-storage #nosql
