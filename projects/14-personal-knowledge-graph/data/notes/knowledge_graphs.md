# Knowledge Graphs

A knowledge graph represents information as a network of entities and their relationships. Unlike relational databases that store data in tables, knowledge graphs capture the rich, interconnected nature of real-world knowledge.

## Structure

Knowledge graphs are built on triples: (subject, predicate, object). For example:
- (Albert Einstein, born_in, Ulm)
- (Albert Einstein, developed, Theory of Relativity)
- (Theory of Relativity, field, Physics)

Entities are nodes, relationships are edges. Both can have properties and types, forming a rich semantic network.

## Building Knowledge Graphs

There are several approaches to constructing KGs:
- **Manual curation**: Domain experts define entities and relationships (e.g., Wikidata)
- **Information extraction**: NLP techniques extract triples from text using [[machine_learning]]
- **Data integration**: Combine structured data from multiple sources
- **LLM-powered extraction**: Use large language models for entity and relationship extraction

## Applications

- **Search engines**: Google's Knowledge Graph enhances search results with structured information
- **Recommendation systems**: Model user-item relationships for personalized recommendations
- **Drug discovery**: Represent molecular interactions and disease pathways
- **Enterprise knowledge management**: Connect organizational knowledge across departments
- **Question answering**: Enable multi-hop reasoning over structured knowledge

## Storage

Knowledge graphs are stored in [[graph_databases]] like Neo4j, Amazon Neptune, or Apache Jena for RDF data. Each has different query languages (Cypher, SPARQL, Gremlin) and performance characteristics.

## KGs and AI

The combination of knowledge graphs with machine learning (neuro-symbolic AI) is a promising research direction, enabling both statistical learning and logical reasoning.

#knowledge-graphs #graph-theory #data-modeling #ai
