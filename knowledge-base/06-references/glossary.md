# Glossary

Key terms used throughout this repository.

## A

**Approximate Nearest Neighbor (ANN)**: An algorithm for finding the closest vectors in a high-dimensional space without exhaustive comparison. Used in vector databases for fast retrieval.

**APOC (Awesome Procedures on Cypher)**: A Neo4j plugin library providing hundreds of useful procedures and functions for graph data processing.

## B

**BM25**: A probabilistic ranking function used in keyword-based information retrieval. Used alongside vector search in hybrid retrieval systems like Graphiti.

## C

**Community**: A cluster of densely connected nodes in a graph, detected by algorithms like Leiden. In GraphRAG, communities are summarized to enable global search.

**Community Detection**: The process of finding groups of related nodes in a graph. The Leiden algorithm is the standard in GraphRAG.

**Community Summarization**: Generating a natural language summary of a graph community using an LLM. These summaries enable answering broad, thematic questions.

**Cypher**: The query language for Neo4j and other property graph databases. Designed for expressive, readable graph pattern matching.

## D

**DRIFT Search**: Dynamic Reasoning and Inference with Flexible Traversal. A GraphRAG search mode that combines global and local search in three phases: primer, follow-up, and output.

**Directed Graph (Digraph)**: A graph where edges have direction — they go from one node to another.

## E

**Edge**: A connection between two nodes in a graph. In a knowledge graph, edges represent relationships between entities.

**Embedding**: A dense vector representation of text, images, or graph elements in a continuous vector space. Similar items have similar embeddings.

**Entity**: A real-world thing represented as a node in a knowledge graph — a person, organization, concept, etc.

**Entity Resolution**: The process of determining that two or more references point to the same real-world entity (e.g., "Einstein" and "Albert Einstein").

## G

**Global Search**: A GraphRAG search mode that answers broad, thematic questions by searching across community summaries using map-reduce.

**Graph Database**: A database that uses graph structures (nodes, edges, properties) as its fundamental data model. Examples: Neo4j, Amazon Neptune, Memgraph.

**Graph Embedding**: A vector representation of a node, edge, or subgraph in a continuous vector space. Methods include TransE, RotatE, and node2vec.

**Graph RAG**: Retrieval Augmented Generation enhanced with a knowledge graph layer that captures entities, relationships, and community structures from documents.

**GraphML**: An XML-based file format for graphs. Supported by NetworkX and many graph tools.

## H

**Hop**: One step along an edge in a graph traversal. A "2-hop query" follows two edges from a starting node.

**Hybrid Retrieval**: Combining multiple retrieval strategies (typically vector search + graph traversal) and fusing their results for higher quality answers.

## K

**Knowledge Graph (KG)**: A graph-structured knowledge base where nodes represent entities and edges represent relationships between them. Stores facts as (subject, predicate, object) triples.

## L

**Leiden Algorithm**: A community detection algorithm that improves upon the Louvain algorithm. Used in GraphRAG for hierarchical community detection. Named after Leiden University.

**LightRAG**: A lightweight alternative to Microsoft's GraphRAG that uses simpler extraction, a flat graph structure, and dual-mode retrieval at ~1/100th the cost.

**Local Search**: A GraphRAG search mode that answers specific, entity-focused questions by retrieving the entity's neighborhood in the graph.

**LLM (Large Language Model)**: A neural network trained on large amounts of text data. Used in Graph RAG for entity extraction, relationship extraction, summarization, and answer generation.

## M

**Map-Reduce**: A programming model for processing large datasets in parallel. In GraphRAG, global search uses map-reduce over community summaries: each community is mapped (summarized), then results are reduced (combined).

**Multi-Hop Reasoning**: Answering questions that require following chains of relationships across multiple entities in a knowledge graph.

## N

**Node**: A fundamental unit in a graph. In a knowledge graph, nodes represent entities.

**NetworkX**: A Python library for creating, manipulating, and studying complex networks and graphs.

**Neo4j**: The world's most popular graph database. Uses the Cypher query language and the property graph model.

## O

**Ontology**: A formal specification of entity types, relationship types, and constraints in a knowledge graph. The "schema" of the graph.

## P

**Property Graph**: A graph model where both nodes and edges can have key-value properties. Used by Neo4j and most modern graph databases.

## R

**RAG (Retrieval Augmented Generation)**: A technique that enhances LLM responses by retrieving relevant external context from a document corpus.

**Reciprocal Rank Fusion (RRF)**: A method for combining ranked lists from multiple retrieval systems. Used in hybrid retrieval to merge vector and graph search results.

**Relationship**: A directed connection between two entities in a knowledge graph. Also called an edge or link.

## S

**Subgraph**: A subset of nodes and edges from a larger graph. Query results from a knowledge graph are typically subgraphs.

## T

**Temporal Knowledge Graph**: A knowledge graph where facts have time-validity metadata (valid_from, valid_to), enabling reasoning about when information was true.

**TransE**: A knowledge graph embedding method that represents relationships as translations in the embedding space.

**Triple**: The atomic unit of a knowledge graph: (subject, predicate, object). Also called a fact or statement.

## V

**Vector Database**: A database optimized for storing and searching high-dimensional vectors. Examples: ChromaDB, Pinecone, Weaviate.

**Vector RAG**: Traditional RAG using vector embeddings and similarity search for retrieval, without a knowledge graph layer.
