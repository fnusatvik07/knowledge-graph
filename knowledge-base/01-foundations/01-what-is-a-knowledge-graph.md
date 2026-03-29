# What is a Knowledge Graph?

A **knowledge graph** is a structured representation of real-world entities and the relationships between them. It organizes information as a network of interconnected facts, making it possible to reason about how concepts relate to each other.

## The Core Idea

At its simplest, a knowledge graph stores information as **triples**:

```
(Subject) --[Relationship]--> (Object)
```

For example:
```
(Albert Einstein) --[born_in]--> (Ulm, Germany)
(Albert Einstein) --[developed]--> (Theory of Relativity)
(Theory of Relativity) --[field_of]--> (Physics)
```

These triples form a directed graph where entities are **nodes** and relationships are **edges**.

## Why Knowledge Graphs Matter

### 1. They Capture Context
Unlike a flat database table or a bag of text chunks, a knowledge graph preserves the *relationships* between pieces of information. When you know that Einstein developed the Theory of Relativity AND that the Theory of Relativity belongs to Physics, you can infer that Einstein contributed to Physics — even if that fact was never explicitly stated.

### 2. They Enable Multi-Hop Reasoning
By traversing edges in the graph, you can answer questions that require connecting multiple facts:
- "What field did the person born in Ulm contribute to?" → Follow: Ulm ← Einstein → Relativity → Physics

### 3. They Handle Heterogeneous Information
A knowledge graph can seamlessly combine information from different sources, formats, and domains into a single unified structure.

## Real-World Knowledge Graphs

| Knowledge Graph | Creator | Scale |
|----------------|---------|-------|
| Google Knowledge Graph | Google | 500B+ facts |
| Wikidata | Wikimedia Foundation | 100M+ items |
| DBpedia | Community | Extracted from Wikipedia |
| YAGO | Max Planck Institute | 10M+ entities |

## Knowledge Graphs vs Other Data Structures

| Feature | Relational DB | Document Store | Vector DB | Knowledge Graph |
|---------|--------------|----------------|-----------|-----------------|
| Schema | Rigid tables | Schema-free | Embeddings | Flexible typed nodes/edges |
| Relationships | Foreign keys (flat) | Nested docs | Similarity | First-class, typed, directional |
| Queries | SQL joins | Document lookup | Nearest neighbor | Graph traversal, pattern matching |
| Reasoning | Limited | None | Semantic similarity | Multi-hop, logical inference |

## How Knowledge Graphs Relate to AI

In the context of LLMs and RAG systems, knowledge graphs serve as a **structured memory** that complements the LLM's parametric knowledge:

1. **Construction**: LLMs extract entities and relationships from unstructured text to *build* knowledge graphs
2. **Retrieval**: Graph traversal retrieves structured, connected context that vector search alone would miss
3. **Generation**: The LLM uses graph-retrieved context to generate more accurate, well-grounded answers

This is the foundation of **Graph RAG**, which we'll explore throughout this repository.

## Key Takeaways

- A knowledge graph stores facts as (subject, relationship, object) triples
- Nodes represent entities, edges represent relationships
- KGs enable multi-hop reasoning by traversing connections
- In AI/RAG systems, KGs provide structured context that vector search misses
