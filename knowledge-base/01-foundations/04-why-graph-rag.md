# Why Graph RAG?

Graph RAG augments the standard RAG pipeline with a **knowledge graph layer** that captures entities, relationships, and community structures from your documents. This section explains the key motivations and when to choose Graph RAG over traditional approaches.

## The Core Insight

Traditional RAG treats documents as independent bags of text chunks. Graph RAG treats documents as a **connected knowledge network** where entities and relationships span across chunks and documents.

```
Traditional RAG:
  Doc1 → [chunk1, chunk2, chunk3]     (isolated)
  Doc2 → [chunk4, chunk5, chunk6]     (isolated)

Graph RAG:
  Doc1 + Doc2 → Knowledge Graph where entities from Doc1
                 connect to entities in Doc2 via shared relationships
```

## What Graph RAG Adds to the Pipeline

### 1. Entity and Relationship Extraction
An LLM reads each text chunk and extracts structured information:
```
Text: "Einstein developed the theory of relativity at the University of Zurich."

Entities: [Einstein (Person), Theory of Relativity (Theory), University of Zurich (Organization)]
Relations: [(Einstein, DEVELOPED, Theory of Relativity), (Einstein, WORKED_AT, University of Zurich)]
```

### 2. Knowledge Graph Construction
Extracted entities and relationships are assembled into a graph. Entities that appear in multiple chunks or documents are merged into single nodes, creating cross-document connections.

### 3. Community Detection
The Leiden algorithm groups densely connected nodes into **communities** — clusters of related entities. These communities form a hierarchy:
- **Level 0**: Fine-grained communities (2-10 entities)
- **Level 1**: Broader groupings
- **Level 2+**: High-level themes

### 4. Community Summarization
An LLM generates a natural language summary of each community, capturing the key entities, relationships, and themes within that cluster.

### 5. Multi-Mode Search
- **Local Search**: For specific questions — retrieves the entity and its neighborhood
- **Global Search**: For broad questions — searches community summaries via map-reduce
- **DRIFT Search**: Combines both — starts global, drills down locally

## When to Use Graph RAG

### Use Graph RAG When:
- Your questions require reasoning across multiple documents
- You need to answer "How is X related to Y?" type questions
- You want global summarization ("What are the main themes?")
- Your corpus has rich entity relationships (people, organizations, events, concepts)
- You need to handle multi-hop queries
- The same entities appear across many documents

### Stick with Vector RAG When:
- Your questions are simple fact lookups
- Documents are independent (no cross-document relationships matter)
- You need real-time indexing (Graph RAG indexing is expensive)
- Cost is the primary concern
- Your corpus is small and simple

### Use Both (Hybrid) When:
- You need to handle both simple and complex queries
- Production reliability matters
- You want the best of both approaches

## The Cost-Quality Trade-off

Graph RAG's main drawback is **cost**. The indexing phase requires LLM calls to extract entities from every chunk:

| Approach | Indexing Cost (1000 pages) | Query Latency | Answer Quality (complex Q) |
|----------|---------------------------|---------------|---------------------------|
| Vector RAG | ~$1-5 | ~100ms | Low-Medium |
| LightRAG | ~$5-50 | ~80ms | Medium-High |
| Microsoft GraphRAG | ~$50-500 | ~2-5s | High |
| Hybrid (Vector + Graph) | ~$50-500 | ~1-3s | Highest |

The ecosystem has evolved to address this cost problem:
- **LightRAG**: 1/100th the cost of GraphRAG with 70-90% quality
- **Dynamic Community Selection**: 79% token reduction in GraphRAG
- **LinearRAG**: Relation-free graph construction (even cheaper)

## The Evolution of Graph RAG (2024-2026)

```
2024 Apr: Microsoft publishes GraphRAG paper
          ↓
2024 Jul: Open-source graphrag library released
          ↓
2024 Oct: LightRAG offers 100x cheaper alternative
          ↓
2025 Jan: Dynamic Community Selection reduces costs 79%
          ↓
2025 Jun: GraphRAG Benchmark published for evaluation
          ↓
2025 Oct: LinearRAG (relation-free) accepted at ICLR 2026
          ↓
2026:     Hybrid (vector + graph) is the production default
          Temporal knowledge graphs (Graphiti) for agent memory
          Ontology-grounded approaches lead in quality
```

## Key Takeaways

- Graph RAG adds entity extraction, graph construction, community detection, and multi-mode search on top of standard RAG
- It excels at cross-document reasoning, multi-hop questions, and global summarization
- The cost is higher but has been dramatically reduced by LightRAG and other innovations
- Hybrid (vector + graph) is the production standard in 2026
- Choose your approach based on query complexity, corpus characteristics, and budget
