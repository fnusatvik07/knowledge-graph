# Key Papers in Knowledge Graphs and Graph RAG

An annotated bibliography of the most important papers, organized by topic.

## Foundational Graph RAG

### From Local to Global: A GraphRAG Approach to Query-Focused Summarization
- **Authors**: Darren Edge, Ha Trinh, et al. (Microsoft Research)
- **Date**: April 2024
- **Link**: arxiv.org/abs/2404.16130
- **Key contribution**: Introduced GraphRAG — building a knowledge graph from documents, applying Leiden community detection, generating hierarchical community summaries, and using map-reduce for global query answering.
- **Results**: 26% improvement in answer comprehensiveness, 57% improvement in diversity over vector-only RAG.
- **Why it matters**: This paper launched the entire Graph RAG field and remains the reference architecture.

### GraphRAG: Improving Global Search via Dynamic Community Selection
- **Authors**: Microsoft Research
- **Date**: January 2025
- **Key contribution**: Reduced token usage by 79% during global search by dynamically selecting relevant communities instead of processing all communities at a given level.
- **Why it matters**: Made GraphRAG practical for cost-sensitive production deployments.

### DRIFT Search: Combining Global and Local Search Methods
- **Authors**: Microsoft Research
- **Date**: 2024
- **Key contribution**: Dynamic Reasoning and Inference with Flexible Traversal — a three-phase search (primer → follow-up → output) that combines global community insights with local entity-level detail.
- **Why it matters**: Bridges the gap between local and global search modes, providing the best of both.

## Alternative Approaches

### LightRAG: Simple and Fast Retrieval-Augmented Generation
- **Authors**: Zirui Guo, Hong-Ning Dai (Hong Kong University)
- **Date**: Late 2024
- **Key contribution**: A stripped-down Graph RAG — simpler extraction, flat graph (no community detection), dual-mode retrieval (graph + vector). 1/100th the cost of GraphRAG.
- **Why it matters**: Made graph-enhanced RAG accessible to teams without large budgets.

### LinearRAG: Relation-Free Graph Construction
- **Date**: October 2025 (accepted at ICLR 2026)
- **Key contribution**: Graph construction without explicit relationship extraction — using only entity co-occurrence patterns.
- **Why it matters**: Further reduces KG construction cost while maintaining retrieval quality.

### HybridRAG: Integrating Knowledge Graphs and Vector Retrieval
- **Date**: 2024
- **Key contribution**: Formal framework for combining vector-based and graph-based retrieval, showing 85%+ accuracy vs 70% for vector-only.
- **Why it matters**: Validated the hybrid approach that became the 2026 production standard.

## Knowledge Graph Construction

### LLM-Empowered Knowledge Graph Construction: A Survey
- **Date**: October 2025
- **Key contribution**: Comprehensive survey of LLM-based approaches to KG construction, covering entity extraction, relationship extraction, ontology learning, and evaluation.
- **Why it matters**: Definitive reference for understanding the KG construction landscape.

### Ontology Learning and KG Construction Impact on RAG
- **Venue**: NeurIPS 2025
- **Key contribution**: Demonstrated that ontology-grounded KG construction achieves the highest RAG performance with minimal hallucination compared to schema-free approaches.
- **Why it matters**: Provides empirical evidence for investing in ontology design.

## Surveys and Benchmarks

### Graph Retrieval-Augmented Generation: A Survey
- **Venue**: ACM Transactions on Information Systems
- **Key contribution**: First comprehensive systematic survey, categorizing Graph RAG into G-Indexing, G-Retrieval, and G-Generation stages.
- **Why it matters**: The definitive academic survey of the field.

### GraphRAG Benchmark
- **Date**: June 2025
- **Key contribution**: Standardized benchmark for evaluating different Graph RAG models across query types, corpus sizes, and domains.
- **Why it matters**: Enables apples-to-apples comparison of Graph RAG approaches.

## Temporal and Agentic KGs

### Graphiti: Building Real-Time Knowledge Graphs for AI Agents
- **Authors**: Zep team
- **Date**: January 2025
- **Link**: arxiv.org/abs/2501.13956
- **Key contribution**: Temporal context graph engine with validity windows for facts, hybrid search (semantic + BM25 + graph traversal), no LLM calls during retrieval. P95 retrieval latency of 300ms.
- **Why it matters**: Standard for AI agent memory systems with temporal awareness.

## Biomedical KGs

### MINERVA: Scaling Biomedical KG Retrieval
- **Date**: 2026
- **Key contribution**: Distilled 129,719 biomedical publications into a KG with 66,444 validated relations covering 2,941 microbes and 3,299 diseases.
- **Why it matters**: Demonstrates KG construction at scale in a critical domain.

### AutoBioKG: Context-Aware Biomedical KG Construction
- **Date**: 2026
- **Key contribution**: End-to-end framework for constructing KGs that encode environmental conditions and entity attributes alongside core relationships.
- **Why it matters**: Shows that context-aware KGs capture more nuanced knowledge.

## Reading Order Suggestion

1. Start with the Microsoft GraphRAG paper (foundational)
2. Read the Graph RAG Survey for a comprehensive overview
3. Read LightRAG for the cost-effective alternative
4. Read HybridRAG for production architecture patterns
5. Read Graphiti for temporal KG concepts
6. Explore domain-specific papers based on your use case
