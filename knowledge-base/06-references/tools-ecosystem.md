# Graph RAG Tools Ecosystem

A comparison of the major tools and frameworks for building Knowledge Graphs and Graph RAG systems.

## Graph RAG Frameworks

| Tool | Type | Cost | Quality | Complexity | Best For |
|------|------|------|---------|------------|----------|
| Microsoft GraphRAG | Full pipeline | High | Highest | Medium | Comprehensive graph RAG with community detection |
| LightRAG | Lightweight pipeline | Low | High | Low | Cost-effective graph RAG |
| LlamaIndex KG Index | Framework | Medium | High | Medium | Integration with existing LlamaIndex pipelines |
| LangChain + Neo4j | Framework | Medium | High | Medium-High | Custom graph RAG with agent capabilities |

### Microsoft GraphRAG (`graphrag`)
- **GitHub**: github.com/microsoft/graphrag
- **Language**: Python
- **What it does**: End-to-end pipeline: entity/relationship extraction → Leiden community detection → community summarization → local/global/DRIFT search
- **Strengths**: Highest quality answers for global queries, well-documented, active development
- **Weaknesses**: Expensive indexing, complex configuration, slower queries
- **When to use**: When answer quality is paramount and budget allows

### LightRAG (`lightrag-hku`)
- **GitHub**: github.com/HKUDS/LightRAG
- **Language**: Python
- **What it does**: Simplified graph RAG — entity extraction, flat graph, dual-mode retrieval (graph + vector)
- **Strengths**: 100x cheaper than GraphRAG, fast indexing (~3 min for 500 pages), simple API
- **Weaknesses**: No community detection, weaker on global/thematic queries
- **When to use**: Cost-sensitive deployments, medium-complexity queries

## Graph Databases

| Database | Type | Hosting | Query Language | Vector Support | Best For |
|----------|------|---------|---------------|----------------|----------|
| Neo4j | Property graph | Self-hosted / AuraDB | Cypher | Native (5.x+) | Production KGs, complex traversals |
| Amazon Neptune | Property graph + RDF | AWS managed | Gremlin / SPARQL | Limited | AWS-native deployments |
| Memgraph | In-memory property graph | Self-hosted / Cloud | Cypher-compatible | Via extensions | Real-time analytics |
| NetworkX | In-memory (Python) | N/A (library) | Python API | N/A | Prototyping, analysis |

### Neo4j
- **Best for**: Production knowledge graphs
- **Key features**: ACID transactions, native vector index, APOC plugin ecosystem, visualization tools
- **Setup**: Docker or Neo4j AuraDB (managed)
- **Cost**: Free (Community Edition), paid for Enterprise features

### NetworkX
- **Best for**: Prototyping and analysis
- **Key features**: Pure Python, extensive algorithms, easy visualization with matplotlib
- **Limitations**: In-memory only, no persistence, not concurrent

## LLM Frameworks with Graph Support

### LangChain / LangGraph
- **Graph features**: Neo4j integration, graph-based RAG chains, Cypher query generation
- **Agent support**: LangGraph for stateful multi-step agent workflows
- **When to use**: Building custom agentic graph RAG systems

### LlamaIndex
- **Graph features**: PropertyGraphIndex, KnowledgeGraphIndex, graph query engine
- **Strengths**: 40% faster document retrieval, built-in re-ranking, hierarchical chunking
- **When to use**: Document-heavy applications, integration with existing LlamaIndex code

## Temporal KG Tools

### Graphiti (by Zep)
- **GitHub**: github.com/getzep/graphiti
- **What it does**: Temporal context graph engine — each fact has a validity window, hybrid search without LLM at retrieval time
- **Key metrics**: P95 retrieval latency 300ms, outperforms MemGPT on Deep Memory Retrieval
- **When to use**: AI agent memory, systems where facts change over time

## Embedding Models

| Model | Provider | Dimensions | Best For |
|-------|----------|-----------|----------|
| text-embedding-3-small | OpenAI | 1536 | General purpose, cost-effective |
| text-embedding-3-large | OpenAI | 3072 | Highest quality, more expensive |
| embed-v3 | Cohere | 1024 | Multilingual support |
| BGE-M3 | BAAI | 1024 | Open-source, multilingual |

## Visualization Tools

| Tool | Type | Best For |
|------|------|----------|
| pyvis | Interactive HTML | Quick graph exploration |
| Gephi | Desktop app | Publication-quality visualizations |
| Neo4j Browser | Web UI | Exploring Neo4j databases |
| NetworkX + matplotlib | Static plots | Notebooks, reports |
| yFiles | Commercial | Enterprise dashboards |

## Decision Matrix

**Choose your stack based on your needs:**

| Need | Recommended Stack |
|------|------------------|
| Quick prototype | NetworkX + OpenAI + matplotlib |
| Cost-effective Graph RAG | LightRAG + OpenAI |
| Highest quality answers | Microsoft GraphRAG + Neo4j |
| Custom agent system | LangGraph + Neo4j + OpenAI |
| Temporal/agent memory | Graphiti + Neo4j |
| Enterprise deployment | Neo4j AuraDB + LangGraph + monitoring |
