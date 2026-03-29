# Project 3: Advanced Multi-Source Knowledge Graph Agent

An advanced LangGraph-powered agent system that builds and queries knowledge graphs
from multiple document sources, with temporal metadata tracking, hybrid retrieval
(vector + graph), and web search enrichment.

## Architecture

```
                         +------------------+
                         |   Orchestrator   |
                         | (LangGraph FSM)  |
                         +--------+---------+
                                  |
                    +-------------+-------------+
                    |                           |
              INGEST MODE                 QUERY MODE
                    |                           |
         +----------+----------+      +---------+---------+
         |                     |      |                   |
  +------+-------+    +-------+--+   |  +---------------+--+
  | Graph Builder |    | Web Search|   |  | Research Agent   |
  |    Agent      |    |   Agent   |   |  | (Multi-hop QA)   |
  +------+--------+    +----------+   |  +--------+---------+
         |                            |           |
         v                            |  +--------+---------+
  +--------------+                    |  | Hybrid Retriever  |
  | Entity/Rel   |                    |  +--+----------+----+
  | Extraction   |                    |     |          |
  | (LLM)        |                    |     v          v
  +--------------+                    |  Vector     Graph
         |                            |  Search     Traversal
         v                            |     |          |
  +------+--------+                   |     +----+-----+
  | Neo4j Store   |<------------------+          |
  | (Graph + Vec) |                         +----+-----+
  | + Temporal    |                         | Reranker |
  +---------------+                         | (LLM)   |
                                            +----------+
```

## Components

| Component | Description |
|---|---|
| **Neo4j Store** | Graph database with vector index support for hybrid storage |
| **Temporal Layer** | Tracks valid_from/valid_to, source provenance, confidence scores |
| **Graph Builder Agent** | Extracts entities and relationships from documents via LLM |
| **Research Agent** | Decomposes complex questions, performs multi-hop KG reasoning |
| **Web Search Agent** | Enriches KG with live web data via Tavily |
| **Hybrid Retriever** | Fuses vector similarity and graph traversal results |
| **Orchestrator** | LangGraph StateGraph coordinating INGEST and QUERY workflows |

## Setup

### 1. Start Neo4j

```bash
cd projects/03-advanced-kg-agent
docker-compose up -d
```

This starts Neo4j with:
- Browser UI at http://localhost:7474
- Bolt protocol at bolt://localhost:7687
- APOC plugin enabled
- Vector index support

Default credentials: `neo4j` / `password`

### 2. Environment Variables

Ensure your `.env` file at the repo root contains:

```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

### 3. Install Dependencies

```bash
pip install neo4j openai langgraph langchain-core tavily-python pydantic python-dotenv
```

## Usage

### Run the Full Pipeline

```bash
python projects/03-advanced-kg-agent/src/agents/orchestrator.py
```

This will:
1. **INGEST** the sample documents into the knowledge graph
2. **QUERY** the graph with example multi-hop questions

### Individual Components

```python
# Build graph from documents
from src.agents.graph_builder_agent import GraphBuilderAgent
agent = GraphBuilderAgent(neo4j_store)
agent.ingest(documents)

# Query with hybrid retrieval
from src.retrieval.hybrid_retriever import HybridRetriever
retriever = HybridRetriever(neo4j_store)
results = retriever.retrieve("What AI research does Google lead?")

# Multi-hop research
from src.agents.research_agent import ResearchAgent
agent = ResearchAgent(neo4j_store)
answer = agent.answer("How are transformer models connected to Google's research?")
```

## Ontology

The knowledge graph uses a typed schema:

- **Entity Types**: PERSON, ORGANIZATION, TECHNOLOGY, CONCEPT, PAPER, EVENT
- **Relationship Types**: WORKS_AT, FOUNDED, DEVELOPED, PUBLISHED, USES, RELATED_TO, PRESENTED_AT, FUNDED_BY, COLLABORATED_WITH, AUTHORED
- **Temporal Properties**: valid_from, valid_to, source, confidence

## Key Design Decisions

- **Reciprocal Rank Fusion** merges vector and graph results without requiring score normalization
- **Temporal metadata** enables point-in-time queries and fact invalidation (inspired by Graphiti)
- **LLM reranking** as a final pass improves precision on complex queries
- **Structured extraction** uses Pydantic models to enforce ontology compliance
