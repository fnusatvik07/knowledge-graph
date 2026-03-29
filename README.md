<p align="center">
  <img src="https://img.shields.io/badge/Knowledge%20Graphs-Expert%20Guide-4ECDC4?style=for-the-badge&logo=graphql&logoColor=white" alt="Knowledge Graphs" />
  <img src="https://img.shields.io/badge/Graph%20RAG-Production%20Ready-6C5CE7?style=for-the-badge&logo=neo4j&logoColor=white" alt="Graph RAG" />
  <img src="https://img.shields.io/badge/Projects-17-F39C12?style=for-the-badge" alt="17 Projects" />
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
</p>

<h1 align="center">Knowledge Graphs & Graph RAG</h1>

<p align="center">
  <strong>The comprehensive repository for mastering Knowledge Graphs — from foundational concepts to production-grade Graph RAG systems.</strong>
</p>

<p align="center">
  16 learning sections · 17 hands-on projects · 7 Jupyter notebooks · 5 graph databases · Interactive React website
</p>

---

## Why This Repository?

Large Language Models are powerful but limited — they hallucinate, lose context across documents, and can't reason about relationships between entities. **Knowledge Graphs** solve this by structuring information as entities and relationships that can be traversed, queried, and reasoned over.

**Graph RAG** (Retrieval Augmented Generation) combines the structured reasoning of knowledge graphs with the generative power of LLMs — delivering answers that are both accurate and relationship-aware.

This repository covers everything: from building your first knowledge graph to deploying production Hybrid RAG systems.

---

## Repository Structure

```
knowledge-graph/
│
├── knowledge-base/              16 learning sections (theory + code examples)
│   ├── 01-foundations/          KG basics, graph terminology, why Graph RAG
│   ├── 02-kg-construction/      Entity & relationship extraction, ontology design
│   ├── 03-graph-storage/        Neo4j, NetworkX, Cypher queries
│   ├── 04-graph-rag/            GraphRAG, LightRAG, community detection, search modes
│   ├── 05-advanced/             Hybrid retrieval, temporal KGs, multi-hop reasoning
│   ├── 06-references/           Papers, tools ecosystem, glossary
│   ├── 07-mcp-servers/          Neo4j MCP, Graphiti, TigerGraph, FalkorDB
│   ├── 08-query-languages/      Cypher vs Gremlin vs AQL vs SPARQL
│   ├── 09-evaluation/           RAGAS metrics, graph quality, LLM-as-judge
│   ├── 10-datasets/             Wikidata, DBpedia, FB15k-237, HotPotQA
│   ├── 11-multimodal/           Beyond text, KG versioning, CI/CD
│   ├── 12-embeddings/           TransE, RotatE, ComplEx, PyKEEN
│   ├── 13-gnn/                  GCN, GraphSAGE, GAT, node classification
│   ├── 14-rdf/                  RDF fundamentals, SPARQL, rdflib
│   ├── 15-algorithms/           PageRank, centrality, shortest paths, motifs
│   └── 16-agentic/              Agent memory, multi-agent KG, autonomous construction
│
├── projects/                    17 hands-on projects
│   ├── 01-beginner-kg-builder/
│   ├── 02-intermediate-graph-rag/
│   ├── 03-advanced-kg-agent/
│   ├── ...
│   └── 17-autonomous-kg-agent/
│
├── workshop/                    Practical examples & notebooks
│   ├── notebooks/               7 Jupyter notebooks
│   ├── cypher-tutorial/         Cypher syntax guide + live demo
│   ├── document-to-kg/          Text → Knowledge Graph (step-by-step)
│   ├── csv-to-kg/               CSV → Knowledge Graph (step-by-step)
│   └── data/                    Shared datasets
│
└── website/                     Interactive React learning site
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker (for Neo4j)
- OpenAI API key

### Setup

```bash
git clone https://github.com/fnusatvik07/knowledge-graph.git
cd knowledge-graph

# Install all dependencies
cp .env.example .env          # add your OpenAI API key
uv sync

# Start Neo4j
cd workshop && docker compose up -d
# Neo4j Browser: http://localhost:7474 (neo4j / workshop2024)
```

### Run a Notebook

```bash
# Open in VS Code — select kernel "knowledgegraphs (Python 3.11)"
code workshop/notebooks/company_kg_workshop.ipynb
```

### Run the Website

```bash
cd website && npm install && npm run dev
# Open http://localhost:5173
```

---

## Learning Sections

<table>
<tr><th>Section</th><th>Topics</th><th>Key Takeaway</th></tr>
<tr><td><strong>01 Foundations</strong></td><td>What is a KG, graph terminology, traditional RAG limitations</td><td>Why graphs beat vector search for relationship questions</td></tr>
<tr><td><strong>02 KG Construction</strong></td><td>Entity extraction, relationship extraction, ontology design, LLM patterns</td><td>How to build a KG from any text using LLMs</td></tr>
<tr><td><strong>03 Graph Storage</strong></td><td>Neo4j, NetworkX, Cypher query language</td><td>Choose the right graph database for your use case</td></tr>
<tr><td><strong>04 Graph RAG</strong></td><td>Microsoft GraphRAG, LightRAG, Leiden communities, local/global/DRIFT search</td><td>How graph-based retrieval outperforms vector-only RAG</td></tr>
<tr><td><strong>05 Advanced</strong></td><td>Hybrid retrieval, temporal KGs, multi-hop reasoning, agentic RAG</td><td>Production patterns: combine vector + graph retrieval</td></tr>
<tr><td><strong>07 MCP Servers</strong></td><td>Neo4j MCP, TigerGraph, Memgraph, ArangoDB, Graphiti</td><td>Build KGs through AI assistants with zero code</td></tr>
<tr><td><strong>08 Query Languages</strong></td><td>Cypher vs Gremlin vs AQL vs GSQL vs SPARQL</td><td>Same query in 5 languages — side by side</td></tr>
<tr><td><strong>12 KG Embeddings</strong></td><td>TransE, RotatE, ComplEx math + PyKEEN tutorial</td><td>Predict missing links in a knowledge graph</td></tr>
<tr><td><strong>13 GNNs</strong></td><td>GCN, GraphSAGE, GAT, node classification</td><td>Classify entities using graph structure</td></tr>
<tr><td><strong>14 RDF</strong></td><td>RDF triples, SPARQL, rdflib, Wikidata, DBpedia</td><td>Query the world's largest public knowledge graphs</td></tr>
<tr><td><strong>15 Graph Algorithms</strong></td><td>PageRank, centrality, shortest paths, community detection</td><td>Find the most important entities and hidden patterns</td></tr>
<tr><td><strong>16 Agentic KG</strong></td><td>Agent memory, multi-agent systems, autonomous construction</td><td>Agents that build and maintain their own knowledge graphs</td></tr>
</table>

---

## Projects

### Beginner

| # | Project | Stack |
|---|---------|-------|
| 1 | **Simple KG Builder** — Extract entities from text, build graph, visualize, Q&A | LangChain + NetworkX + pyvis |
| 14 | **Personal Knowledge Graph** — "Second brain" from markdown notes | NetworkX + LLM embeddings |

### Intermediate

| # | Project | Stack |
|---|---------|-------|
| 2 | **Graph RAG Pipeline** — GraphRAG + LightRAG + vector RAG comparison | graphrag + lightrag + ChromaDB |
| 4 | **Multi-DB Explorer** — Same KG across Neo4j, ArangoDB, Memgraph, FalkorDB | 4 graph databases |
| 5 | **MCP KG Studio** — Build KGs through MCP servers | Neo4j MCP + Graphiti |
| 7 | **Code Knowledge Graph** — Parse codebases, build call graphs | tree-sitter + NetworkX |
| 11 | **Fraud Detection** — Transaction networks, cycle detection, anomaly scoring | NetworkX + PageRank |
| 12 | **Recommendation Engine** — Movie KG with collaborative + content-based filtering | NetworkX + LLM embeddings |
| 13 | **RDF & SPARQL Explorer** — Query Wikidata and DBpedia | rdflib + SPARQLWrapper |

### Advanced

| # | Project | Stack |
|---|---------|-------|
| 3 | **Multi-Source KG Agent** — Agentic system with temporal KG and hybrid retrieval | LangGraph + Neo4j + Tavily |
| 6 | **Biomedical KG** — Drug-disease-gene relationships, drug repurposing | BioPython + Neo4j |
| 8 | **Real-Time Streaming KG** — Live updates with WebSocket dashboard | watchfiles + websockets + Neo4j |
| 10 | **KG Embeddings** — Train TransE/RotatE/ComplEx, predict missing links | PyKEEN |
| 15 | **GNN Node Classification** — Train GCN/GAT/GraphSAGE on a KG | PyTorch Geometric |
| 16 | **Multi-Agent KG System** — 4 agents collaborating through shared KG | LangGraph + Neo4j |
| 17 | **Autonomous KG Agent** — Self-improving KG with gap detection | LangGraph + Neo4j |

---

## Notebooks

| Notebook | What You'll Build |
|----------|------------------|
| `company_kg_workshop.ipynb` | Knowledge graph from a company profile — people, hobbies, careers |
| `kg_vs_rag_showdown.ipynb` | Honest comparison: Classic RAG vs Graph RAG on the same document |
| `hybrid_rag.ipynb` | Hybrid RAG — graph structure + original text (the production answer) |
| `lightrag_explained.ipynb` | LightRAG — automated Graph RAG in 5 lines of code |
| `csv_sqlite_to_kg.ipynb` | Transform structured data (CSV, SQLite) into knowledge graphs |
| `large_document_chunking.ipynb` | Chunk big documents, extract per chunk, merge into one KG |
| `graph_rag_workshop.ipynb` | End-to-end Graph RAG with a real research paper |

---

## Graph RAG Approaches Compared

| Approach | Cost | Quality | Incremental Updates | Best For |
|----------|------|---------|--------------------|---------|
| **Classic RAG** | Lowest | Good for facts | Yes | Simple fact lookup |
| **LightRAG** | Low (~$0.50) | 70-90% | Yes (seamless) | Cost-effective Graph RAG |
| **Microsoft GraphRAG** | Very High (~$33K+) | Highest | No (full rebuild) | Global corpus-wide queries |
| **Hybrid RAG** | Moderate | Best overall | Yes | Production systems |

> **Our testing showed**: Classic RAG found 1 out of 5 people who left a company. Graph RAG found 4 out of 5. Hybrid RAG found all 5 with exact details. [See the comparison notebook →](workshop/notebooks/kg_vs_rag_showdown.ipynb)

---

## Tech Stack

<table>
<tr><td><strong>LLM Framework</strong></td><td>LangChain (OpenAI, Anthropic, Ollama)</td></tr>
<tr><td><strong>Agent Framework</strong></td><td>LangGraph</td></tr>
<tr><td><strong>Graph Databases</strong></td><td>Neo4j, ArangoDB, Memgraph, FalkorDB</td></tr>
<tr><td><strong>RDF</strong></td><td>rdflib, SPARQLWrapper, Wikidata, DBpedia</td></tr>
<tr><td><strong>KG Embeddings</strong></td><td>PyKEEN (TransE, RotatE, ComplEx)</td></tr>
<tr><td><strong>GNNs</strong></td><td>PyTorch Geometric (GCN, GAT, GraphSAGE)</td></tr>
<tr><td><strong>Graph RAG</strong></td><td>LightRAG, Microsoft GraphRAG</td></tr>
<tr><td><strong>Vector Store</strong></td><td>ChromaDB, Neo4j Vector Index</td></tr>
<tr><td><strong>Visualization</strong></td><td>pyvis, matplotlib, Streamlit</td></tr>
<tr><td><strong>Web</strong></td><td>React, Tailwind CSS, Framer Motion</td></tr>
</table>

---

## Graph Database Coverage

```
Neo4j ─────────── Projects 3, 4, 6, 8, 16, 17  (Cypher)
ArangoDB ───────── Project 4                     (AQL)
Memgraph ───────── Project 4                     (Cypher)
FalkorDB ───────── Project 4                     (OpenCypher)
NetworkX ───────── Projects 1, 2, 7, 9, 11-15   (Python API)
RDF/rdflib ─────── Project 13                    (SPARQL)
```

---

## License

MIT

---

<p align="center">
  <strong>Star this repo if you find it useful!</strong>
</p>
