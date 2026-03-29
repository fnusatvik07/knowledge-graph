# Knowledge Graph & Graph RAG — Hands-On Examples

Practical examples and Jupyter notebooks for building Knowledge Graphs and Graph RAG systems.

## Setup

```bash
# From repo root
uv sync                              # install Python deps
cd workshop && docker compose up -d   # start Neo4j
# Neo4j Browser: http://localhost:7474 (neo4j / workshop2024)
```

## Structure

```
workshop/
├── notebooks/              # Jupyter notebooks (main learning material)
├── scripts/                # Standalone Python scripts
├── cypher-tutorial/        # Cypher query examples (copy-paste into Neo4j)
├── document-to-kg/         # Build KG from a research paper (step-by-step scripts)
├── csv-to-kg/              # Build KG from CSV data (step-by-step scripts)
├── data/                   # Shared data files
├── docker-compose.yml      # Neo4j container
└── requirements.txt        # Python dependencies
```

## Notebooks

| Notebook | What It Covers |
|----------|---------------|
| `company_kg_workshop.ipynb` | Build KG from text — people, companies, hobbies, relationships |
| `kg_vs_rag_showdown.ipynb` | Classic RAG vs Graph RAG — honest practical comparison |
| `hybrid_rag.ipynb` | Hybrid RAG — combines graph structure + original text |
| `lightrag_explained.ipynb` | LightRAG — automated Graph RAG in 5 lines of code |
| `csv_sqlite_to_kg.ipynb` | CSV and SQLite → Knowledge Graph |
| `large_document_chunking.ipynb` | Chunk big documents, batch extract, merge entities |
| `graph_rag_workshop.ipynb` | End-to-end Graph RAG with research paper |

## Cypher Tutorial

Open Neo4j Browser at http://localhost:7474, then copy-paste queries from `cypher-tutorial/` files:

| File | What You Learn |
|------|---------------|
| `01_create_nodes.cypher` | CREATE nodes with labels and properties |
| `02_create_relationships.cypher` | Connect nodes with typed relationships |
| `03_read_queries.cypher` | MATCH, RETURN, WHERE — find patterns |
| `04_filter_and_aggregate.cypher` | Filtering, counting, ordering |
| `05_multi_hop.cypher` | Multi-hop traversal — shortest path, 2-hop queries |
| `06_update_and_delete.cypher` | SET, DELETE, DETACH DELETE |
| `07_now_with_llm.cypher` | Bridge to LLM-based extraction |

## Step-by-Step Scripts

**Document → KG** (`document-to-kg/`): Extract entities from a research paper, load into Neo4j, query with Graph RAG.

**CSV → KG** (`csv-to-kg/`): Transform AI companies CSV into a connected graph, run Cypher queries.
