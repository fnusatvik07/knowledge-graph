# Project 4: Multi-DB Knowledge Graph Explorer

Build the **same knowledge graph** across four different graph databases and compare their query languages, performance characteristics, and developer experience.

## Databases Compared

| Database | Query Language | Protocol | Port(s) |
|----------|---------------|----------|---------|
| **Neo4j** | Cypher | Bolt | 7474 (HTTP), 7687 (Bolt) |
| **ArangoDB** | AQL (Arango Query Language) | HTTP | 8529 |
| **Memgraph** | Cypher (compatible) | Bolt | 7688 (Bolt), 3000 (Lab UI) |
| **FalkorDB** | OpenCypher | Redis protocol | 6379 |

## What We Build

A knowledge graph about **tech companies, AI models, and their relationships** — the same domain used across the Knowledge Graphs repo. The identical dataset (`data/entities_and_relations.json`) is loaded into all four databases so we can make apples-to-apples comparisons.

## Setup

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- Required packages: `neo4j`, `python-arango`, `gqlalchemy`, `falkordb`, `langchain-neo4j`, `tabulate`

### 1. Start All Databases

```bash
docker compose up -d
```

This launches all four databases. Wait ~30 seconds for them to initialize.

### 2. Install Python Dependencies

```bash
pip install neo4j python-arango gqlalchemy falkordb tabulate langchain-neo4j langchain-community
```

### 3. Run the Scripts

```bash
# Load data into each database
python src/01_load_neo4j.py
python src/02_load_arangodb.py
python src/03_load_memgraph.py
python src/04_load_falkordb.py

# Compare queries across all databases
python src/05_query_comparison.py

# LangChain graph integrations
python src/06_langchain_integration.py
```

### 4. Access Database UIs

- **Neo4j Browser**: http://localhost:7474
- **ArangoDB Web UI**: http://localhost:8529
- **Memgraph Lab**: http://localhost:3000

## Key Takeaways

- **Neo4j**: Most mature, best tooling, largest community. Cypher is the gold standard.
- **ArangoDB**: Multi-model (documents + graphs + search). AQL is SQL-like and powerful.
- **Memgraph**: Cypher-compatible, optimized for real-time streaming and in-memory performance.
- **FalkorDB**: Redis-based, extremely fast for simple traversals. OpenCypher subset.

## Cleanup

```bash
docker compose down -v
```
