# Project 8: Real-Time Streaming Knowledge Graph

Build a knowledge graph that ingests data in real-time from streaming data sources. This project demonstrates how to watch a directory for new files, extract entities and relationships on arrival using LLM-powered extraction, and update a Neo4j graph database with temporal metadata.

## Architecture

1. **File Watcher** — Monitors a directory for new files using `watchfiles`
2. **Stream Processor** — Extracts entities/relationships from each new document via LLM, upserts them into Neo4j with timestamps
3. **WebSocket Server** — Broadcasts graph change events to connected clients in real-time
4. **Live Dashboard** — Terminal-based UI showing graph statistics as data streams in
5. **Temporal Queries** — Query the graph with time awareness: what was true at time T, what changed, what was invalidated

## Key Concepts

- **Entity Resolution**: Duplicate entities are merged using name normalization and LLM-assisted matching
- **Temporal Metadata**: Every node and edge carries `valid_from` and `valid_until` timestamps
- **Fact Invalidation**: When new information contradicts existing facts, old facts are marked as invalidated with a reason
- **Live Updates**: WebSocket-based push notifications for real-time graph change streaming

## Prerequisites

- Docker (for Neo4j)
- Python 3.11+
- Dependencies: `neo4j`, `watchfiles`, `websockets`, `langchain`, `langchain-openai`

## Quick Start

```bash
# 1. Start Neo4j
docker compose up -d

# 2. Initialize the graph schema
python src/01_neo4j_setup.py

# 3. Run the full demo (starts processor, drops sample files, shows results)
python src/06_run_demo.py
```

## Manual Usage

```bash
# Start the stream processor watching a directory
python src/02_stream_processor.py --watch-dir ./data/watch

# In another terminal, start the WebSocket server
python src/03_websocket_server.py

# In another terminal, start the live dashboard
python src/04_live_dashboard.py

# Drop files into ./data/watch/ and watch the graph grow!
cp data/stream_sample_01.txt data/watch/

# Run temporal queries
python src/05_temporal_queries.py
```

## File Structure

```
08-realtime-streaming-kg/
├── README.md
├── docker-compose.yml
├── data/
│   ├── stream_sample_01.txt
│   ├── stream_sample_02.txt
│   └── stream_sample_03.txt
└── src/
    ├── __init__.py
    ├── 01_neo4j_setup.py
    ├── 02_stream_processor.py
    ├── 03_websocket_server.py
    ├── 04_live_dashboard.py
    ├── 05_temporal_queries.py
    └── 06_run_demo.py
```
