# Streamlit KG Explorer

Interactive web app for exploring any knowledge graph. Load from GraphML, JSON, or connect to Neo4j. Search entities, click to explore neighbors, view relationships, and see graph statistics.

## Features

- **Explorer**: Search for entities, view details and neighbors, click to navigate
- **Graph View**: Interactive pyvis visualization embedded in the browser
- **Statistics**: Node/edge counts, type distributions, degree histograms, centrality rankings, community detection
- **Query**: Ask natural language questions answered using graph context and an LLM

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Data Sources

- **Upload GraphML**: Load any `.graphml` file exported from NetworkX
- **Upload JSON**: Load a JSON file with `{"nodes": [...], "edges": [...]}`  format
- **Neo4j**: Connect to a running Neo4j instance (requires URI, username, password)

## Usage

1. Select a data source in the sidebar
2. Upload a file or enter Neo4j credentials
3. Use the tabs to explore your graph:
   - **Explorer** — Search and browse entities interactively
   - **Graph View** — See the full graph visualization
   - **Statistics** — Analyze graph structure and properties
   - **Query** — Ask questions about your knowledge graph

## Requirements

See `requirements.txt`. Uses the shared LLM layer from `shared/llm_clients.py` for the Q&A feature.
