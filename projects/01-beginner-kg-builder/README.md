# Project 1: Simple Knowledge Graph Builder (Beginner)

Build a knowledge graph from text documents using LLM-based extraction, visualize it, and answer questions by traversing the graph.

## What You'll Learn

- LLM-based entity and relationship extraction with structured output
- Building graphs with NetworkX
- Graph visualization with matplotlib and pyvis
- Simple question answering via graph traversal

## Setup

```bash
# From the repo root
uv pip install -e ".[beginner]"
```

## Run

Execute the scripts in order:

```bash
cd projects/01-beginner-kg-builder

# Step 1: Extract entities from sample articles
python src/01_extract_entities.py

# Step 2: Extract relationships between entities
python src/02_extract_relationships.py

# Step 3: Build the NetworkX graph
python src/03_build_graph.py

# Step 4: Visualize the graph
python src/04_visualize.py

# Step 5: Ask questions about the graph
python src/05_query_graph.py
```

Each script builds on the output of the previous one. Results are saved to `output/`.

## Data

Three sample articles about AI, graph databases, and RAG are included in `data/sample_articles/`. You can add your own `.txt` files to build a knowledge graph from any topic.

## Project Structure

```
01-beginner-kg-builder/
├── data/sample_articles/     # Input documents
├── src/
│   ├── 01_extract_entities.py
│   ├── 02_extract_relationships.py
│   ├── 03_build_graph.py
│   ├── 04_visualize.py
│   ├── 05_query_graph.py
│   └── prompts.py            # All LLM prompts
├── notebooks/
│   └── walkthrough.ipynb     # Interactive walkthrough
└── output/                   # Generated outputs
```
