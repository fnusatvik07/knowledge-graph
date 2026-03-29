# Project 14: Personal Knowledge Graph ("Second Brain")

Turn your markdown notes into a connected knowledge graph. Like Obsidian but with a real graph backend, LLM-powered concept extraction, and semantic search.

## What It Does

- **Parse markdown notes** with [[wiki-style links]], #tags, and headings
- **Auto-extract entities** and concepts from your notes using LLMs
- **Discover hidden connections** between notes that share concepts but don't explicitly link
- **Semantic search** across your entire knowledge base with natural language queries
- **Visualize your brain** as an interactive knowledge graph

## Pipeline

| Step | Script | Description |
|------|--------|-------------|
| 1 | `src/01_parse_notes.py` | Parse markdown files, extract links, tags, and metadata |
| 2 | `src/02_extract_concepts.py` | Use LLM to find implicit concepts and entities in each note |
| 3 | `src/03_build_personal_kg.py` | Build a NetworkX knowledge graph with notes and concepts |
| 4 | `src/04_discover_connections.py` | Find hidden connections, suggest missing links, identify knowledge gaps |
| 5 | `src/05_semantic_search.py` | Embed notes and search with natural language queries |
| 6 | `src/06_visualize_brain.py` | Interactive pyvis visualization of your personal KG |

## Quick Start

```bash
# 1. Add your markdown notes to data/notes/ (samples provided)
# 2. Parse notes
python src/01_parse_notes.py

# 3. Extract concepts with LLM
python src/02_extract_concepts.py

# 4. Build the knowledge graph
python src/03_build_personal_kg.py

# 5. Find hidden connections
python src/04_discover_connections.py

# 6. Search your knowledge base
python src/05_semantic_search.py --query "What do I know about optimization?"

# 7. Visualize
python src/06_visualize_brain.py
```

## Sample Notes

The `data/notes/` directory contains sample notes about ML/AI topics with wiki-style links to demonstrate the system. Replace them with your own notes.

## Requirements

- Python 3.11+
- networkx, pyvis, numpy
- LangChain + an LLM provider (OpenAI, Anthropic, or Ollama)
