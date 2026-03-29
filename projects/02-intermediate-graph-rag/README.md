# Project 2: Intermediate Graph RAG Pipeline

Compare **Microsoft GraphRAG**, **LightRAG**, and a traditional **Vector RAG** baseline
on the same corpus to understand when graph-based retrieval outperforms plain
vector search.

## What You Will Learn

- How to index a document corpus with Microsoft GraphRAG and LightRAG
- Community detection with the Leiden algorithm on knowledge graphs
- Local search (entity-centric) vs. global search (community-summary) vs. DRIFT search
- Building a ChromaDB-based vector RAG baseline
- Systematic comparison of retrieval approaches using LLM-as-judge evaluation

## Prerequisites

```bash
pip install graphrag lightrag-hku chromadb networkx leidenalg igraph
pip install openai python-dotenv tabulate
```

Make sure your `.env` file at the repository root contains:

```
OPENAI_API_KEY=sk-...
```

## Project Structure

```
02-intermediate-graph-rag/
├── README.md
├── data/corpus/             # 5 sample documents on interconnected topics
├── src/
│   ├── 01_index_graphrag.py   # Index corpus with Microsoft GraphRAG
│   ├── 02_index_lightrag.py   # Index corpus with LightRAG
│   ├── 03_community_detection.py  # Leiden community detection + visualization
│   ├── 04_local_search.py     # Entity-focused local search (GraphRAG)
│   ├── 05_global_search.py    # Community-summary global search (GraphRAG)
│   ├── 06_drift_search.py     # DRIFT / hybrid search
│   ├── 07_vector_rag_baseline.py  # Traditional vector RAG with ChromaDB
│   └── 08_comparison.py       # Side-by-side evaluation with LLM-as-judge
├── config/
│   ├── graphrag_settings.yaml # GraphRAG configuration
│   └── lightrag_config.py     # LightRAG configuration
└── output/                    # Generated artifacts
```

## How to Run

Run the scripts in order from the `src/` directory:

```bash
# Step 1 & 2: Build indices (run either or both)
python src/01_index_graphrag.py
python src/02_index_lightrag.py

# Step 3: Explore community structure
python src/03_community_detection.py

# Steps 4-6: Search with GraphRAG
python src/04_local_search.py
python src/05_global_search.py
python src/06_drift_search.py

# Step 7: Build baseline
python src/07_vector_rag_baseline.py

# Step 8: Compare all approaches
python src/08_comparison.py
```

## Corpus Topics

The five sample documents cover interconnected sustainability topics:

1. **Climate Change** - greenhouse gases, global warming, IPCC findings
2. **Renewable Energy** - solar, wind, policy incentives
3. **Electric Vehicles** - batteries, charging infrastructure, adoption
4. **Carbon Capture** - CCS/CCUS technologies, direct air capture
5. **Sustainable Agriculture** - regenerative practices, soil carbon, food security

Entities and relationships span across documents (e.g., "carbon dioxide" appears
in climate change, carbon capture, and agriculture contexts), making this an ideal
test case for graph-based retrieval.
