# Project 11: Fraud Detection KG

Build a transaction knowledge graph and detect fraudulent patterns using graph algorithms. A classic and high-impact application of knowledge graphs in financial services.

## What This Project Does

1. **Build Transaction Graph** -- Construct a directed graph from synthetic transaction data. Nodes are accounts (with metadata), edges are transactions (with amount, timestamp, category).
2. **Graph Analysis** -- Run structural algorithms to surface suspicious accounts: degree analysis, PageRank, betweenness centrality, connected components, money flow imbalance.
3. **Cycle Detection** -- Find circular money flows (A->B->C->A) that may indicate money laundering. Report cycle lengths, total amounts, and involved accounts.
4. **Community Detection** -- Apply Louvain community detection to find transaction clusters. Identify accounts that bridge multiple communities (potential mule accounts).
5. **Temporal Analysis** -- Time-based fraud detection: burst detection, unusual-hour transactions, rapid successive transactions (velocity checks).
6. **Fraud Scoring** -- Combine all signals into a composite fraud risk score per account. Use LLM to generate human-readable fraud reports for top suspects.
7. **Network Visualization** -- Interactive visualization with pyvis. Nodes colored by fraud risk (green to red), detected cycles highlighted, node size by transaction volume.

## Fraud Patterns Detected

- **Circular Money Flows**: A->B->C->A loops indicating potential laundering
- **Burst Activity**: Sudden spikes in transaction frequency
- **Unusual Hours**: Transactions at atypical times (e.g., 2-5 AM)
- **High Betweenness**: Accounts that bridge otherwise disconnected clusters (mule accounts)
- **Flow Imbalance**: Large discrepancy between money in and money out
- **Flagged Associations**: Transactions with known suspicious accounts

## Prerequisites

- Python 3.11+
- Dependencies: `networkx`, `matplotlib`, `pyvis`, `pandas`, `numpy`, `python-louvain`
- Optional: `neo4j` (for persistent graph storage)
- LLM access for fraud report generation (via shared LLM clients)

## Quick Start

```bash
# Build the transaction graph
python src/01_build_transaction_graph.py

# Run graph analysis algorithms
python src/02_graph_analysis.py

# Detect circular money flows
python src/03_cycle_detection.py

# Find transaction communities
python src/04_community_detection.py

# Time-based fraud detection
python src/05_temporal_analysis.py

# Generate composite fraud scores and reports
python src/06_fraud_scoring.py

# Visualize the network
python src/07_visualize_network.py
```

## File Structure

```
11-fraud-detection-kg/
├── README.md
├── data/
│   ├── transactions.json
│   └── accounts.json
├── output/
└── src/
    ├── __init__.py
    ├── 01_build_transaction_graph.py
    ├── 02_graph_analysis.py
    ├── 03_cycle_detection.py
    ├── 04_community_detection.py
    ├── 05_temporal_analysis.py
    ├── 06_fraud_scoring.py
    └── 07_visualize_network.py
```
