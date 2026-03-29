# Knowledge Base — Reading Guide

15 sections organized as a progressive learning path. Each section builds on previous ones and prepares you for specific projects.

## Core Path (Sections 01–06)

| # | Section | Prepares For |
|---|---------|-------------|
| 01 | **Foundations** — KG basics, graph terminology, RAG recap, why Graph RAG | Everything |
| 02 | **KG Construction** — Entity/relationship extraction, ontology, LLM patterns, pipelines | Project 1 |
| 03 | **Graph Storage** — NetworkX, Neo4j, Cypher, choosing a graph store | Projects 1, 3, 4 |
| 04 | **Graph RAG Architectures** — GraphRAG, Leiden, local/global/DRIFT, LightRAG | Project 2 |
| 05 | **Advanced Topics** — Hybrid retrieval, temporal KGs, multi-hop, agentic RAG | Project 3 |
| 06 | **References** — Papers, tools ecosystem, glossary | Reference |

## Extended Path (Sections 07–11)

| # | Section | Prepares For |
|---|---------|-------------|
| 07 | **MCP Servers** — Graph DB MCPs, KG construction MCPs, visualization MCPs | Project 5 |
| 08 | **Query Languages** — Cypher vs Gremlin vs AQL vs GSQL vs SPARQL, NL-to-query | Project 4 |
| 09 | **Evaluation Metrics** — Graph quality, RAGAS, LLM-as-judge | Project 9 |
| 10 | **Real-World Datasets** — Wikidata, DBpedia, YAGO, FB15k-237, HotPotQA | Experimentation |
| 11 | **Multimodal KGs** — Beyond text, KG versioning and CI/CD | Projects 7, 8 |

## Classical KG Techniques (Sections 12–15)

| # | Section | Prepares For |
|---|---------|-------------|
| 12 | **KG Embeddings Deep Dive** — TransE/RotatE/ComplEx math, PyKEEN tutorial, embeddings for RAG | Project 10 |
| 13 | **Graph Neural Networks** — GCN, GraphSAGE, GAT, node classification, GNNs vs embeddings | Research |
| 14 | **RDF & Semantic Web** — RDF fundamentals, SPARQL mastery, rdflib tutorial | Project 13 |
| 15 | **Graph Algorithms Cookbook** — Centrality, PageRank, paths, k-core, cliques, motifs | Projects 11, 12 |

## LangChain Integration

All code uses [LangChain](https://python.langchain.com/docs/) for model-agnostic access:

- [Chat Models](https://python.langchain.com/docs/integrations/chat/) — OpenAI, Anthropic, Ollama
- [Structured Output](https://python.langchain.com/docs/how_to/structured_output/) — Type-safe extraction
- [Text Embeddings](https://python.langchain.com/docs/integrations/text_embedding/) — Embedding providers
- [Graph Integrations](https://python.langchain.com/docs/integrations/graphs/) — Neo4j, ArangoDB
- [LangGraph](https://langchain-ai.github.io/langgraph/) — Agent orchestration
