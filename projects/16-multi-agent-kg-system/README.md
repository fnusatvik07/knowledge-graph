# Project 16: Multi-Agent Knowledge Graph System

Multiple specialized agents collaborating through a shared Neo4j knowledge graph.
Uses LangGraph multi-agent patterns with a supervisor orchestrator that delegates
to Extractor, Validator, Enricher, and Researcher agents.

## Architecture

```
                    +---------------------+
                    |   Supervisor Agent   |
                    | (LangGraph Router)   |
                    +----------+----------+
                               |
          +--------------------+--------------------+
          |          |              |                |
   +------+------+  +------+------+ +------+------+ +------+------+
   |  Extractor  |  |  Validator  | |  Enricher   | |  Researcher |
   |    Agent    |  |    Agent    | |    Agent    | |    Agent    |
   +------+------+  +------+------+ +------+------+ +------+------+
          |                |              |                |
          |  writes        | reads/fixes  | reads/writes   | reads
          v                v              v                v
   +------+----------------+--------------+----------------+------+
   |                    Shared Neo4j KG                           |
   |  - Entities with types, provenance, timestamps               |
   |  - Relationships with source attribution                     |
   |  - Agent memory nodes for session persistence                |
   +-------------------------------------------------------------+
          ^                                              |
          |              +----------------+              |
          +--------------+  KG Tools      +--------------+
                         | (LangChain)    |
                         +-------+--------+
                                 |
                         +-------+--------+
                         |  Web Tools     |
                         | (Tavily)       |
                         +----------------+
```

## Agents

| Agent | Role |
|---|---|
| **Supervisor** | Orchestrates all agents, delegates tasks, tracks completion |
| **Extractor** | Parses documents, extracts entities/relationships, writes to KG |
| **Validator** | Checks data quality, fixes duplicates, enforces type consistency |
| **Enricher** | Detects gaps in the KG, searches the web, adds missing facts |
| **Researcher** | Answers questions using dynamic tool selection (Cypher, vector, web) |

## Features

- **Parallel extraction**: Supervisor ingests multiple documents simultaneously
- **Provenance tracking**: Every fact records which agent wrote it, from which source, and when
- **Dynamic tool use**: Researcher agent uses LLM function calling to pick the right tool
- **KG-backed memory**: Agents persist state across sessions via Neo4j
- **LangGraph checkpointing**: Interrupted workflows can be resumed

## Prerequisites

- Python 3.11+
- Neo4j (via Docker)
- OpenAI API key (or Anthropic/Ollama)
- Tavily API key (for web search)

## Quick Start

```bash
# 1. Start Neo4j
docker compose up -d

# 2. Install dependencies
pip install langchain langchain-openai langchain-anthropic langgraph neo4j tavily-python pydantic

# 3. Set environment variables in .env at repo root
#    OPENAI_API_KEY=...
#    TAVILY_API_KEY=...
#    NEO4J_URI=bolt://localhost:7687
#    NEO4J_PASSWORD=password

# 4. Run the demo
python src/run_demo.py
```

## References

- [LangGraph Multi-Agent](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)
- [LangGraph Persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- [LangChain Tools](https://python.langchain.com/docs/how_to/custom_tools/)
