# Project 17: Autonomous Knowledge Graph Agent

An autonomous agent that builds, maintains, and improves a knowledge graph with
minimal human intervention. It monitors data sources, decides what to ingest based
on relevance and coverage gaps, extracts and validates facts, detects conflicts
with existing knowledge, self-evaluates quality, and plans improvement actions.

This is the most advanced project in the repository.

## Architecture

```
                        +---------------------------+
                        |    Autonomous Agent Loop   |
                        |     (LangGraph FSM)        |
                        +-------------+-------------+
                                      |
         +----------------------------+----------------------------+
         |              |             |             |              |
   +-----+------+ +----+------+ +----+------+ +----+------+ +----+------+
   |  Assess    | |  Detect   | |   Plan    | |  Execute  | | Evaluate  |
   |  State     | |  Gaps     | |  Actions  | |  Action   | |  Quality  |
   +-----+------+ +-----+----+ +-----+-----+ +-----+-----+ +-----+-----+
         |               |           |             |               |
         +-------+-------+-----------+------+------+-------+-------+
                 |                          |                |
          +------+------+           +------+------+  +------+------+
          | KG Analyzer |           | Source Mgr  |  |  Conflict   |
          | (Stats,     |           | (Evaluate,  |  |  Resolver   |
          |  Coverage,  |           |  Track,     |  |  (Assess,   |
          |  Quality)   |           |  Prioritize)|  |   Decide)   |
          +------+------+           +------+------+  +------+------+
                 |                          |                |
                 +----------+------+--------+----------------+
                            |      |
                     +------+------+------+
                     |    Neo4j KG        |
                     | - Entities         |
                     | - Relationships    |
                     | - Provenance       |
                     | - Temporal history |
                     +--------------------+

   Autonomous Loop: assess -> detect -> plan -> execute -> evaluate
                                   ^                          |
                                   |   should_continue?       |
                                   +-----------+--------------+
                                               |
                                          (done / budget)
```

## How It Works

1. **Seed**: The agent starts with a seed document about a topic (e.g., "History of AI")
2. **Assess**: Analyzes the current KG state (size, coverage, quality score)
3. **Detect**: Finds coverage gaps -- sparse entities, missing subtopics, disconnected areas
4. **Plan**: Uses LLM reasoning to prioritize next actions (ingest, enrich, validate, search)
5. **Execute**: Carries out the planned action (extract from source, web search, dedup)
6. **Evaluate**: Checks if the action improved the KG quality score
7. **Loop**: Decides whether to continue (quality threshold? budget exhausted?) or stop

The agent also evaluates candidate sources for relevance before ingesting them,
and resolves conflicts when new facts contradict existing ones.

## Components

| Component | Description |
|---|---|
| **Autonomous Agent** | Main LangGraph loop with assess/detect/plan/execute/evaluate cycle |
| **Relevance Scorer** | Evaluates candidate documents against current KG state |
| **Conflict Resolver** | Handles contradictory facts using source authority and recency |
| **Gap Detector** | Analyzes entity types, relationship density, connected components |
| **Action Planner** | LLM-powered planning based on gaps and priorities |
| **KG Analyzer** | Tools for introspection: stats, coverage, quality, duplicates |
| **Source Manager** | Tracks processing history and manages document queue |

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
pip install langchain langchain-openai langgraph neo4j tavily-python pydantic matplotlib

# 3. Set environment variables in .env at repo root

# 4. Run the autonomous agent
python src/run_autonomous.py

# Optional: customize iterations
python src/run_autonomous.py --iterations 10
```

## References

- [LangGraph StateGraph](https://langchain-ai.github.io/langgraph/concepts/low_level/)
- [LangGraph Conditional Edges](https://langchain-ai.github.io/langgraph/how-tos/branching/)
