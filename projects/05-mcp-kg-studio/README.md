# Project 5: MCP-Powered KG Studio

Use **Model Context Protocol (MCP)** servers to build and query knowledge graphs through AI assistants — without writing graph database code directly.

## What is MCP?

The [Model Context Protocol](https://modelcontextprotocol.io/) is an open standard that lets AI assistants (like Claude) interact with external tools and data sources through a unified protocol. Instead of writing database drivers and query languages yourself, you configure MCP servers that expose graph operations as tools the AI can invoke.

This project demonstrates three MCP servers for knowledge graph work:

| MCP Server | Purpose | Backend | Key Feature |
|-----------|---------|---------|-------------|
| **Neo4j MCP** | Production graph DB access | Neo4j | Full Cypher query support |
| **Memory MCP** | Lightweight KG storage | JSONL files | Zero infrastructure needed |
| **Graphiti MCP** | Temporal knowledge graphs | Neo4j + Zep | Time-aware facts |

## Why MCP for Knowledge Graphs?

1. **Conversational KG construction** — Tell Claude "add a relationship between OpenAI and GPT-4" and it calls the right MCP tool
2. **No query language needed** — The AI translates your intent into Cypher/AQL/etc.
3. **Composable** — Combine multiple MCP servers (e.g., Memory + Neo4j) in the same session
4. **Portable** — Same MCP tools work across Claude Code, Claude Desktop, and other MCP clients

## Setup

### Prerequisites

- Node.js 18+ (for MCP servers)
- Docker (for Neo4j backend)
- Python 3.11+
- Claude Code CLI (for MCP integration)

### 1. Install MCP Servers

```bash
# Official MCP Memory Server
npm install -g @modelcontextprotocol/server-memory

# Neo4j MCP Server
npx -y @neo4j/mcp-neo4j

# Graphiti MCP (via pip)
pip install graphiti-mcp
```

### 2. Start Neo4j (for Neo4j MCP and Graphiti)

```bash
# Use the docker-compose from Project 4, or run standalone:
docker run -d \
  --name kg-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 \
  neo4j:5-community
```

### 3. Configure MCP Servers

See `configs/` for configuration files, or follow the guide in `src/05_claude_code_setup.md`.

### 4. Run the Demos

```bash
python src/01_setup_mcp_servers.py    # Check server availability
python src/02_memory_mcp_demo.py      # Memory MCP demo
python src/03_neo4j_mcp_demo.py       # Neo4j MCP demo
python src/04_graphiti_demo.py        # Graphiti temporal KG demo
```

## File Overview

```
configs/
  neo4j_mcp.json          # MCP config for Neo4j server
  memory_mcp.json         # MCP config for Memory server
  graphiti_mcp.json       # MCP config for Graphiti server
src/
  01_setup_mcp_servers.py # Diagnostic and setup helper
  02_memory_mcp_demo.py   # Build KG with Memory MCP protocol
  03_neo4j_mcp_demo.py    # Query Neo4j through MCP
  04_graphiti_demo.py     # Temporal KG with Graphiti
  05_claude_code_setup.md # Step-by-step Claude Code configuration
```

## References

- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [MCP Server Registry](https://github.com/modelcontextprotocol/servers)
- [Neo4j MCP Server](https://github.com/neo4j/mcp-neo4j)
- [Graphiti by Zep](https://github.com/getzep/graphiti)
- [Claude Code MCP Docs](https://docs.anthropic.com/en/docs/claude-code/mcp)
