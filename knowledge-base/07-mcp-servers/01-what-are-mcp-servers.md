# What Are MCP Servers?

The Model Context Protocol (MCP) is an open standard created by Anthropic that enables AI assistants (like Claude) to interact with external tools, data sources, and services through a unified protocol. For knowledge graphs, MCP is a game-changer: it allows you to build, query, and visualize graphs through natural conversation with no code required.

## Core Concepts

MCP follows a client-server architecture:

- **MCP Host**: The AI application (Claude Desktop, Claude Code, Cursor, etc.)
- **MCP Client**: Runs inside the host, maintains a 1:1 connection to a server
- **MCP Server**: A lightweight program that exposes capabilities to the AI

### What MCP Servers Expose

An MCP server can provide three types of capabilities:

| Capability | Description | Example |
|-----------|-------------|---------|
| **Tools** | Functions the AI can call | `run_cypher_query(query)`, `create_node(label, props)` |
| **Resources** | Data the AI can read | Database schema, graph statistics, documentation |
| **Prompts** | Reusable prompt templates | "Analyze this graph for communities", "Generate a Cypher query" |

### How It Works

```
User: "Find all people connected to Anthropic in the graph"
  |
  v
Claude (MCP Host)
  |-- interprets the request
  |-- selects the appropriate MCP tool (e.g., read_cypher)
  |-- sends: run_cypher("MATCH (p:Person)-[]->(o:Org {name:'Anthropic'}) RETURN p")
  |
  v
Neo4j MCP Server
  |-- executes the Cypher query against Neo4j
  |-- returns results as structured data
  |
  v
Claude
  |-- formats results for the user
  |-- "I found 12 people connected to Anthropic..."
```

## Transport Mechanisms

MCP supports two transport types for communication between client and server:

### STDIO Transport (Local)

The server runs as a local subprocess. Communication happens over standard input/output streams.

```json
{
  "mcpServers": {
    "neo4j": {
      "command": "npx",
      "args": ["-y", "@neo4j/mcp-neo4j"],
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "password"
      }
    }
  }
}
```

**Pros**: Simple setup, no network overhead, secure (local only)
**Cons**: Must run on the same machine, one instance per connection

### Streamable HTTP Transport (Remote)

The server runs as a web service. Communication happens over HTTP with Server-Sent Events for streaming.

```json
{
  "mcpServers": {
    "remote-graph": {
      "url": "https://my-mcp-server.example.com/mcp",
      "headers": {
        "Authorization": "Bearer <token>"
      }
    }
  }
}
```

**Pros**: Remote access, shared server, scalable
**Cons**: Network latency, requires authentication setup

## Why MCP Matters for Knowledge Graphs

### 1. No-Code Graph Construction

With MCP, you can build knowledge graphs entirely through conversation:

```
User: "Read this PDF about quantum computing and create a knowledge graph in Neo4j"
Claude: [uses file-read tool] -> [extracts entities with LLM] -> [uses Neo4j MCP to create nodes/edges]
```

### 2. Natural Language Querying

Instead of learning Cypher, Gremlin, or AQL:

```
User: "What are the shortest paths between Einstein and Feynman in my graph?"
Claude: [generates Cypher] -> [runs via MCP] -> [explains results]
```

### 3. Multi-Database Orchestration

Connect multiple graph database MCPs simultaneously:

```
User: "Compare the entity counts in my Neo4j production graph vs my Memgraph staging graph"
Claude: [queries Neo4j MCP] -> [queries Memgraph MCP] -> [compares and reports]
```

### 4. Graph + RAG Integration

Combine graph MCPs with vector database MCPs and LangChain for hybrid retrieval:

```
User: "Answer this question using both the knowledge graph and the document store"
Claude: [graph MCP for structured facts] + [vector MCP for relevant passages] -> [synthesized answer]
```

> **LangChain Tool Integration**: MCP tools can be wrapped as LangChain tools for use in chains and agents. See https://python.langchain.com/docs/integrations/tools/

## Setting Up MCP Servers

### In Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "server-name": {
      "command": "npx",
      "args": ["-y", "package-name"],
      "env": {
        "KEY": "value"
      }
    }
  }
}
```

### In Claude Code

Use the `claude mcp add` command:

```bash
# Add a STDIO-based MCP server
claude mcp add neo4j -- npx -y @neo4j/mcp-neo4j

# Add with environment variables
claude mcp add neo4j \
  --env NEO4J_URI=bolt://localhost:7687 \
  --env NEO4J_USERNAME=neo4j \
  --env NEO4J_PASSWORD=password \
  -- npx -y @neo4j/mcp-neo4j

# List configured servers
claude mcp list

# Remove a server
claude mcp remove neo4j
```

### In Cursor / VS Code

Add to `.cursor/mcp.json` or your editor's MCP configuration file.

## The MCP Ecosystem for Knowledge Graphs

The following sections cover MCP servers organized by function:

- [02 - Graph Database MCPs](./02-graph-database-mcps.md) -- Neo4j, TigerGraph, Memgraph, ArangoDB, FalkorDB
- [03 - KG Construction MCPs](./03-kg-construction-mcps.md) -- Memory servers, Graphiti, knowledge extraction
- [04 - Graph Visualization MCPs](./04-graph-visualization-mcps.md) -- Cytoscape, Arrows.app, MemoryMesh Viewer

## Key Takeaways

1. **MCP standardizes AI-to-tool communication** -- one protocol for all integrations
2. **STDIO for local, HTTP for remote** -- choose based on your deployment needs
3. **Tools, Resources, and Prompts** are the three capability types
4. **Knowledge graphs benefit enormously** from MCP -- enabling no-code construction, natural language querying, and multi-database orchestration
5. **LangChain integration** is possible by wrapping MCP tools as LangChain tools
