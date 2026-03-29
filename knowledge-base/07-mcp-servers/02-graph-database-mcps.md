# Graph Database MCP Servers

This section catalogs MCP servers that provide direct access to graph databases. These servers let AI assistants query, modify, and manage graph databases through natural conversation.

## Neo4j Official MCP Server

**Repository**: https://github.com/neo4j/mcp-neo4j
**Package**: `@neo4j/mcp-neo4j`

The official MCP server maintained by Neo4j. Provides safe, controlled access to Neo4j databases.

### Available Tools

| Tool | Description |
|------|-------------|
| `get-neo4j-schema` | Retrieves the database schema (node labels, relationship types, properties) |
| `read-neo4j-cypher` | Executes read-only Cypher queries (uses `ACCESS` mode) |
| `write-neo4j-cypher` | Executes write Cypher queries (uses `WRITE` mode) |
| `list-neo4j-gds-procedures` | Lists available Graph Data Science library procedures |

### Setup

```bash
# Claude Code
claude mcp add neo4j \
  --env NEO4J_URI=bolt://localhost:7687 \
  --env NEO4J_USERNAME=neo4j \
  --env NEO4J_PASSWORD=password \
  -- npx -y @neo4j/mcp-neo4j

# Or with Neo4j Aura (cloud)
claude mcp add neo4j-aura \
  --env NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io \
  --env NEO4J_USERNAME=neo4j \
  --env NEO4J_PASSWORD=your-aura-password \
  -- npx -y @neo4j/mcp-neo4j
```

### Claude Desktop Configuration

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

### Usage Example

Once connected, you can interact naturally:

```
User: "Show me the schema of my Neo4j database"
Claude: [calls get-neo4j-schema] -> displays node labels, relationships, and properties

User: "Find all Person nodes connected to more than 5 organizations"
Claude: [calls read-neo4j-cypher with appropriate Cypher] -> returns matching persons

User: "Create a MENTORED relationship between Ada Lovelace and Charles Babbage"
Claude: [calls write-neo4j-cypher] -> confirms creation
```

## Neo4j Labs / Community MCP Servers

**Repository**: https://github.com/neo4j-contrib/mcp-neo4j
**Packages**: Multiple specialized servers

The Neo4j community maintains several specialized MCP servers:

### mcp-neo4j-cypher

General-purpose Cypher execution server.

```bash
claude mcp add neo4j-cypher \
  --env NEO4J_URI=bolt://localhost:7687 \
  --env NEO4J_USERNAME=neo4j \
  --env NEO4J_PASSWORD=password \
  -- npx -y @neo4j-contrib/mcp-neo4j-cypher
```

### mcp-neo4j-memory

Knowledge graph-based memory system built on Neo4j. Stores entities, observations, and relations as a persistent memory layer for AI assistants.

```bash
claude mcp add neo4j-memory \
  --env NEO4J_URI=bolt://localhost:7687 \
  --env NEO4J_USERNAME=neo4j \
  --env NEO4J_PASSWORD=password \
  -- npx -y @neo4j-contrib/mcp-neo4j-memory
```

### mcp-neo4j-cloud

Manages Neo4j Aura cloud instances -- create, pause, resume, and delete databases.

```bash
claude mcp add neo4j-cloud \
  --env NEO4J_CLIENT_ID=your-client-id \
  --env NEO4J_CLIENT_SECRET=your-client-secret \
  -- npx -y @neo4j-contrib/mcp-neo4j-cloud
```

### mcp-neo4j-data-modeling

Interactive graph data modeling using the Arrows.app visualization tool. Design your graph schema visually and export it as Cypher.

```bash
claude mcp add neo4j-modeling \
  -- npx -y @neo4j-contrib/mcp-neo4j-data-modeling
```

## TigerGraph MCP Server

**Repository**: https://github.com/tigergraph/tigergraph-mcp
**Tools**: 70+ tools covering the full TigerGraph API

TigerGraph's MCP server provides comprehensive access to TigerGraph's distributed graph analytics platform.

### Key Tool Categories

| Category | Tools | Examples |
|----------|-------|---------|
| Schema Management | ~10 tools | `create_vertex_type`, `create_edge_type`, `get_schema` |
| Data Loading | ~8 tools | `upsert_vertex`, `upsert_edge`, `load_data` |
| Query Execution | ~12 tools | `run_query`, `run_interpreted_query` |
| Graph Analytics | ~15 tools | `pagerank`, `community_detection`, `shortest_path` |
| Admin | ~10 tools | `create_graph`, `drop_graph`, `get_statistics` |

### Setup

```bash
# Install
pip install tigergraph-mcp

# Claude Code
claude mcp add tigergraph \
  --env TG_HOST=https://your-instance.i.tgcloud.io \
  --env TG_USERNAME=tigergraph \
  --env TG_PASSWORD=password \
  --env TG_GRAPH_NAME=MyGraph \
  -- python -m tigergraph_mcp
```

### Claude Desktop Configuration

```json
{
  "mcpServers": {
    "tigergraph": {
      "command": "python",
      "args": ["-m", "tigergraph_mcp"],
      "env": {
        "TG_HOST": "https://your-instance.i.tgcloud.io",
        "TG_USERNAME": "tigergraph",
        "TG_PASSWORD": "password",
        "TG_GRAPH_NAME": "MyGraph"
      }
    }
  }
}
```

## Memgraph MCP Server

**Part of**: Memgraph AI Toolkit
**Repository**: Available through Memgraph's AI toolkit

Memgraph's MCP server provides access to Memgraph, a high-performance in-memory graph database compatible with Cypher.

### Setup

```bash
# Start Memgraph (Docker)
docker run -d -p 7687:7687 -p 7444:7444 memgraph/memgraph-platform

# Claude Code
claude mcp add memgraph \
  --env MEMGRAPH_URI=bolt://localhost:7687 \
  --env MEMGRAPH_USERNAME="" \
  --env MEMGRAPH_PASSWORD="" \
  -- npx -y @memgraph/mcp-memgraph
```

### Claude Desktop Configuration

```json
{
  "mcpServers": {
    "memgraph": {
      "command": "npx",
      "args": ["-y", "@memgraph/mcp-memgraph"],
      "env": {
        "MEMGRAPH_URI": "bolt://localhost:7687"
      }
    }
  }
}
```

### Key Features

- Full Cypher query execution (read and write)
- Schema introspection
- MAGE (Memgraph Advanced Graph Extensions) algorithm access
- Streaming query results

## ArangoDB MCP Server

**Repository**: Available through ArangoDB's ecosystem
**Query Language**: AQL (ArangoDB Query Language)

ArangoDB's MCP server exposes its multi-model database (graph + document + key-value) to AI assistants.

### Setup

```bash
# Start ArangoDB (Docker)
docker run -d -p 8529:8529 -e ARANGO_ROOT_PASSWORD=password arangodb/arangodb

# Claude Code
claude mcp add arangodb \
  --env ARANGO_URL=http://localhost:8529 \
  --env ARANGO_USERNAME=root \
  --env ARANGO_PASSWORD=password \
  --env ARANGO_DATABASE=_system \
  -- npx -y @arangodb/mcp-arangodb
```

### Key Features

- Natural language to AQL query conversion
- Graph traversal queries
- Document collection management
- Multi-model queries (combine graph + document operations)

### Usage Example

```
User: "Find the shortest path between node A and node B in my ArangoDB graph"
Claude: [generates AQL traversal query] -> [executes via MCP] -> [returns path]
```

## FalkorDB MCP Server

**Repository**: https://github.com/FalkorDB/FalkorDB-MCPServer
**Query Language**: OpenCypher (subset of Cypher)

FalkorDB is a low-latency graph database optimized for AI workloads. Its MCP server provides OpenCypher query access.

### Setup

```bash
# Start FalkorDB (Docker)
docker run -d -p 6379:6379 falkordb/falkordb

# Claude Code
claude mcp add falkordb \
  --env FALKORDB_HOST=localhost \
  --env FALKORDB_PORT=6379 \
  -- npx -y falkordb-mcp
```

### Claude Desktop Configuration

```json
{
  "mcpServers": {
    "falkordb": {
      "command": "npx",
      "args": ["-y", "falkordb-mcp"],
      "env": {
        "FALKORDB_HOST": "localhost",
        "FALKORDB_PORT": "6379"
      }
    }
  }
}
```

### Key Features

- OpenCypher query execution
- Schema retrieval
- Graph creation and deletion
- Optimized for low-latency AI workloads (sub-millisecond queries)

## Comparison Table

| MCP Server | Database | Query Language | Tools Count | Setup Complexity | Best For |
|-----------|----------|---------------|------------|-----------------|---------|
| Neo4j Official | Neo4j | Cypher | 4 | Low | Production Neo4j |
| Neo4j Labs | Neo4j | Cypher | 10+ | Low | Memory, modeling, cloud |
| TigerGraph | TigerGraph | GSQL | 70+ | Medium | Enterprise analytics |
| Memgraph | Memgraph | Cypher | ~6 | Low | Real-time / streaming |
| ArangoDB | ArangoDB | AQL | ~8 | Medium | Multi-model workloads |
| FalkorDB | FalkorDB | OpenCypher | ~5 | Low | Low-latency AI apps |

## Using with LangChain

MCP tools can be integrated into LangChain chains and agents. Wrap MCP tool calls as LangChain tools:

```python
from langchain_core.tools import tool

@tool
def query_neo4j(cypher_query: str) -> str:
    """Execute a read-only Cypher query against Neo4j via MCP."""
    # In practice, this calls the MCP tool
    pass
```

> **LangChain Tool docs**: https://python.langchain.com/docs/how_to/custom_tools/
> **LangChain Graph integrations**: https://python.langchain.com/docs/integrations/graphs/

## Next Steps

- [03 - KG Construction MCPs](./03-kg-construction-mcps.md) -- MCP servers for building knowledge graphs
- [04 - Graph Visualization MCPs](./04-graph-visualization-mcps.md) -- visualize your graphs through MCP
