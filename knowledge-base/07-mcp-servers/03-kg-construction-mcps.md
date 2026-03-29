# KG Construction MCP Servers

These MCP servers are designed specifically for building and maintaining knowledge graphs. Unlike the database MCPs (which provide raw query access), these servers handle entity extraction, relationship management, and graph construction as first-class operations.

## Official MCP Memory Server

**Package**: `@modelcontextprotocol/server-memory`
**Source**: Part of the official MCP reference servers
**Storage**: JSON file-based (local)

The reference memory server from the MCP specification. It provides a simple entity-observation-relation model for persistent memory.

### Core Concepts

- **Entities**: Named nodes with an entity type (e.g., "John" of type "Person")
- **Observations**: Facts about an entity (stored as a list of strings)
- **Relations**: Directed edges between entities (e.g., "John" --WORKS_AT--> "Acme")

### Available Tools

| Tool | Description |
|------|-------------|
| `create_entities` | Create new entities with names, types, and observations |
| `add_observations` | Add new observations to existing entities |
| `create_relations` | Create directed relationships between entities |
| `delete_entities` | Remove entities and their associated relations |
| `delete_observations` | Remove specific observations from entities |
| `delete_relations` | Remove relationships between entities |
| `read_graph` | Read the entire knowledge graph |
| `search_nodes` | Search for entities by name or observation content |
| `open_nodes` | Open specific nodes by name |

### Setup

```bash
# Claude Code
claude mcp add memory -- npx -y @modelcontextprotocol/server-memory

# With custom storage path
claude mcp add memory \
  --env MEMORY_FILE_PATH=/path/to/my-memory.json \
  -- npx -y @modelcontextprotocol/server-memory
```

### Claude Desktop Configuration

```json
{
  "mcpServers": {
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"],
      "env": {
        "MEMORY_FILE_PATH": "/Users/you/knowledge-graph.json"
      }
    }
  }
}
```

### Usage Example

```
User: "Remember that Dr. Sarah Chen works at MIT and specializes in NLP"
Claude: [calls create_entities with entity "Sarah Chen" of type "Person"]
        [calls add_observations: "Works at MIT", "Specializes in NLP"]
        [calls create_relations: Sarah Chen --WORKS_AT--> MIT]
```

### Limitations

- File-based storage (not suitable for large graphs)
- No query language -- only search and full graph read
- No schema enforcement
- No built-in visualization

## Graphiti by Zep

**Repository**: https://github.com/getzep/graphiti
**Stars**: 24,000+
**Backend**: Neo4j
**Key Feature**: Temporal knowledge graph with episodic memory

Graphiti is a sophisticated knowledge graph framework that maintains temporal awareness -- it tracks when facts were learned, when they changed, and supports point-in-time queries.

### Architecture

```
User Conversations / Documents
        |
        v
   Graphiti Engine
   - Entity extraction (LLM-based)
   - Relationship extraction
   - Temporal metadata
   - Entity resolution
   - Contradiction detection
        |
        v
   Neo4j Database
   (nodes, edges, episodes, temporal metadata)
```

### Key Features

- **Temporal awareness**: Every fact has a valid_from and valid_to timestamp
- **Episodic memory**: Raw conversation turns stored as episodes
- **Entity resolution**: Automatically merges duplicate entities
- **Contradiction handling**: Detects and resolves conflicting facts
- **MCP server built-in**: Ships with an MCP server for Claude integration

### Setup

```bash
# Install Graphiti
pip install graphiti-core

# Start Neo4j (required backend)
docker run -d -p 7687:7687 -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest

# Run the Graphiti MCP server
claude mcp add graphiti \
  --env NEO4J_URI=bolt://localhost:7687 \
  --env NEO4J_USER=neo4j \
  --env NEO4J_PASSWORD=password \
  --env OPENAI_API_KEY=sk-... \
  -- python -m graphiti_core.mcp_server
```

### Using Graphiti with LangChain

Graphiti uses LLMs internally for extraction. You can configure it to use any LangChain-compatible model:

```python
from graphiti_core import Graphiti
from langchain_openai import ChatOpenAI

# Graphiti uses LangChain under the hood for LLM calls
graphiti = Graphiti(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password",
)

# Add episodes (conversations, documents, etc.)
await graphiti.add_episode(
    name="meeting-notes",
    episode_body="Sarah mentioned that the project deadline moved to March 2026.",
    source_description="Team meeting transcript",
)
```

> **LangChain ChatOpenAI**: https://python.langchain.com/docs/integrations/chat/openai/

## kg-memory-mcp

**Focus**: Transforms unstructured text into structured knowledge graphs using AI extraction

This MCP server combines LLM-based extraction with graph storage, automatically converting free-form text into entities and relationships.

### Key Features

- Automatic entity and relationship extraction from text
- Schema-aware extraction (define entity types and relation types)
- Multiple storage backends
- Incremental graph building

### Setup

```bash
claude mcp add kg-memory \
  --env OPENAI_API_KEY=sk-... \
  -- npx -y kg-memory-mcp
```

### Usage

```
User: "Process this text and add it to my knowledge graph: 'Tesla, founded by
       Elon Musk in 2003, manufactures electric vehicles at its Gigafactory in Austin, Texas.'"
Claude: [calls extract_and_store tool]
        -> Creates entities: Tesla (ORG), Elon Musk (PERSON), Austin (LOCATION), Texas (LOCATION)
        -> Creates relations: FOUNDED_BY, MANUFACTURES_AT, LOCATED_IN
```

## MemoryMesh

**Focus**: Schema-driven knowledge graph with auto-generated tools
**Key Feature**: Define a schema, get MCP tools automatically

MemoryMesh is unique in that it generates MCP tools dynamically based on your schema definition. Define your entity types and relationships, and MemoryMesh creates corresponding create/read/update/delete tools.

### How It Works

1. You define a schema (entity types, properties, relationship types)
2. MemoryMesh generates MCP tools for each entity type
3. Claude uses these tools to build and query the graph

### Schema Example

```json
{
  "entities": {
    "Person": {
      "properties": {
        "name": "string",
        "role": "string",
        "expertise": "string[]"
      }
    },
    "Project": {
      "properties": {
        "name": "string",
        "status": "string",
        "deadline": "date"
      }
    }
  },
  "relationships": ["WORKS_ON", "MANAGES", "DEPENDS_ON"]
}
```

This generates tools like `create_person`, `update_person`, `get_person`, `create_works_on_relation`, etc.

### Setup

```bash
claude mcp add memorymesh \
  --env SCHEMA_PATH=/path/to/schema.json \
  -- npx -y memorymesh-mcp
```

## memory-graph

**Focus**: Multi-backend MCP memory server
**Backends**: SQLite, Neo4j, FalkorDB

This server provides a unified memory interface that can store knowledge graphs in different backends depending on your needs.

### Backend Comparison

| Backend | Best For | Setup Complexity |
|---------|----------|-----------------|
| SQLite | Local development, small graphs | None (embedded) |
| Neo4j | Production, complex queries | Docker or Aura |
| FalkorDB | Low-latency, AI-optimized | Docker |

### Setup with SQLite (Simplest)

```bash
claude mcp add memory-graph \
  --env BACKEND=sqlite \
  --env DB_PATH=/path/to/memory.db \
  -- npx -y memory-graph-mcp
```

### Setup with Neo4j

```bash
claude mcp add memory-graph \
  --env BACKEND=neo4j \
  --env NEO4J_URI=bolt://localhost:7687 \
  --env NEO4J_USERNAME=neo4j \
  --env NEO4J_PASSWORD=password \
  -- npx -y memory-graph-mcp
```

### Setup with FalkorDB

```bash
claude mcp add memory-graph \
  --env BACKEND=falkordb \
  --env FALKORDB_HOST=localhost \
  --env FALKORDB_PORT=6379 \
  -- npx -y memory-graph-mcp
```

## Choosing the Right KG Construction MCP

| Server | Extraction | Storage | Temporal | Schema | Best For |
|--------|-----------|---------|----------|--------|---------|
| MCP Memory | Manual | JSON file | No | No | Simple note-taking, prototyping |
| Graphiti | Automatic (LLM) | Neo4j | Yes | Auto | Conversations, evolving knowledge |
| kg-memory-mcp | Automatic (LLM) | Configurable | No | Optional | Document processing |
| MemoryMesh | Manual | In-memory | No | Required | Structured domains |
| memory-graph | Manual | Multi-backend | No | No | Flexible deployment |

## Integration with LangChain

All of these MCP servers can be used alongside LangChain-based extraction pipelines. For example, use LangChain's `ChatOpenAI.with_structured_output()` for initial extraction (see [04 - LLM Extraction Patterns](../02-kg-construction/04-llm-extraction-patterns.md)), then store results via an MCP server.

```python
from langchain_openai import ChatOpenAI

# Extract with LangChain
llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm = llm.with_structured_output(ExtractionResult)
result = structured_llm.invoke(messages)

# Store via MCP (conceptual -- actual MCP calls go through the host)
# The AI assistant handles this automatically when both tools are available
```

> **LangChain structured output**: https://python.langchain.com/docs/how_to/structured_output/

## Next Steps

- [02 - Graph Database MCPs](./02-graph-database-mcps.md) -- for raw database access
- [04 - Graph Visualization MCPs](./04-graph-visualization-mcps.md) -- visualize your constructed graphs
