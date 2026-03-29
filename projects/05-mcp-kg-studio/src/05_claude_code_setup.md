# Setting Up MCP Servers in Claude Code

This guide walks through configuring MCP servers for knowledge graph operations in Claude Code. Once configured, Claude can build and query knowledge graphs conversationally.

## What is MCP in Claude Code?

Claude Code supports the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), which lets Claude interact with external tools and data sources. When you add an MCP server, Claude gains new capabilities (tools) that it can invoke during your conversation.

For knowledge graphs, this means Claude can:
- Create and query graph databases
- Store and retrieve structured knowledge
- Build temporal knowledge graphs
- All through natural language conversation

## Method 1: CLI Commands (Recommended)

### Add the Memory MCP Server

The Memory server stores a simple knowledge graph as a JSONL file. No database required.

```bash
claude mcp add memory -- npx -y @modelcontextprotocol/server-memory
```

To specify a custom storage path:

```bash
claude mcp add memory \
  -e MEMORY_FILE_PATH=/path/to/your/knowledge.jsonl \
  -- npx -y @modelcontextprotocol/server-memory
```

### Add the Neo4j MCP Server

Requires Neo4j running (see Docker command below).

```bash
claude mcp add neo4j \
  -e NEO4J_URI=bolt://localhost:7687 \
  -e NEO4J_USERNAME=neo4j \
  -e NEO4J_PASSWORD=password123 \
  -- npx -y @neo4j/mcp-neo4j
```

### Add the Graphiti MCP Server

Requires Neo4j running and an OpenAI API key (for embeddings).

```bash
claude mcp add graphiti \
  -e NEO4J_URI=bolt://localhost:7687 \
  -e NEO4J_USERNAME=neo4j \
  -e NEO4J_PASSWORD=password123 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -- python -m graphiti_mcp
```

### Start Neo4j (if needed)

```bash
docker run -d \
  --name kg-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 \
  neo4j:5-community
```

### Verify MCP Servers

```bash
# List all configured MCP servers
claude mcp list

# Check server status
claude mcp status
```

## Method 2: Manual settings.json Configuration

You can also add MCP servers by editing the Claude Code settings file directly.

### File Location

- **Project-level** (recommended): `.claude/settings.json` in your project root
- **User-level**: `~/.claude/settings.json`

### Full Configuration

```json
{
  "mcpServers": {
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"],
      "env": {
        "MEMORY_FILE_PATH": "./data/memory_kg.jsonl"
      }
    },
    "neo4j": {
      "command": "npx",
      "args": ["-y", "@neo4j/mcp-neo4j"],
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "password123"
      }
    },
    "graphiti": {
      "command": "python",
      "args": ["-m", "graphiti_mcp"],
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "password123",
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
        "GRAPHITI_GROUP_ID": "my_project"
      }
    }
  }
}
```

## Using MCP Servers Conversationally

Once configured, start Claude Code and interact naturally:

### Memory MCP Examples

```
You: "Remember that OpenAI developed GPT-4 in 2023"
Claude: [calls create_entities and create_relations tools]
       Done! I've stored that OpenAI developed GPT-4 (2023).

You: "What do you know about OpenAI?"
Claude: [calls search_nodes tool]
       Based on my knowledge graph, OpenAI developed GPT-4 in 2023...

You: "Add that GPT-4 uses the Transformer architecture"
Claude: [calls create_relations tool]
       Added the relationship: GPT-4 uses Transformer architecture.
```

### Neo4j MCP Examples

```
You: "Show me the schema of the knowledge graph"
Claude: [calls get_neo4j_schema tool]
       The graph has these node types: Company, Model, Architecture...

You: "Which companies have invested in AI startups?"
Claude: [calls read_neo4j_cypher tool with generated Cypher]
       Microsoft invested $13B in OpenAI, Google invested $2B in Anthropic...

You: "Create a new node for GPT-5 as an upcoming model by OpenAI"
Claude: [calls write_neo4j_cypher tool]
       Created the GPT-5 node and DEVELOPED relationship from OpenAI.
```

### Graphiti MCP Examples

```
You: "Record that OpenAI released GPT-4o on May 13, 2024"
Claude: [calls add_episode tool with timestamp]
       Recorded the GPT-4o release as a temporal fact.

You: "What was OpenAI's latest model as of January 2024?"
Claude: [calls search tool with temporal context]
       As of January 2024, GPT-4 was OpenAI's latest model.

You: "Mark the fact about GPT-3 being the best model as outdated"
Claude: [calls invalidate_edge tool]
       Marked that fact as superseded.
```

## Tips

1. **Start with Memory MCP** if you just want to experiment. It needs no database.
2. **Use Neo4j MCP** when you need production-grade graph queries with Cypher.
3. **Use Graphiti** when temporal context matters (facts that change over time).
4. **Combine servers** — you can have all three active simultaneously. Claude will pick the right tool based on your request.
5. **Project-level config** keeps MCP settings scoped to specific projects.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "MCP server not found" | Run `claude mcp list` to verify registration |
| Neo4j connection refused | Ensure Docker container is running: `docker ps` |
| Graphiti errors | Check OPENAI_API_KEY is set in your environment |
| Tools not appearing | Restart Claude Code after adding MCP servers |
| Permission errors | Check that npx/python are on your PATH |
