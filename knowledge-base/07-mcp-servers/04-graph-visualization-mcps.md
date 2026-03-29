# Graph Visualization MCP Servers

Visualization is critical for understanding, debugging, and presenting knowledge graphs. These MCP servers enable AI assistants to create, modify, and interact with graph visualizations programmatically.

## Cytoscape MCP Server

**Repository**: https://github.com/dexterpratt/cytoscape-mcp
**Visualization Engine**: Cytoscape (desktop application)
**Use Case**: Network visualization, biological networks, complex graph layouts

Cytoscape is a mature, open-source platform for visualizing complex networks. The Cytoscape MCP server bridges it with AI assistants, enabling programmatic control of visualizations through natural language.

### What Cytoscape MCP Can Do

| Tool Category | Examples |
|--------------|---------|
| Network Creation | Create networks from node/edge lists, import from files |
| Layout | Apply force-directed, hierarchical, circular layouts |
| Styling | Set node colors by property, size by degree, edge width by weight |
| Analysis | Compute centrality, clustering, connected components |
| Export | Save as PNG, SVG, PDF, or Cytoscape session |
| Selection | Select nodes by property, expand selection to neighbors |

### Setup

```bash
# 1. Install Cytoscape desktop application (https://cytoscape.org/download.html)

# 2. Install CyREST app in Cytoscape (Apps -> App Manager -> search "CyREST")
#    This enables the REST API that the MCP server communicates with.

# 3. Add the MCP server to Claude Code
claude mcp add cytoscape \
  --env CYTOSCAPE_HOST=localhost \
  --env CYTOSCAPE_PORT=1234 \
  -- npx -y cytoscape-mcp
```

### Claude Desktop Configuration

```json
{
  "mcpServers": {
    "cytoscape": {
      "command": "npx",
      "args": ["-y", "cytoscape-mcp"],
      "env": {
        "CYTOSCAPE_HOST": "localhost",
        "CYTOSCAPE_PORT": "1234"
      }
    }
  }
}
```

### Usage Examples

#### Creating a Network from a Knowledge Graph

```
User: "Visualize my knowledge graph about the AI research landscape"
Claude: [creates network with nodes for researchers, institutions, papers]
        [applies force-directed layout]
        [colors nodes by type: blue=Person, green=Institution, orange=Paper]
        [sizes nodes by number of connections]
```

#### Analyzing Graph Structure

```
User: "Highlight the most influential nodes in my network"
Claude: [computes betweenness centrality via Cytoscape]
        [maps centrality to node size]
        [identifies top 10 bridge nodes]
        "The most influential nodes are..."
```

#### Exporting Visualizations

```
User: "Export this graph as a high-resolution PNG for my presentation"
Claude: [adjusts layout for readability]
        [exports as PNG at 300 DPI]
        "Saved to /path/to/graph-visualization.png"
```

### Integrating Neo4j Data with Cytoscape

A common workflow: query Neo4j via MCP, then visualize in Cytoscape via MCP.

```
1. User: "Show me the collaboration network for the NLP research community"
2. Claude: [Neo4j MCP] MATCH (a:Researcher)-[c:COAUTHORED]->(p:Paper)<-[c2:COAUTHORED]-(b:Researcher) RETURN a, b, count(p) as papers
3. Claude: [Cytoscape MCP] Create network with results, edge weight = paper count
4. Claude: [Cytoscape MCP] Apply force-directed layout, color by institution
```

## Neo4j Data Modeling MCP

**Package**: `@neo4j-contrib/mcp-neo4j-data-modeling`
**Repository**: https://github.com/neo4j-contrib/mcp-neo4j
**Visualization**: Arrows.app (web-based graph schema designer)

This MCP server focuses on graph schema design rather than data visualization. It connects to Arrows.app, a visual tool for designing graph data models.

### What It Does

- Design graph schemas visually (node labels, relationship types, properties)
- Export schemas as Cypher DDL statements
- Import existing database schemas for visualization
- Collaborate on graph model design

### Setup

```bash
claude mcp add neo4j-modeling \
  -- npx -y @neo4j-contrib/mcp-neo4j-data-modeling
```

### Usage Examples

#### Designing a Schema

```
User: "Design a graph schema for a movie recommendation system"
Claude: [creates node types: Movie, Person, Genre, Studio]
        [creates relationships: ACTED_IN, DIRECTED, HAS_GENRE, PRODUCED_BY]
        [adds properties: Movie.title, Movie.year, Person.name, etc.]
        [renders in Arrows.app]
```

#### Evolving an Existing Schema

```
User: "Add a Rating node to my movie graph schema that connects Users to Movies"
Claude: [loads existing schema]
        [adds Rating node with properties: score, timestamp]
        [creates RATED relationship: User -> Rating -> Movie]
        [updates Arrows.app visualization]
```

### Arrows.app Integration

Arrows.app (https://arrows.app) is a free, web-based tool for drawing graph models. The MCP integration allows:

1. **Programmatic model creation** -- Claude describes the model, MCP creates it in Arrows
2. **Schema export** -- export the visual model as Cypher `CREATE` statements
3. **Import from database** -- pull a live Neo4j schema into Arrows for visualization

## MemoryMesh Memory Viewer

**Part of**: MemoryMesh MCP server
**Type**: Built-in visualization component

MemoryMesh includes a memory viewer that provides a real-time visual representation of the knowledge graph being built through MCP interactions.

### Features

- **Live updates**: Graph visualization updates as entities and relations are added
- **Schema-aware rendering**: Different entity types rendered with distinct colors/shapes
- **Relationship labels**: Displays relationship types on edges
- **Search and filter**: Find specific entities or relationship types
- **Timeline view**: See when entities were added (if temporal metadata is available)

### Setup

The memory viewer starts automatically with MemoryMesh:

```bash
claude mcp add memorymesh \
  --env SCHEMA_PATH=/path/to/schema.json \
  --env VIEWER_PORT=3000 \
  -- npx -y memorymesh-mcp
```

Access the viewer at `http://localhost:3000` while the MCP server is running.

## Programmatic Visualization with Python

When MCP visualization servers are not available, you can generate visualizations programmatically using LangChain to query the graph and Python libraries to render.

### NetworkX + Matplotlib

```python
import networkx as nx
import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI

# Query graph data (via LangChain Neo4j integration)
# See: https://python.langchain.com/docs/integrations/graphs/neo4j_cypher/

G = nx.DiGraph()
# Add nodes and edges from query results
G.add_edge("Marie Curie", "University of Paris", relation="WORKED_AT")
G.add_edge("Marie Curie", "Nobel Prize", relation="AWARDED")

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=2000)
edge_labels = nx.get_edge_attributes(G, "relation")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.savefig("knowledge_graph.png", dpi=150, bbox_inches="tight")
```

### PyVis (Interactive HTML)

```python
from pyvis.network import Network

net = Network(height="600px", width="100%", directed=True)
net.add_node("Marie Curie", label="Marie Curie", color="lightblue")
net.add_node("University of Paris", label="University of Paris", color="lightgreen")
net.add_edge("Marie Curie", "University of Paris", title="WORKED_AT")

net.show("knowledge_graph.html")
```

> **LangChain Neo4j integration**: https://python.langchain.com/docs/integrations/graphs/neo4j_cypher/

## Choosing a Visualization Approach

| Approach | Interactive | Setup | Best For |
|----------|-----------|-------|---------|
| Cytoscape MCP | Yes (desktop) | Medium | Complex networks, analysis |
| Arrows.app MCP | Yes (web) | Low | Schema design |
| MemoryMesh Viewer | Yes (web) | Low | Live KG monitoring |
| NetworkX + Matplotlib | No (static) | None | Quick exports, papers |
| PyVis | Yes (HTML) | None | Shareable interactive graphs |
| Neo4j Browser | Yes (web) | None (built into Neo4j) | Cypher result exploration |

## Next Steps

- [01 - What Are MCP Servers](./01-what-are-mcp-servers.md) -- MCP fundamentals
- [02 - Graph Database MCPs](./02-graph-database-mcps.md) -- query databases to get data for visualization
