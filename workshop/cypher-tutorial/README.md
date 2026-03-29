# Hour 1: Cypher Basics — Learn by Doing

Open Neo4j Browser at http://localhost:7474 and run each `.cypher` file step by step.

**Copy-paste each query into the Neo4j query bar and press Enter/Play.**

## Order

| File | What You'll Learn |
|------|------------------|
| `01_create_nodes.cypher` | Create nodes (entities) with labels and properties |
| `02_create_relationships.cypher` | Connect nodes with typed relationships — builds a mini knowledge graph |
| `03_read_queries.cypher` | MATCH queries — read data, find patterns |
| `04_filter_and_aggregate.cypher` | WHERE, COUNT, ORDER BY — filter and summarize |
| `05_multi_hop.cypher` | Multi-hop traversal — THIS is why graphs beat tables |
| `06_update_and_delete.cypher` | SET, DELETE, DETACH DELETE — modify the graph |
| `07_now_with_llm.cypher` | Bridge to Hour 2 — the LLM does exactly what you just did manually |

## Key Cypher Syntax

```
CREATE (n:Label {prop: "value"})          -- create a node
CREATE (a)-[:REL_TYPE]->(b)               -- create a relationship
MATCH (n:Label) RETURN n                  -- find nodes
MATCH (a)-[r]->(b) RETURN a, r, b        -- find relationships
WHERE n.prop = "value"                    -- filter
SET n.prop = "new value"                  -- update
DELETE r                                  -- delete relationship
MATCH (n) DETACH DELETE n                 -- delete everything
```
