# The Graph Building Pipeline

This section ties together entity extraction, relationship extraction, and ontology design into a complete end-to-end pipeline for constructing a knowledge graph from unstructured text.

## Pipeline Overview

```
Documents
    │
    ▼
┌─────────────┐
│  1. Load &   │
│    Chunk     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  2. Extract  │  ← LLM calls (most expensive step)
│  Entities &  │
│  Relations   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  3. Resolve  │  ← Merge duplicate entities
│  & Merge     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  4. Build    │  ← Construct the graph data structure
│    Graph     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  5. Store    │  ← NetworkX (prototyping) or Neo4j (production)
│  & Index     │
└─────────────┘
```

## Step 1: Load and Chunk Documents

```python
from shared.document_loader import load_text_files, chunk_text

documents = load_text_files("data/articles/")
all_chunks = []
for doc in documents:
    chunks = chunk_text(doc["content"], chunk_size=1500, overlap=200)
    for i, chunk in enumerate(chunks):
        all_chunks.append({
            "chunk_id": f"{doc['title']}_{i}",
            "text": chunk,
            "source": doc["source"],
        })
```

**Chunking considerations**:
- Larger chunks (1500-2000 chars) capture more context per LLM call → fewer calls → lower cost
- Overlap (200 chars) prevents entities near boundaries from being split
- Track chunk provenance (source document, chunk index) for traceability

## Step 2: Extract Entities and Relationships

Process each chunk through the LLM:

```python
import json
from shared.llm_clients import chat_completion

EXTRACTION_PROMPT = """Extract all entities and relationships from this text.

Entity types: PERSON, ORGANIZATION, LOCATION, CONCEPT, TECHNOLOGY, EVENT
Relationship types: DEVELOPED, WORKS_AT, LOCATED_IN, PART_OF, USES, RELATED_TO

Return JSON:
{
  "entities": [{"name": "...", "type": "...", "description": "..."}],
  "relationships": [{"source": "...", "target": "...", "type": "...", "description": "..."}]
}

Text: {text}"""

all_entities = []
all_relationships = []

for chunk in all_chunks:
    result = chat_completion(
        EXTRACTION_PROMPT.format(text=chunk["text"]),
        response_format={"type": "json_object"}
    )
    data = json.loads(result)
    # Tag each entity/relationship with its source chunk
    for entity in data["entities"]:
        entity["source_chunk"] = chunk["chunk_id"]
        all_entities.append(entity)
    for rel in data["relationships"]:
        rel["source_chunk"] = chunk["chunk_id"]
        all_relationships.append(rel)
```

## Step 3: Resolve and Merge Entities

The same entity may appear across multiple chunks with slightly different names:

```python
def resolve_entities(entities: list[dict]) -> dict:
    """Merge entities by normalized name. Returns {canonical_name: merged_entity}."""
    merged = {}
    for entity in entities:
        key = entity["name"].strip().lower()
        if key in merged:
            # Merge descriptions
            existing = merged[key]
            if entity["description"] not in existing["description"]:
                existing["description"] += f" {entity['description']}"
            existing["mentions"] += 1
        else:
            merged[key] = {
                "name": entity["name"],
                "type": entity["type"],
                "description": entity["description"],
                "mentions": 1,
            }
    return merged
```

For more sophisticated resolution (handling "Einstein" vs "Albert Einstein"), use an LLM-based merge step or embedding similarity.

## Step 4: Build the Graph

```python
import networkx as nx

def build_graph(entities: dict, relationships: list[dict]) -> nx.DiGraph:
    G = nx.DiGraph()

    # Add nodes
    for key, entity in entities.items():
        G.add_node(
            entity["name"],
            type=entity["type"],
            description=entity["description"],
            mentions=entity["mentions"],
        )

    # Add edges
    for rel in relationships:
        source = rel["source"].strip()
        target = rel["target"].strip()
        if source in [e["name"] for e in entities.values()] and \
           target in [e["name"] for e in entities.values()]:
            G.add_edge(
                source, target,
                relation_type=rel["type"],
                description=rel["description"],
            )

    return G
```

## Step 5: Store and Index

### For Prototyping (NetworkX)
```python
# Save to file
nx.write_graphml(G, "output/knowledge_graph.graphml")

# Basic stats
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Connected components: {nx.number_weakly_connected_components(G)}")
```

### For Production (Neo4j)
```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

with driver.session() as session:
    for node_name, attrs in G.nodes(data=True):
        session.run(
            "MERGE (n:Entity {name: $name}) SET n.type = $type, n.description = $desc",
            name=node_name, type=attrs["type"], desc=attrs["description"]
        )
    for source, target, attrs in G.edges(data=True):
        session.run(
            """MATCH (a:Entity {name: $source}), (b:Entity {name: $target})
               MERGE (a)-[r:RELATES_TO {type: $rel_type}]->(b)
               SET r.description = $desc""",
            source=source, target=target,
            rel_type=attrs["relation_type"], desc=attrs["description"]
        )
```

## Pipeline Performance Tips

| Optimization | Impact |
|-------------|--------|
| Batch multiple chunks per LLM call | 2-3x fewer API calls |
| Use `gpt-4o-mini` for extraction | 10x cheaper, ~90% quality |
| Async LLM calls | 3-5x faster processing |
| Skip boilerplate chunks | 10-20% fewer calls |
| Cache extractions | Avoid re-processing unchanged docs |

## What Comes Next

After building the graph:
- **Project 1** uses this pipeline with NetworkX and visualization
- **Project 2** replaces this manual pipeline with Microsoft GraphRAG / LightRAG
- **Project 3** adds temporal metadata, Neo4j storage, and agentic construction
