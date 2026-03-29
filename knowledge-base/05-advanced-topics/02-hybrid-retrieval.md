# Hybrid Retrieval: Vector Search + Graph Traversal

Hybrid retrieval combines vector similarity search with graph traversal to answer questions that neither approach handles well alone. As of 2026, this is the **production default** for Graph RAG systems -- pure vector search or pure graph traversal are now considered incomplete approaches.

## Why Neither Approach Is Sufficient Alone

### Vector Search Limitations

Vector search excels at finding semantically similar content but fails when the answer requires **structural reasoning**:

```
Query: "What side effects does the drug prescribed for Patient X's condition cause?"

Vector search finds: chunks about side effects, chunks about Patient X
Vector search misses: the chain Patient X → diagnosed_with → Condition → treated_by → Drug → causes → Side Effect
```

Vector search also struggles with:
- **Multi-hop questions** that require traversing relationships
- **Precise filtering** (e.g., "all drugs approved after 2023")
- **Negation** ("which entities are NOT connected to X")

### Graph Traversal Limitations

Graph traversal follows explicit relationships but cannot handle **fuzzy or semantic queries**:

```
Query: "Tell me about Einstein's impact on modern physics"

Graph traversal finds: Einstein → developed → Relativity (exact edges)
Graph traversal misses: contextual paragraphs discussing influence, legacy, philosophical impact
```

Graph traversal also struggles with:
- **Ambiguous entities** (which "John Smith"?)
- **Implicit relationships** not encoded as edges
- **Open-ended exploration** without a clear starting node

### The Hybrid Advantage

| Query Type | Vector | Graph | Hybrid |
|-----------|--------|-------|--------|
| Semantic similarity | Strong | Weak | Strong |
| Multi-hop reasoning | Weak | Strong | Strong |
| Precise filtering | Weak | Strong | Strong |
| Fuzzy + structural | Weak | Weak | Strong |

## Architecture Patterns

### Pattern 1: Parallel Retrieval with Fusion

The simplest and most common pattern. Run both retrieval paths independently and merge results.

```
                    ┌─── Vector Search ───────┐
                    │   (embed query, ANN)     │
User Query ────────┤                           ├──→ Fusion → LLM → Answer
                    │                           │
                    └─── Graph Traversal ──────┘
                        (entity extract, Cypher)
```

**When to use**: general-purpose systems, when you cannot predict query type in advance.

```python
async def parallel_retrieval(query: str) -> list[dict]:
    """Run vector and graph retrieval in parallel."""
    # Extract entities for graph lookup
    entities = extract_entities(query)

    # Run both retrievals concurrently
    vector_results, graph_results = await asyncio.gather(
        vector_search(query, top_k=10),
        graph_traversal(entities, max_hops=2),
    )

    # Fuse results
    fused = reciprocal_rank_fusion(vector_results, graph_results)
    return fused[:15]
```

### Pattern 2: Sequential Retrieval (Graph-Guided Vector Search)

Use graph traversal first to identify relevant subgraph, then run vector search scoped to that subgraph.

```
User Query → Entity Extraction → Graph Traversal → Subgraph
                                                       ↓
                                              Vector Search (scoped)
                                                       ↓
                                                   LLM → Answer
```

**When to use**: when you need precise, contextual retrieval and can afford the latency of two sequential steps.

```python
def graph_guided_vector_search(query: str) -> list[dict]:
    """Use graph context to scope vector search."""
    # Step 1: identify relevant entities and subgraph
    entities = extract_entities(query)
    subgraph_node_ids = traverse_graph(entities, max_hops=2)

    # Step 2: vector search filtered to subgraph nodes
    results = vector_store.similarity_search(
        query,
        top_k=10,
        filter={"node_id": {"$in": subgraph_node_ids}},
    )
    return results
```

### Pattern 3: Vector-Guided Graph Traversal

Use vector search to find entry points, then expand via graph traversal.

```
User Query → Vector Search → Top-K Nodes → Graph Expansion → Context → LLM
```

**When to use**: when entity extraction is unreliable or the query is too vague for direct graph lookup.

```python
def vector_guided_graph_traversal(query: str) -> list[dict]:
    """Use vector similarity to find graph entry points."""
    # Step 1: find semantically relevant nodes
    seed_nodes = vector_search(query, top_k=5)

    # Step 2: expand each seed node via graph traversal
    expanded_context = []
    for node in seed_nodes:
        neighbors = graph.traverse(
            start=node["id"],
            max_hops=2,
            max_nodes=20,
        )
        expanded_context.extend(neighbors)

    return deduplicate(expanded_context)
```

## Fusion Strategies

### Reciprocal Rank Fusion (RRF)

RRF combines ranked lists without requiring comparable scores. Each item's fused score is based on its rank position in each list:

```
RRF_score(item) = Σ  1 / (k + rank_i(item))
```

Where `k` is a constant (typically 60) that dampens the effect of high rankings.

```python
def reciprocal_rank_fusion(
    *result_lists: list[dict],
    k: int = 60,
) -> list[dict]:
    """Fuse multiple ranked result lists using RRF."""
    scores = {}

    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            item_id = item["id"]
            if item_id not in scores:
                scores[item_id] = {"item": item, "score": 0.0}
            scores[item_id]["score"] += 1.0 / (k + rank)

    # Sort by fused score
    fused = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [entry["item"] for entry in fused]
```

**Pros**: simple, no tuning required, works across heterogeneous score distributions.

### Weighted Merging

Assign explicit weights to each retrieval source based on query characteristics:

```python
def weighted_fusion(
    vector_results: list[dict],
    graph_results: list[dict],
    query_type: str,
) -> list[dict]:
    """Weight sources based on query type."""
    weights = {
        "factual":    {"vector": 0.3, "graph": 0.7},
        "semantic":   {"vector": 0.7, "graph": 0.3},
        "multi_hop":  {"vector": 0.2, "graph": 0.8},
        "default":    {"vector": 0.5, "graph": 0.5},
    }
    w = weights.get(query_type, weights["default"])

    for r in vector_results:
        r["fused_score"] = r["score"] * w["vector"]
    for r in graph_results:
        r["fused_score"] = r["score"] * w["graph"]

    combined = vector_results + graph_results
    combined.sort(key=lambda x: x["fused_score"], reverse=True)
    return deduplicate(combined)
```

### LLM-Based Reranking

Use an LLM to rerank the merged candidate set. More expensive but handles nuance:

```python
def llm_rerank(query: str, candidates: list[dict], top_k: int = 10) -> list[dict]:
    """Use an LLM to rerank retrieval candidates."""
    prompt = f"""Given the query: "{query}"

Rank these candidates by relevance (most relevant first).
Return only the IDs in order.

Candidates:
{format_candidates(candidates[:20])}
"""
    response = llm.invoke(prompt)
    ranked_ids = parse_ranked_ids(response)
    return [c for c in candidates if c["id"] in ranked_ids[:top_k]]
```

## Implementation with Neo4j Vector Index + Cypher

Neo4j natively supports vector indexes alongside its graph engine, making it ideal for hybrid retrieval.

### Setting Up the Vector Index

```cypher
-- Create a vector index on Document nodes
CREATE VECTOR INDEX document_embeddings IF NOT EXISTS
FOR (d:Document)
ON (d.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
  }
}
```

### Hybrid Query in a Single Cypher Statement

```cypher
// Hybrid: vector search + graph expansion in one query
WITH "What treatments exist for diabetes?" AS query

// Step 1: Vector search for semantically relevant nodes
CALL db.index.vector.queryNodes('document_embeddings', 10, $query_embedding)
YIELD node AS doc, score AS vector_score

// Step 2: Graph expansion from vector results
MATCH (doc)-[:MENTIONS]->(entity:Entity)
OPTIONAL MATCH (entity)-[:RELATED_TO|TREATS|CAUSES*1..2]-(connected:Entity)

// Step 3: Collect and return fused context
WITH doc, vector_score, collect(DISTINCT connected) AS graph_context
RETURN doc.text AS text,
       vector_score,
       [n IN graph_context | n.name] AS related_entities
ORDER BY vector_score DESC
LIMIT 15
```

### Python Implementation

```python
from neo4j import GraphDatabase
from openai import OpenAI

openai_client = OpenAI()
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

def hybrid_search(query: str, top_k: int = 10) -> list[dict]:
    """Hybrid retrieval combining Neo4j vector index and Cypher traversal."""
    # Embed the query
    response = openai_client.embeddings.create(
        input=query, model="text-embedding-3-small"
    )
    query_embedding = response.data[0].embedding

    cypher = """
    CALL db.index.vector.queryNodes('document_embeddings', $top_k, $embedding)
    YIELD node, score

    // Expand via graph relationships
    MATCH (node)-[:MENTIONS]->(e:Entity)
    OPTIONAL MATCH path = (e)-[*1..2]-(related:Entity)

    WITH node, score, collect(DISTINCT {
        name: related.name,
        type: labels(related)[0],
        path: [r IN relationships(path) | type(r)]
    }) AS graph_context

    RETURN node.text AS text,
           node.source AS source,
           score AS vector_score,
           graph_context
    ORDER BY score DESC
    """

    with driver.session() as session:
        results = session.run(cypher, embedding=query_embedding, top_k=top_k)
        return [dict(record) for record in results]
```

## Production Considerations

### Latency Budget

| Component | Typical Latency | Notes |
|-----------|----------------|-------|
| Query embedding | 20-50ms | API call to embedding model |
| Vector search | 5-20ms | ANN index lookup |
| Graph traversal (2 hops) | 10-50ms | Depends on graph density |
| Fusion | 1-5ms | In-memory ranking |
| LLM reranking | 200-500ms | Optional, adds quality |
| **Total (without reranking)** | **36-125ms** | Fast enough for real-time |

### When to Use Which Pattern

- **Parallel + RRF**: best default, fast, robust
- **Graph-guided vector**: when you need precision and have reliable entity extraction
- **Vector-guided graph**: when queries are vague or entities are ambiguous
- **LLM reranking**: when quality matters more than latency (e.g., research, complex analysis)

### Monitoring Hybrid Retrieval

Track these metrics to tune your hybrid system:
- **Retrieval source attribution**: what percentage of final context came from vector vs graph
- **Query classification accuracy**: if using weighted fusion, is the classifier correct
- **Latency per component**: identify bottlenecks
- **Answer quality by source**: does adding graph context actually improve answers

## Key Takeaways

- Neither vector search nor graph traversal alone covers all query types -- hybrid retrieval combines both
- Three main architecture patterns: parallel with fusion, graph-guided vector, and vector-guided graph
- Reciprocal rank fusion is the simplest and most robust fusion strategy
- Neo4j vector indexes enable hybrid queries in a single Cypher statement
- Production systems should monitor which retrieval source contributes to answer quality
- Hybrid retrieval adds modest latency (under 125ms without reranking) and is viable for real-time applications
