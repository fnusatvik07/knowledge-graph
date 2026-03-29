# Temporal Knowledge Graphs

Facts change over time. A person's job title, a company's CEO, a country's population -- these are not permanent truths but **temporally scoped facts**. Temporal knowledge graphs add time awareness to the graph structure, enabling point-in-time queries and automatic invalidation of outdated information.

## Why Temporal Awareness Matters

Standard knowledge graphs treat facts as eternal:

```
(Alice, works_at, Acme Corp)
```

But what happens when Alice changes jobs? Without temporal metadata, you face a choice: delete the old fact (losing history) or keep both (creating contradictions). Neither is acceptable for production systems.

### Real-World Consequences of Ignoring Time

| Scenario | Without Temporal KG | With Temporal KG |
|----------|-------------------|-----------------|
| "Who is the CEO of Twitter?" | Returns outdated answer | Returns answer valid at query time |
| Compliance audit for 2024 | Cannot reconstruct past state | Point-in-time snapshot available |
| AI agent memory | Contradictory memories accumulate | Old memories are invalidated, history preserved |
| Drug interaction check | May reference withdrawn drugs | Filters to currently approved drugs |

## Temporal Metadata Model

Every edge (and optionally every node) in a temporal KG carries time-aware properties:

```
(Alice)-[:WORKS_AT {
    valid_from: 2022-03-15,
    valid_to: 2024-11-30,        # null if still current
    source: "linkedin_profile",
    confidence: 0.95,
    extracted_at: 2024-01-10
}]->(Acme Corp)

(Alice)-[:WORKS_AT {
    valid_from: 2024-12-01,
    valid_to: null,               # currently valid
    source: "company_announcement",
    confidence: 0.99,
    extracted_at: 2024-12-02
}]->(NewCo)
```

### Core Temporal Properties

| Property | Purpose | Example |
|----------|---------|---------|
| `valid_from` | When the fact became true in the real world | `2024-12-01` |
| `valid_to` | When the fact stopped being true (`null` = still current) | `2025-06-30` or `null` |
| `source` | Provenance of the fact | `"sec_filing_2024Q3"` |
| `confidence` | Extraction confidence score | `0.92` |
| `extracted_at` | When the fact was added to the KG | `2025-01-15` |
| `invalidated_by` | Reference to the fact that superseded this one | Edge ID or source |

### Two Types of Time

Temporal KGs distinguish between:

- **Valid time**: when the fact is true in reality (Alice worked at Acme from 2022 to 2024)
- **Transaction time**: when the fact was recorded in the system (we learned this on 2024-01-10)

Tracking both enables **bitemporal queries**: "What did we believe about Alice's employer as of January 2024?"

## Point-in-Time Queries

Temporal KGs support queries scoped to a specific moment:

```cypher
// Who does Alice work for RIGHT NOW?
MATCH (alice:Person {name: "Alice"})-[w:WORKS_AT]->(company)
WHERE w.valid_from <= datetime()
  AND (w.valid_to IS NULL OR w.valid_to > datetime())
RETURN company.name

// Who did Alice work for on 2023-06-15?
MATCH (alice:Person {name: "Alice"})-[w:WORKS_AT]->(company)
WHERE w.valid_from <= date("2023-06-15")
  AND (w.valid_to IS NULL OR w.valid_to > date("2023-06-15"))
RETURN company.name

// Full employment history
MATCH (alice:Person {name: "Alice"})-[w:WORKS_AT]->(company)
RETURN company.name, w.valid_from, w.valid_to
ORDER BY w.valid_from
```

### Python Helper for Temporal Queries

```python
from datetime import datetime, date
from typing import Optional

def query_at_time(
    tx,
    entity: str,
    relation: str,
    query_date: Optional[date] = None,
) -> list[dict]:
    """Query the KG for facts valid at a specific point in time."""
    if query_date is None:
        query_date = date.today()

    cypher = """
    MATCH (e:Entity {name: $entity})-[r]->(target)
    WHERE type(r) = $relation
      AND r.valid_from <= date($query_date)
      AND (r.valid_to IS NULL OR r.valid_to > date($query_date))
    RETURN target.name AS value,
           r.valid_from AS since,
           r.source AS source,
           r.confidence AS confidence
    """
    result = tx.run(
        cypher,
        entity=entity,
        relation=relation,
        query_date=query_date.isoformat(),
    )
    return [dict(record) for record in result]
```

## Invalidation of Outdated Facts

When new information arrives, the temporal KG must **invalidate** old facts rather than delete them:

```python
def update_fact(tx, subject: str, relation: str, old_object: str,
                new_object: str, source: str, effective_date: date):
    """Invalidate old fact and insert new one."""
    # Step 1: Close out the old fact
    tx.run("""
        MATCH (s:Entity {name: $subject})-[r:""" + relation + """]->(o:Entity {name: $old_object})
        WHERE r.valid_to IS NULL
        SET r.valid_to = date($effective_date),
            r.invalidated_by = $source
    """, subject=subject, old_object=old_object,
         effective_date=effective_date.isoformat(), source=source)

    # Step 2: Create the new fact
    tx.run("""
        MATCH (s:Entity {name: $subject}), (o:Entity {name: $new_object})
        CREATE (s)-[:""" + relation + """ {
            valid_from: date($effective_date),
            valid_to: null,
            source: $source,
            confidence: 0.95,
            extracted_at: datetime()
        }]->(o)
    """, subject=subject, new_object=new_object,
         effective_date=effective_date.isoformat(), source=source)
```

### Conflict Resolution Strategies

When two sources disagree about a fact:

1. **Recency wins**: the most recently reported fact takes priority
2. **Source authority**: trusted sources override less trusted ones
3. **Confidence threshold**: only accept facts above a confidence score
4. **LLM arbitration**: use an LLM to resolve ambiguous conflicts with reasoning

## Event-Driven KG Updates

Production temporal KGs are updated by event streams rather than batch processes:

```
Document Ingested → Extract Facts → Compare with Existing → Invalidate/Update → Emit Change Event
```

```python
async def process_document_event(document: dict):
    """Event handler for new document ingestion."""
    # Extract facts from the new document
    new_facts = await extract_facts(document["text"])

    for fact in new_facts:
        existing = query_current_fact(
            subject=fact["subject"],
            relation=fact["relation"],
        )

        if existing and existing["object"] != fact["object"]:
            # Fact has changed -- invalidate old, insert new
            update_fact(
                subject=fact["subject"],
                relation=fact["relation"],
                old_object=existing["object"],
                new_object=fact["object"],
                source=document["source"],
                effective_date=document["date"],
            )
            emit_event("fact_changed", fact)
        elif not existing:
            # New fact -- insert directly
            insert_fact(fact, source=document["source"])
            emit_event("fact_added", fact)
```

## Graphiti's Architecture: A Reference Implementation

Graphiti (by Zep) is the leading open-source temporal knowledge graph framework, designed specifically for AI agent memory systems.

### Core Design Principles

1. **Incremental updates**: new information is integrated episode by episode, not batch-rebuilt
2. **Temporal awareness**: every edge carries `valid_from` / `valid_to` timestamps
3. **Hybrid retrieval at query time**: combines three search methods with **no LLM calls during retrieval**

### Graphiti's Triple Search Strategy

```
Query
  ├── Semantic Search (vector similarity on node/edge embeddings)
  ├── BM25 Search (keyword matching for precise terms)
  └── Graph Traversal (follow relationships from matched nodes)
      │
      └── Merge + Rank → Return Context
```

The critical design decision: **no LLM is invoked at retrieval time**. LLMs are used during ingestion (to extract entities and relationships) but retrieval is purely algorithmic. This gives Graphiti:

- **Low latency**: retrieval completes in milliseconds
- **Predictable cost**: no per-query LLM charges for retrieval
- **Deterministic behavior**: same query always returns same results

### Graphiti Ingestion Flow

```
New Episode (text)
    ↓
LLM: Extract entities and relationships
    ↓
Deduplicate against existing nodes (embedding similarity + name matching)
    ↓
Resolve contradictions (invalidate outdated edges, set valid_to)
    ↓
Store nodes with embeddings + edges with temporal metadata
```

### Graphiti Usage Example

```python
from graphiti_core import Graphiti
from datetime import datetime

# Initialize
graphiti = Graphiti("bolt://localhost:7687", "neo4j", "password")

# Ingest episodes with timestamps
await graphiti.add_episode(
    name="team_update_q1",
    episode_body="Alice joined the ML team in January. She reports to Bob.",
    source_description="team_standup",
    reference_time=datetime(2025, 1, 15),
)

await graphiti.add_episode(
    name="team_update_q3",
    episode_body="Alice transferred to the Platform team in July. She now reports to Carol.",
    source_description="team_standup",
    reference_time=datetime(2025, 7, 1),
)

# Query: Graphiti returns temporally-aware results
# The January "reports to Bob" edge is now invalidated
results = await graphiti.search("Who does Alice report to?")
# Returns: Carol (with valid_from=2025-07-01)
```

## Application: AI Agent Memory Systems

Temporal KGs are the backbone of long-term memory for AI agents. Unlike simple vector stores that accumulate contradictory memories, temporal KGs maintain a **consistent, evolving world model**.

### Agent Memory Architecture

```
Agent Interaction
    ↓
Short-term memory (conversation buffer)
    ↓
Temporal KG (long-term structured memory)
    ├── User preferences (may change)
    ├── Project state (evolves)
    ├── Decisions made (historical record)
    └── Facts learned (with validity periods)
```

### Why Temporal KGs Beat Vector Stores for Agent Memory

| Aspect | Vector Store Memory | Temporal KG Memory |
|--------|-------------------|-------------------|
| Contradictions | Accumulate silently | Detected and resolved |
| History | All memories equal | Time-ordered with validity |
| Reasoning | Nearest-neighbor only | Multi-hop traversal |
| Updates | Append-only | Invalidate + insert |
| Query: "current state" | May return stale facts | Filters to valid_to IS NULL |

## Key Takeaways

- Facts change over time -- temporal KGs track validity periods instead of overwriting or accumulating contradictions
- Every edge carries `valid_from`, `valid_to`, `source`, and `confidence` metadata
- Point-in-time queries reconstruct the state of knowledge at any historical moment
- Invalidation preserves history while marking outdated facts as superseded
- Graphiti demonstrates the production pattern: LLM at ingestion, algorithmic retrieval with semantic + BM25 + graph traversal
- Temporal KGs are essential for AI agent memory systems that need consistent, evolving world models
