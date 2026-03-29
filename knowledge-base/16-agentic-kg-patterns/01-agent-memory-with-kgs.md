# Agent Memory with Knowledge Graphs

## Why Vector Stores Alone Are Not Enough

Vector stores (Pinecone, Weaviate, Chroma) excel at semantic similarity search but
fundamentally lack structure. When an agent needs memory, it needs more than "find
similar text" — it needs to understand relationships, track changes over time, and
reason over connected facts.

### Limitations of Pure Vector Memory

**No relationships.** A vector store can tell you that "Alice works at Acme" and
"Acme is headquartered in London" are semantically related. But it cannot traverse
the relationship: "Where does Alice's employer have its headquarters?" requires
a join that vectors cannot express.

**No temporal reasoning.** Vectors represent the state of text at embedding time.
If Alice leaves Acme and joins Beta Corp, the old vector still exists. There is no
native mechanism to mark facts as superseded, expired, or versioned.

**No contradiction detection.** Two documents might state conflicting facts. Vector
stores retrieve both, leaving the agent to reconcile them in-context every single
time. A KG can flag the contradiction at write time and resolve it once.

**No structured aggregation.** "How many people work at Acme?" requires counting
entities with a specific relationship. Vector similarity cannot count.

### What KGs Add

| Capability | Vector Store | Knowledge Graph |
|------------|-------------|-----------------|
| Semantic search | Native | Via embeddings or hybrid |
| Relationship traversal | Cannot | Native (Cypher, SPARQL) |
| Temporal versioning | Manual | Temporal properties on edges |
| Contradiction detection | Cannot | Constraint checks at write time |
| Aggregation / counting | Cannot | Native queries |
| Provenance tracking | Metadata field | First-class edge properties |
| Explainability | "These chunks are similar" | "This path connects A to B" |

The practical answer is not "KG instead of vectors" but "KG and vectors together."
Hybrid retrieval — vector similarity for fuzzy matches, graph traversal for precise
relationships — is the pattern that works in production.

---

## KG Memory Architecture

### Core Data Model

An agent's knowledge graph memory has three layers:

```
Layer 1: Entities (nodes)
    - Facts the agent knows
    - Each entity has: name, type, description, metadata
    - Example: (Alice, PERSON, "Software engineer at Acme")

Layer 2: Relations (edges)
    - Connections between facts
    - Each relation has: type, confidence, source, timestamp
    - Example: (Alice)-[WORKS_AT {since: 2023, confidence: 0.95}]->(Acme)

Layer 3: Temporal Metadata
    - When things were true (valid_from, valid_to)
    - When the agent learned them (created_at, updated_at)
    - Who/what said them (source_document, extraction_model)
```

### Schema Design for Agent Memory

```cypher
// Entity node
CREATE (e:Entity {
    id: randomUUID(),
    name: "Alice",
    type: "PERSON",
    description: "Software engineer specializing in distributed systems",
    created_at: datetime(),
    updated_at: datetime(),
    confidence: 0.95,
    source: "conversation_2024_01_15"
})

// Relationship with temporal and provenance metadata
CREATE (alice)-[:WORKS_AT {
    since: date("2023-06-01"),
    confidence: 0.9,
    source: "linkedin_profile",
    valid_from: datetime(),
    valid_to: null,  // null means "still true"
    extraction_model: "gpt-4o"
}]->(acme)

// Session metadata node
CREATE (s:Session {
    id: "session_abc123",
    started_at: datetime(),
    ended_at: null,
    user_id: "user_42",
    topic: "project planning"
})
```

### Reading from Memory

When an agent receives a query, it should search memory in multiple ways:

1. **Entity lookup**: Exact match on entity names mentioned in the query.
2. **Neighborhood expansion**: For each matched entity, traverse 1-2 hops to find
   related context.
3. **Temporal filter**: Prefer recent facts; flag stale ones.
4. **Community context**: Which cluster does this entity belong to? Retrieve the
   community summary for broader context.

```python
def retrieve_memory(query: str, kg_client, top_k: int = 10):
    """Multi-strategy memory retrieval."""
    results = []

    # Strategy 1: Entity name matching
    entities = extract_entities_from_query(query)
    for entity in entities:
        node = kg_client.find_entity(entity)
        if node:
            neighbors = kg_client.get_neighbors(node, max_hops=2)
            results.extend(neighbors)

    # Strategy 2: Semantic search over entity descriptions
    similar = kg_client.vector_search(query, top_k=top_k)
    results.extend(similar)

    # Strategy 3: Community summaries for broad context
    communities = kg_client.get_relevant_communities(query)
    results.extend(communities)

    # Deduplicate and rank
    return rank_and_deduplicate(results)
```

---

## Session Persistence Patterns

### Checkpointing LangGraph State to Neo4j

LangGraph provides built-in checkpointing for agent state persistence. The default
uses SQLite or Postgres, but a Neo4j-backed checkpointer stores state as a graph,
enabling relationship-aware state queries.

Reference: https://langchain-ai.github.io/langgraph/concepts/persistence/

**Pattern: Save agent state after each tool call.**

```python
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

# Define state schema
class AgentState(TypedDict):
    messages: list
    kg_entities_found: list
    kg_queries_executed: list
    current_reasoning: str

# Build graph with checkpointing
workflow = StateGraph(AgentState)
workflow.add_node("reason", reason_node)
workflow.add_node("query_kg", query_kg_node)
workflow.add_node("update_kg", update_kg_node)

# Compile with checkpointer
checkpointer = MemorySaver()  # Or Neo4jSaver for production
app = workflow.compile(checkpointer=checkpointer)

# Invoke with thread_id for session tracking
config = {"configurable": {"thread_id": "session_abc123"}}
result = app.invoke({"messages": [user_message]}, config)
```

**Resuming from graph state.** When a user returns, load the previous session's
checkpoint and continue the conversation with full context:

```python
# Resume a previous session
config = {"configurable": {"thread_id": "session_abc123"}}
state = app.get_state(config)
# state.values contains the full agent state from the last checkpoint

# Continue the conversation
result = app.invoke({"messages": [new_user_message]}, config)
```

### Neo4j-Backed Session Storage

For production systems, persist sessions as graph structures:

```cypher
// Session node links to all entities discovered during that session
CREATE (s:Session {id: $session_id, started_at: datetime()})

// Link entities found during this session
MATCH (e:Entity {id: $entity_id})
CREATE (s)-[:DISCOVERED {at: datetime(), confidence: $conf}]->(e)

// Link sessions that share entities (cross-session connections)
MATCH (s1:Session)-[:DISCOVERED]->(e)<-[:DISCOVERED]-(s2:Session)
WHERE s1 <> s2
MERGE (s1)-[:SHARES_CONTEXT]->(s2)
```

---

## Memory Summarization

### The Growth Problem

An agent that continuously builds a KG will eventually create a graph too large
to retrieve from efficiently. After thousands of sessions, the graph may contain
hundreds of thousands of entities and millions of edges.

### Summarization Strategy

**Community detection + summarization.** Use graph algorithms (Louvain, Leiden) to
find communities, then summarize each community into a single summary node.

```python
def summarize_old_communities(kg_client, llm, max_age_days: int = 30):
    """Summarize communities older than max_age_days."""

    # Find old, large communities
    communities = kg_client.run_cypher("""
        CALL gds.louvain.stream('memory_graph')
        YIELD nodeId, communityId
        WITH communityId, collect(gds.util.asNode(nodeId)) AS members
        WHERE size(members) > 10
        AND all(m IN members WHERE m.updated_at < datetime() - duration({days: $max_age}))
        RETURN communityId, members
    """, {"max_age": max_age_days})

    for community_id, members in communities:
        # Build a text summary of all entities and relations in the community
        descriptions = [f"{m['name']}: {m['description']}" for m in members]
        summary_text = llm.summarize("\n".join(descriptions))

        # Create summary node, link to community, prune detailed nodes
        kg_client.run_cypher("""
            CREATE (s:CommunitySummary {
                community_id: $cid,
                summary: $summary,
                member_count: $count,
                created_at: datetime()
            })
        """, {"cid": community_id, "summary": summary_text, "count": len(members)})
```

### Pruning Rules

1. **Never prune high-confidence recent facts** — they are active memory.
2. **Summarize before pruning** — always create a community summary first.
3. **Keep provenance** — store which entities were pruned into which summary.
4. **Graduated decay** — reduce detail gradually: full entities -> summaries -> archive.

---

## Cross-Session Accumulation

### Agents That Get Smarter Over Time

The most powerful pattern: an agent whose KG grows across sessions, making it
progressively better at answering questions.

**Session 1:** User asks about machine learning. Agent builds entities for
supervised learning, neural networks, gradient descent.

**Session 5:** User asks about deep learning. Agent already has "neural networks"
in memory, connects new deep learning entities to existing graph.

**Session 20:** User asks "How does attention relate to neural networks?" Agent
traverses the rich graph built over 20 sessions and gives a detailed answer
that would be impossible from a single RAG retrieval.

### Implementation Pattern

```python
class AccumulativeAgent:
    """Agent that builds KG memory across sessions."""

    def __init__(self, kg_client, llm):
        self.kg = kg_client
        self.llm = llm

    def process_query(self, query: str, session_id: str):
        # 1. Retrieve existing memory
        context = self.retrieve_memory(query)

        # 2. Generate response using memory + LLM
        response = self.llm.generate(query, context=context)

        # 3. Extract new entities and relations from the conversation
        new_facts = self.extract_facts(query, response)

        # 4. Merge new facts into KG (deduplicate, resolve conflicts)
        self.merge_facts(new_facts, session_id)

        # 5. Update session metadata
        self.update_session(session_id, query, response)

        return response

    def merge_facts(self, facts, session_id):
        """Merge new facts, handling duplicates and conflicts."""
        for fact in facts:
            existing = self.kg.find_similar_entity(fact.entity)
            if existing:
                # Update existing entity with new information
                self.kg.merge_entity(existing, fact, source=session_id)
            else:
                # Create new entity
                self.kg.create_entity(fact, source=session_id)
```

---

## Graphiti's Approach: Hybrid Retrieval Without LLM at Read Time

Graphiti (by Zep) demonstrates a production pattern where the KG is structured
so that retrieval does not require an LLM call — achieving 300ms P95 latency.

### Key Design Decisions

1. **Pre-computed summaries**: Entity and community summaries are generated at
   write time (when data is ingested), not at read time.
2. **Indexed embeddings**: Entity descriptions are embedded and indexed for
   fast vector search.
3. **Typed edges with scores**: Relationships carry confidence scores and
   timestamps, enabling efficient filtering without LLM reasoning.
4. **No LLM at read time**: The retrieval path is pure database queries
   (graph traversal + vector similarity). The LLM is only used during
   ingestion for entity extraction and summarization.

### Performance Implications

| Operation | With LLM at Read | Without LLM at Read |
|-----------|-----------------|---------------------|
| Latency (P50) | 800-2000ms | 50-150ms |
| Latency (P95) | 3000-8000ms | 200-300ms |
| Cost per query | $0.01-0.05 | $0.0001 (DB only) |
| Consistency | Varies by LLM | Deterministic |

The tradeoff: more compute at write time (LLM extraction, summarization,
embedding) in exchange for fast, cheap, deterministic reads.

---

## Implementation Patterns with LangGraph + Neo4j

### Pattern 1: Memory-Augmented ReAct Agent

```python
from langgraph.graph import StateGraph, END

class MemoryState(TypedDict):
    messages: list
    memory_context: str
    new_facts: list

def retrieve_memory(state: MemoryState) -> MemoryState:
    """Retrieve relevant memory from KG before reasoning."""
    query = state["messages"][-1].content
    context = kg_client.hybrid_search(query, top_k=10)
    return {"memory_context": format_context(context)}

def reason_with_memory(state: MemoryState) -> MemoryState:
    """Generate response using retrieved memory."""
    prompt = f"Memory context:\n{state['memory_context']}\n\nQuery: {state['messages'][-1].content}"
    response = llm.invoke(prompt)
    return {"messages": state["messages"] + [response]}

def extract_and_store(state: MemoryState) -> MemoryState:
    """Extract facts from conversation and store in KG."""
    facts = extract_facts(state["messages"])
    for fact in facts:
        kg_client.upsert_entity(fact)
    return {"new_facts": facts}

# Build the graph
workflow = StateGraph(MemoryState)
workflow.add_node("retrieve", retrieve_memory)
workflow.add_node("reason", reason_with_memory)
workflow.add_node("store", extract_and_store)
workflow.add_edge("retrieve", "reason")
workflow.add_edge("reason", "store")
workflow.add_edge("store", END)
workflow.set_entry_point("retrieve")
```

### Pattern 2: Selective Memory Updates

Not every conversation turn produces new facts worth storing. Use a relevance
gate to decide:

```python
def should_update_memory(state: MemoryState) -> str:
    """Decide if the conversation produced new facts worth storing."""
    last_exchange = state["messages"][-2:]  # User query + agent response
    has_new_info = llm.classify(
        f"Does this exchange contain new factual information?\n{last_exchange}",
        options=["yes", "no"]
    )
    return "store" if has_new_info == "yes" else "end"

workflow.add_conditional_edges("reason", should_update_memory, {
    "store": "store",
    "end": END,
})
```

---

## Key Takeaways

1. **KGs complement vector stores** — use both for robust agent memory. Vectors
   for fuzzy semantic search, graphs for structured relationship traversal.

2. **Temporal metadata is essential** — facts change. Every entity and relation
   needs `created_at`, `updated_at`, `valid_from`, `valid_to`, and `source`.

3. **Summarize before you prune** — community detection + LLM summarization
   lets you compress old memories without losing them entirely.

4. **Avoid LLM at read time** — pre-compute summaries and embeddings during
   ingestion. This is the difference between 300ms and 3000ms latency.

5. **Cross-session accumulation is the payoff** — an agent that builds a richer
   KG over time gives progressively better answers that no stateless RAG system
   can match.

6. **LangGraph checkpointers** provide the mechanism for session persistence.
   Pair with Neo4j for graph-native state storage.
