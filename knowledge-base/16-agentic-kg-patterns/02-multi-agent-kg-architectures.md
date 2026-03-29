# Multi-Agent KG Architectures

## Overview

When a single agent is not enough — because the task is too complex, requires
multiple perspectives, or needs parallel processing — multiple agents must
coordinate. A shared knowledge graph becomes the coordination layer: the single
source of truth that all agents read from and write to.

This section covers architectural patterns for multi-agent KG systems, from
simple sequential pipelines to complex concurrent topologies.

---

## Pattern 1: Sequential Pipeline

### Description

Agents execute in a fixed order. Each agent reads the KG, performs its task,
writes results back to the KG, and the next agent picks up from there.

```
[Extractor] --> KG --> [Resolver] --> KG --> [Enricher] --> KG --> [Validator]
```

This is the simplest multi-agent pattern and is what Project 3 (KG pipeline)
implements.

### Strengths

- **Simple to reason about.** Each agent has a clear input and output.
- **No concurrency issues.** Only one agent writes at a time.
- **Easy to debug.** You can inspect the KG state between each agent.
- **Deterministic.** Same input produces same output (given same LLM seed).

### Limitations

- **Slow.** Total time = sum of all agent times. No parallelism.
- **Rigid.** The pipeline order is fixed. Cannot adapt to different document types.
- **No feedback loops.** If the validator finds errors, it cannot tell the extractor
  to re-extract. It can only flag issues.
- **Single point of failure.** If one agent fails, the entire pipeline stalls.

### Implementation

```python
from langgraph.graph import StateGraph, END

class PipelineState(TypedDict):
    document: str
    entities: list
    relations: list
    resolved_entities: list
    validation_report: dict

def extract(state):
    """Agent 1: Extract entities and relations from document."""
    entities, relations = extractor_llm.extract(state["document"])
    return {"entities": entities, "relations": relations}

def resolve(state):
    """Agent 2: Resolve duplicates and link to existing KG entities."""
    resolved = resolver_llm.resolve(state["entities"], kg_client)
    return {"resolved_entities": resolved}

def validate(state):
    """Agent 3: Validate extracted facts against known constraints."""
    report = validator_llm.validate(state["resolved_entities"], state["relations"])
    return {"validation_report": report}

workflow = StateGraph(PipelineState)
workflow.add_node("extract", extract)
workflow.add_node("resolve", resolve)
workflow.add_node("validate", validate)
workflow.add_edge("extract", "resolve")
workflow.add_edge("resolve", "validate")
workflow.add_edge("validate", END)
workflow.set_entry_point("extract")
```

---

## Pattern 2: Parallel Agents with Shared Graph

### Description

Multiple agents read from and write to the same KG concurrently. Each agent
handles a different aspect of the task (e.g., one extracts entities, another
extracts relations, a third enriches from external sources).

```
[Entity Extractor] ----\
                        \
[Relation Extractor] ----> Shared KG
                        /
[Web Enricher] --------/
```

### Concurrency Challenges

**Read-after-write consistency.** Agent A creates entity "Apple Inc." Agent B,
running concurrently, creates "Apple" as a separate entity. Now there are
duplicates that need post-hoc resolution.

**Write conflicts.** Two agents try to update the same entity simultaneously.
One writes confidence=0.8, the other writes confidence=0.6. Which wins?

**Ordering dependencies.** The enricher should only enrich entities that have
been extracted. But with parallel execution, it might run before extraction
is complete.

### Conflict Resolution Strategies

**Strategy 1: Optimistic locking.** Each agent reads the entity version before
writing. If the version has changed when it tries to write, it retries.

```cypher
// Read with version
MATCH (e:Entity {id: $id})
RETURN e.version AS version, e

// Write with version check (optimistic lock)
MATCH (e:Entity {id: $id, version: $expected_version})
SET e.confidence = $new_confidence, e.version = e.version + 1
RETURN e
```

**Strategy 2: Append-only with reconciliation.** Agents never update existing
facts. They only append new observations. A reconciliation agent periodically
merges observations into canonical facts.

```cypher
// Agent appends an observation (never modifies existing)
CREATE (o:Observation {
    entity_name: "Apple",
    entity_type: "ORGANIZATION",
    confidence: 0.9,
    source_agent: "extractor_v2",
    observed_at: datetime()
})

// Reconciliation agent merges observations
MATCH (o:Observation {entity_name: $name})
WITH $name AS name, collect(o) AS observations
WITH name, observations,
     avg([o IN observations | o.confidence]) AS avg_conf,
     max([o IN observations | o.observed_at]) AS latest
MERGE (e:Entity {name: name})
SET e.confidence = avg_conf, e.updated_at = latest
```

**Strategy 3: Agent-specific namespaces.** Each agent writes to its own subgraph.
A merge agent combines subgraphs into a canonical graph.

```cypher
// Agent A writes to its namespace
CREATE (e:Entity:AgentA_Output {name: "Apple", type: "ORGANIZATION"})

// Merge agent combines namespaces
MATCH (a:AgentA_Output {name: $name}), (b:AgentB_Output {name: $name})
MERGE (canonical:Entity {name: $name})
SET canonical.type = CASE
    WHEN a.confidence > b.confidence THEN a.type
    ELSE b.type
END
```

### LangGraph Implementation

```python
from langgraph.graph import StateGraph, END
import asyncio

class ParallelState(TypedDict):
    document: str
    entity_results: list
    relation_results: list
    enrichment_results: list
    merged_kg: dict

async def extract_entities(state):
    entities = await entity_extractor.aextract(state["document"])
    return {"entity_results": entities}

async def extract_relations(state):
    relations = await relation_extractor.aextract(state["document"])
    return {"relation_results": relations}

async def enrich_from_web(state):
    enrichments = await web_enricher.aenrich(state["document"])
    return {"enrichment_results": enrichments}

def merge_results(state):
    """Merge parallel results into the KG."""
    merged = merge_and_deduplicate(
        state["entity_results"],
        state["relation_results"],
        state["enrichment_results"],
    )
    kg_client.bulk_upsert(merged)
    return {"merged_kg": merged}

workflow = StateGraph(ParallelState)
workflow.add_node("extract_entities", extract_entities)
workflow.add_node("extract_relations", extract_relations)
workflow.add_node("enrich", enrich_from_web)
workflow.add_node("merge", merge_results)

# Fan-out: all three run in parallel
workflow.set_entry_point("extract_entities")
# LangGraph handles parallel execution when nodes have no dependencies
workflow.add_edge("extract_entities", "merge")
workflow.add_edge("extract_relations", "merge")
workflow.add_edge("enrich", "merge")
workflow.add_edge("merge", END)
```

---

## Pattern 3: Supervisor Pattern

### Description

A supervisor agent delegates tasks to specialist agents based on the query or
document type. The supervisor reads from the KG to decide which specialist to
invoke, and specialists write results back to the KG.

```
                   [Supervisor]
                   /    |     \
                  v     v      v
        [Science    [Business   [Technical
         Agent]      Agent]      Agent]
                  \     |      /
                   v    v     v
                   Shared KG
```

Reference: https://langchain-ai.github.io/langgraph/concepts/multi_agent/

### When to Use

- Documents span multiple domains (medical, legal, technical).
- Different entity types require different extraction strategies.
- You want to route expensive LLM calls only to documents that need them.

### Implementation

```python
def supervisor(state):
    """Decide which specialist agent to invoke."""
    document = state["document"]

    # Use the KG to understand the document's domain
    existing_context = kg_client.get_domain_context(document)

    # Ask LLM to classify and route
    classification = llm.classify(
        f"Document: {document[:500]}\nExisting context: {existing_context}\n"
        f"Which specialist should handle this: science, business, or technical?",
        options=["science", "business", "technical"]
    )
    return {"next_agent": classification}

def route(state) -> str:
    return state["next_agent"]

workflow = StateGraph(SupervisorState)
workflow.add_node("supervisor", supervisor)
workflow.add_node("science", science_agent)
workflow.add_node("business", business_agent)
workflow.add_node("technical", technical_agent)
workflow.add_node("write_kg", write_to_kg)

workflow.set_entry_point("supervisor")
workflow.add_conditional_edges("supervisor", route, {
    "science": "science",
    "business": "business",
    "technical": "technical",
})
workflow.add_edge("science", "write_kg")
workflow.add_edge("business", "write_kg")
workflow.add_edge("technical", "write_kg")
workflow.add_edge("write_kg", END)
```

---

## Pattern 4: Debate Pattern

### Description

Multiple agents propose competing facts or interpretations. The KG records all
proposals with provenance. A judge agent or confidence-weighted voting determines
which facts become canonical.

This pattern is useful when:
- Extraction is ambiguous (is "Apple" the company or the fruit?).
- Multiple sources disagree on a fact.
- High stakes require multiple verification.

### Architecture

```
Document --> [Agent A] --> Proposal A (conf: 0.8)
         --> [Agent B] --> Proposal B (conf: 0.6)
         --> [Agent C] --> Proposal C (conf: 0.9)
                              |
                              v
                        [Judge Agent]
                              |
                              v
                     KG (canonical fact)
```

### Implementation

```python
class DebateState(TypedDict):
    document: str
    proposals: list  # Each proposal: {agent, entities, confidence, reasoning}
    resolved_facts: list

def agent_a_propose(state):
    """Agent A extracts with GPT-4 (high quality, expensive)."""
    result = gpt4_extractor.extract(state["document"])
    proposal = {"agent": "gpt4", "entities": result, "confidence": 0.9}
    return {"proposals": state.get("proposals", []) + [proposal]}

def agent_b_propose(state):
    """Agent B extracts with Claude (different perspective)."""
    result = claude_extractor.extract(state["document"])
    proposal = {"agent": "claude", "entities": result, "confidence": 0.85}
    return {"proposals": state.get("proposals", []) + [proposal]}

def judge(state):
    """Judge resolves disagreements between proposals."""
    proposals = state["proposals"]

    # Find agreements and disagreements
    agreements, disagreements = compare_proposals(proposals)

    resolved = []
    # Agreements go straight to KG
    for fact in agreements:
        resolved.append({**fact, "method": "consensus"})

    # Disagreements require resolution
    for disagreement in disagreements:
        # Use confidence-weighted voting
        winner = max(disagreement["options"],
                     key=lambda o: o["confidence"])
        resolved.append({
            **winner,
            "method": "confidence_vote",
            "alternatives": disagreement["options"],
        })

    return {"resolved_facts": resolved}
```

### KG Provenance for Debates

```cypher
// Store the winning fact
CREATE (e:Entity {name: $name, type: $type, confidence: $confidence})

// Store provenance: which agents proposed what
CREATE (p:Proposal {
    agent: $agent_name,
    proposed_type: $proposed_type,
    confidence: $agent_confidence,
    reasoning: $reasoning,
    was_selected: $is_winner
})
CREATE (p)-[:PROPOSED]->(e)
```

---

## Pattern 5: CrewAI + KG Integration

### Description

CrewAI provides a framework for defining agents with roles, goals, and tools.
Integrating a KG as a shared tool gives all crew members access to structured
knowledge.

### Implementation Pattern

```python
from crewai import Agent, Task, Crew
from crewai.tools import tool

@tool
def query_knowledge_graph(query: str) -> str:
    """Query the knowledge graph for entities and relationships."""
    results = kg_client.hybrid_search(query, top_k=5)
    return format_results(results)

@tool
def add_to_knowledge_graph(entity: str, entity_type: str,
                           description: str) -> str:
    """Add a new entity to the knowledge graph."""
    kg_client.upsert_entity(entity, entity_type, description)
    return f"Added {entity} ({entity_type}) to KG"

researcher = Agent(
    role="Research Analyst",
    goal="Extract accurate entities and relationships from documents",
    tools=[query_knowledge_graph, add_to_knowledge_graph],
)

validator = Agent(
    role="Fact Checker",
    goal="Validate extracted facts against existing knowledge",
    tools=[query_knowledge_graph],
)

crew = Crew(
    agents=[researcher, validator],
    tasks=[extract_task, validate_task],
    process="sequential",
)
```

---

## Locking Strategies for Concurrent Neo4j Writes

### The Problem

Neo4j is ACID-compliant for individual transactions, but concurrent agent writes
can still cause logical conflicts (duplicate entities, inconsistent relationships).

### Strategy 1: Transaction-Level Locking

Neo4j automatically acquires locks on nodes and relationships modified within a
transaction. Use explicit write transactions to prevent conflicts:

```python
def safe_upsert(driver, entity_name, entity_type, confidence):
    """Thread-safe entity upsert using Neo4j write transactions."""
    with driver.session() as session:
        session.execute_write(
            lambda tx: tx.run("""
                MERGE (e:Entity {name: $name})
                ON CREATE SET e.type = $type, e.confidence = $confidence,
                              e.created_at = datetime()
                ON MATCH SET e.confidence = CASE
                    WHEN $confidence > e.confidence THEN $confidence
                    ELSE e.confidence
                END,
                e.updated_at = datetime()
            """, name=entity_name, type=entity_type, confidence=confidence)
        )
```

### Strategy 2: Queue-Based Writes

Instead of agents writing directly to Neo4j, they publish write operations to
a queue. A single writer agent processes the queue sequentially:

```python
import queue
import threading

write_queue = queue.Queue()

def writer_loop():
    """Single writer thread processes all KG mutations."""
    while True:
        operation = write_queue.get()
        if operation is None:
            break
        execute_kg_write(operation)
        write_queue.task_done()

# Agents enqueue writes instead of writing directly
def agent_write(entity, relation):
    write_queue.put({"type": "upsert", "entity": entity, "relation": relation})
```

### Strategy 3: Batch-and-Merge

Agents accumulate writes in local buffers. Periodically, a merge operation
flushes all buffers to Neo4j in a single transaction:

```python
class BufferedKGWriter:
    def __init__(self, kg_client, flush_interval: int = 100):
        self.buffer = []
        self.kg = kg_client
        self.flush_interval = flush_interval

    def write(self, entity):
        self.buffer.append(entity)
        if len(self.buffer) >= self.flush_interval:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        # Deduplicate within buffer before writing
        unique = deduplicate(self.buffer)
        self.kg.bulk_upsert(unique)
        self.buffer = []
```

---

## Choosing the Right Pattern

| Pattern | Complexity | Latency | Best For |
|---------|-----------|---------|----------|
| Sequential Pipeline | Low | High (serial) | Simple ETL, debugging |
| Parallel Agents | Medium | Low (parallel) | Independent tasks, throughput |
| Supervisor | Medium | Medium | Multi-domain documents |
| Debate | High | High (multiple LLM calls) | High-stakes, ambiguous facts |
| CrewAI + KG | Medium | Medium | Rapid prototyping, role-based |

### Decision Framework

1. **Start with sequential.** It is the easiest to debug and reason about.
2. **Move to parallel** when sequential is too slow and agents are independent.
3. **Add a supervisor** when documents span multiple domains.
4. **Use debate** only when accuracy is critical and you can afford the latency.
5. **Use CrewAI** for rapid prototyping; migrate to LangGraph for production.

---

## Key Takeaways

1. **The KG is the coordination layer.** In multi-agent systems, the KG serves as
   the shared state that all agents read from and write to.

2. **Conflict resolution is the hard problem.** Choose a strategy (optimistic locking,
   append-only, namespaces) early and stick with it.

3. **Provenance is non-negotiable.** Every fact in the KG should record which agent
   created it, with what confidence, from what source.

4. **Start simple, add complexity when needed.** Sequential pipelines cover 80% of
   use cases. Only add parallelism or debate when you have a clear need.

5. **LangGraph handles the orchestration.** Use its StateGraph for defining agent
   topologies and checkpointers for persistence.
