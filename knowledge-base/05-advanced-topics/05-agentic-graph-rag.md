# Agentic Graph RAG

Static RAG pipelines follow a fixed retrieve-then-generate pattern. Agentic Graph RAG replaces this rigid pipeline with an **autonomous agent** that dynamically decides what to retrieve, how to query the graph, and when to enrich the knowledge base -- adapting its strategy based on the question and intermediate results.

## Limitations of Static Pipelines

A typical static Graph RAG pipeline looks like this:

```
Query → Extract Entities → Fixed Cypher Template → Retrieve → Generate → Answer
```

This breaks down in predictable ways:

| Limitation | Example |
|-----------|---------|
| Fixed retrieval strategy | Always uses 2-hop traversal, even when 1 hop or 4 hops would be better |
| No adaptation | Cannot change approach when initial retrieval yields poor results |
| No self-correction | If entity extraction fails, the pipeline fails silently |
| Single retrieval pass | Cannot retrieve additional context based on what it learns |
| No enrichment | Cannot fetch missing information from external sources |
| One-size-fits-all | Uses the same approach for simple lookups and complex research questions |

An agentic system treats each of these as a **decision point** rather than a hardcoded step.

## Agent Architecture

An agentic Graph RAG system has three core components: a **reasoning loop**, a **tool set**, and a **state manager**.

```
                    ┌─────────────────────────┐
                    │     Reasoning LLM        │
                    │  (decides what to do)     │
                    └──────────┬──────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼
        ┌──────────┐   ┌──────────┐    ┌──────────────┐
        │  Graph    │   │  Vector  │    │    Web       │
        │  Query    │   │  Search  │    │  Enrichment  │
        │  Tool     │   │  Tool    │    │    Tool      │
        └──────────┘   └──────────┘    └──────────────┘
              │                │                 │
              └────────────────┼────────────────┘
                               ▼
                    ┌─────────────────────────┐
                    │    State Manager         │
                    │  (tracks evidence,       │
                    │   plans, confidence)     │
                    └─────────────────────────┘
```

### The Tool Set

```python
from langchain_core.tools import tool

@tool
def graph_query(cypher: str) -> list[dict]:
    """Execute a Cypher query against the knowledge graph.
    Use this to traverse relationships, find paths, or
    retrieve structured facts."""
    with driver.session() as session:
        result = session.run(cypher)
        return [dict(record) for record in result]

@tool
def vector_search(query: str, top_k: int = 10,
                  filter_labels: list[str] = None) -> list[dict]:
    """Search for semantically similar nodes or documents.
    Use this when you need fuzzy matching or the entity
    name is uncertain."""
    params = {"query": query, "top_k": top_k}
    if filter_labels:
        params["filter"] = {"label": {"$in": filter_labels}}
    return vector_store.similarity_search(**params)

@tool
def graph_schema() -> dict:
    """Retrieve the current graph schema (node labels,
    relationship types, property keys). Use this to
    understand what's in the graph before writing queries."""
    with driver.session() as session:
        result = session.run("CALL db.schema.visualization()")
        return format_schema(result)

@tool
def web_search(query: str) -> list[dict]:
    """Search the web for information not in the knowledge graph.
    Use this when graph and vector search yield insufficient results."""
    return search_api.search(query, num_results=5)

@tool
def add_to_graph(triples: list[dict]) -> str:
    """Add new facts to the knowledge graph.
    Each triple: {subject, relation, object, source, confidence}.
    Use this after web enrichment to persist new knowledge."""
    for triple in triples:
        insert_triple(triple)
    return f"Added {len(triples)} triples to the graph."
```

## LangGraph for Orchestration

LangGraph is the natural fit for agentic Graph RAG because it provides:
- **Stateful execution**: tracks accumulated evidence across tool calls
- **Conditional routing**: different paths based on intermediate results
- **Human-in-the-loop**: pause for approval before modifying the graph
- **Persistence**: resume interrupted workflows

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    query: str
    strategy: str                    # "local", "global", "hybrid"
    evidence: Annotated[list, operator.add]
    confidence: float
    iterations: int

tools = [graph_query, vector_search, graph_schema, web_search, add_to_graph]

def route_query(state: AgentState) -> dict:
    """Classify the query and select initial strategy."""
    classification = llm.invoke(f"""
    Classify this query into one of these categories:
    - SIMPLE_LOOKUP: answerable with a single graph query
    - MULTI_HOP: requires following a chain of relationships
    - EXPLORATORY: broad question needing community-level summarization
    - ENRICHMENT_NEEDED: likely requires information not in the graph

    Query: {state["query"]}

    Respond with just the category name.
    """).strip()

    strategy_map = {
        "SIMPLE_LOOKUP": "local",
        "MULTI_HOP": "local",
        "EXPLORATORY": "global",
        "ENRICHMENT_NEEDED": "hybrid",
    }
    return {"strategy": strategy_map.get(classification, "hybrid")}

def agent_node(state: AgentState) -> dict:
    """The reasoning LLM decides which tool to call next."""
    system_prompt = f"""You are a Graph RAG agent. Your current strategy is: {state["strategy"]}.

    Available tools: graph_query, vector_search, graph_schema, web_search, add_to_graph.

    Strategy guidelines:
    - local: start with graph_query for precise traversal
    - global: start with vector_search for broad coverage, then graph_query for structure
    - hybrid: use both, consider web_search if graph is insufficient

    Current evidence collected: {len(state.get("evidence", []))} items
    Current confidence: {state.get("confidence", 0.0)}

    If you have sufficient evidence (confidence > 0.8), provide the final answer.
    Otherwise, call the most appropriate tool to gather more evidence.
    """

    response = llm_with_tools.invoke(
        [{"role": "system", "content": system_prompt}] + state["messages"]
    )
    return {"messages": [response], "iterations": state.get("iterations", 0) + 1}

def should_continue(state: AgentState) -> str:
    """Decide whether to continue gathering evidence or answer."""
    last_message = state["messages"][-1]

    # If the LLM made tool calls, execute them
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Max iterations safety check
    if state.get("iterations", 0) >= 8:
        return "answer"

    return "answer"

def answer_node(state: AgentState) -> dict:
    """Generate the final answer from accumulated evidence."""
    final = llm.invoke(f"""
    Question: {state["query"]}

    Evidence gathered through graph traversal and search:
    {format_evidence(state.get("evidence", []))}

    Provide a comprehensive, well-sourced answer.
    """)
    return {"messages": [{"role": "assistant", "content": final}]}

# Build the agent graph
workflow = StateGraph(AgentState)
workflow.add_node("route", route_query)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("answer", answer_node)

workflow.set_entry_point("route")
workflow.add_edge("route", "agent")
workflow.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    "answer": "answer",
})
workflow.add_edge("tools", "agent")
workflow.add_edge("answer", END)

app = workflow.compile()
```

## Dynamic Strategy Selection

The agent selects between local, global, and hybrid strategies based on query characteristics:

### Local Retrieval

Best for: specific factual questions about known entities.

```
"What year was company X founded?" → graph_query with direct match
```

The agent writes a targeted Cypher query and retrieves the answer in one step.

### Global Retrieval

Best for: broad, thematic questions that span many entities.

```
"What are the main themes in this research corpus?" → vector_search for broad coverage
                                                     → graph_query for community detection
```

The agent uses community summaries or vector search across the entire graph to build a high-level picture.

### Hybrid Retrieval

Best for: complex questions that need both precision and breadth.

```
"How does drug X compare to alternatives for condition Y?"
→ graph_query: find Drug X → treats → Condition Y
→ graph_query: find other drugs treating Condition Y
→ vector_search: find clinical study summaries
→ web_search: check for recent trial results (if graph is stale)
```

### Strategy Adaptation

The agent can **switch strategies mid-execution** if initial results are poor:

```python
def evaluate_evidence(state: AgentState) -> dict:
    """Assess whether current evidence is sufficient and adapt strategy."""
    assessment = llm.invoke(f"""
    Question: {state["query"]}
    Strategy: {state["strategy"]}
    Evidence so far: {state["evidence"]}

    Rate the evidence: SUFFICIENT, PARTIAL, or INSUFFICIENT.
    If PARTIAL or INSUFFICIENT, suggest a strategy change.
    """)

    if "INSUFFICIENT" in assessment and state["strategy"] == "local":
        return {"strategy": "hybrid"}  # escalate to hybrid
    if "INSUFFICIENT" in assessment and state["strategy"] == "hybrid":
        return {"strategy": "hybrid"}  # trigger web enrichment

    confidence = 0.9 if "SUFFICIENT" in assessment else 0.5
    return {"confidence": confidence}
```

## Autonomous KG Construction

Agentic systems can decide **what to ingest** into the knowledge graph, not just how to query it:

```python
@tool
def evaluate_and_ingest(document_url: str, relevance_criteria: str) -> str:
    """Evaluate whether a document should be ingested into the KG.
    Downloads the document, assesses relevance, and if relevant,
    extracts entities/relations and adds them to the graph."""

    # Fetch the document
    content = fetch_document(document_url)

    # Assess relevance
    assessment = llm.invoke(f"""
    Should this document be added to our knowledge graph?
    Criteria: {relevance_criteria}

    Document preview: {content[:2000]}

    Respond: RELEVANT (with justification) or SKIP (with reason).
    """)

    if "SKIP" in assessment:
        return f"Skipped: {assessment}"

    # Extract and ingest
    triples = extract_triples(content)
    insert_triples(triples, source=document_url)
    return f"Ingested {len(triples)} triples from {document_url}"
```

### Agent-Driven KG Maintenance

Beyond construction, agents can maintain KG quality:

- **Gap detection**: "The graph has no information about X's funding history -- should I search for it?"
- **Staleness detection**: "This fact was last updated 18 months ago -- should I verify it?"
- **Conflict resolution**: "Two sources disagree about X -- let me find a third source to arbitrate."

## Multi-LLM Coordination

Production agentic Graph RAG systems use different models for different tasks to optimize cost and quality:

```python
# Model assignment by task
models = {
    "entity_extraction": "gpt-4o-mini",       # cheap, fast, good enough
    "query_classification": "gpt-4o-mini",     # simple classification
    "cypher_generation": "claude-sonnet",       # needs precision
    "reasoning": "claude-opus",                 # complex multi-hop reasoning
    "summarization": "gpt-4o-mini",            # bulk summarization
}

class MultiModelAgent:
    def __init__(self, model_config: dict):
        self.models = {
            task: get_llm(model_name)
            for task, model_name in model_config.items()
        }

    def extract_entities(self, text: str) -> list[dict]:
        """Use cheap model for high-volume extraction."""
        return self.models["entity_extraction"].invoke(
            entity_extraction_prompt(text)
        )

    def generate_cypher(self, question: str, schema: dict) -> str:
        """Use mid-tier model for query generation."""
        return self.models["cypher_generation"].invoke(
            cypher_generation_prompt(question, schema)
        )

    def reason(self, question: str, evidence: list[dict]) -> str:
        """Use powerful model for final reasoning."""
        return self.models["reasoning"].invoke(
            reasoning_prompt(question, evidence)
        )
```

### Cost Profile Example

| Task | Model | Calls/Query | Cost/Call | Total |
|------|-------|------------|-----------|-------|
| Classification | gpt-4o-mini | 1 | $0.001 | $0.001 |
| Entity extraction | gpt-4o-mini | 1 | $0.002 | $0.002 |
| Cypher generation | claude-sonnet | 1-3 | $0.01 | $0.01-0.03 |
| Final reasoning | claude-opus | 1 | $0.05 | $0.05 |
| **Total per query** | | | | **$0.06-0.08** |

Using a single powerful model for everything would cost $0.15-0.25 per query -- 2-3x more with no quality improvement on simple sub-tasks.

## Production Considerations

### Guardrails

Agentic systems need boundaries:

```python
AGENT_GUARDRAILS = {
    "max_tool_calls": 10,           # prevent infinite loops
    "max_graph_mutations": 5,       # limit writes per query
    "require_approval_for_writes": True,  # human-in-the-loop for KG changes
    "allowed_cypher_operations": ["MATCH", "RETURN"],  # read-only by default
    "max_web_searches": 3,          # control external API costs
    "timeout_seconds": 30,          # overall time budget
}
```

### Observability

Track agent behavior to debug and improve:

- **Tool call sequences**: which tools were called and in what order
- **Strategy decisions**: why the agent chose local vs global vs hybrid
- **Evidence quality**: did retrieved context actually contribute to the answer
- **Cost per query**: model usage across the multi-LLM pipeline
- **Dead ends**: how often the agent retrieves irrelevant context

### When to Use Agentic vs Static

| Factor | Static Pipeline | Agentic |
|--------|----------------|---------|
| Query variety | Low (known patterns) | High (unpredictable questions) |
| Latency requirement | < 500ms | 2-10 seconds acceptable |
| Cost sensitivity | Very high | Moderate |
| KG completeness | High (well-curated) | Low (gaps expected) |
| Required accuracy | Good enough | Best possible |

## Key Takeaways

- Static Graph RAG pipelines fail on complex, varied queries because every decision is hardcoded
- Agentic systems use an LLM reasoning loop to dynamically select tools and strategies
- Tools include graph query, vector search, web enrichment, and graph mutation
- LangGraph provides the state management, conditional routing, and persistence needed for agent orchestration
- Dynamic strategy selection adapts between local, global, and hybrid retrieval based on query type
- Autonomous KG construction lets agents decide what to ingest, detect gaps, and resolve conflicts
- Multi-LLM coordination assigns cheap models to simple tasks and powerful models to reasoning, cutting costs by 2-3x
- Production systems need guardrails (max iterations, read-only defaults, approval for writes) and observability
