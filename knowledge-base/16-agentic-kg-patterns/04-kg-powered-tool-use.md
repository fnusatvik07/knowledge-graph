# KG-Powered Tool Use

## Overview

Agents do not just store knowledge in graphs — they use knowledge graphs as
reasoning tools. This section covers patterns for wrapping KG queries as LangChain
tools, routing queries dynamically, chaining tools together, and optimizing cost.

---

## KG Query as a LangChain Tool

### The Core Pattern

Wrap Cypher queries as LangChain tools so the LLM can dynamically decide when
to query the knowledge graph.

Reference: https://python.langchain.com/docs/how_to/tool_calling/

```python
from langchain_core.tools import tool
from langchain_neo4j import Neo4jGraph

graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="password")


@tool
def search_entity(name: str) -> str:
    """Search for an entity in the knowledge graph by name.
    Returns the entity's type, description, and immediate relationships."""
    result = graph.query("""
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($name)
        OPTIONAL MATCH (e)-[r]-(neighbor:Entity)
        RETURN e.name, e.type, e.description,
               collect({
                   relation: type(r),
                   neighbor: neighbor.name,
                   neighbor_type: neighbor.type
               }) AS relationships
        LIMIT 5
    """, {"name": name})
    if not result:
        return f"No entity found matching '{name}'"
    return format_entity_results(result)


@tool
def find_path(entity_a: str, entity_b: str) -> str:
    """Find the shortest relationship path between two entities in the KG."""
    result = graph.query("""
        MATCH (a:Entity), (b:Entity)
        WHERE toLower(a.name) CONTAINS toLower($a)
        AND toLower(b.name) CONTAINS toLower($b)
        MATCH path = shortestPath((a)-[*..5]-(b))
        RETURN [n IN nodes(path) | n.name] AS node_names,
               [r IN relationships(path) | type(r)] AS rel_types
        LIMIT 1
    """, {"a": entity_a, "b": entity_b})
    if not result:
        return f"No path found between '{entity_a}' and '{entity_b}'"
    return format_path_results(result)


@tool
def count_entities(entity_type: str) -> str:
    """Count entities of a specific type in the knowledge graph."""
    result = graph.query("""
        MATCH (e:Entity {type: $type})
        RETURN count(e) AS count
    """, {"type": entity_type.upper()})
    count = result[0]["count"] if result else 0
    return f"There are {count} {entity_type} entities in the knowledge graph."


@tool
def get_community_summary(topic: str) -> str:
    """Get a summary of a topic community in the knowledge graph.
    Use this for broad questions about a domain rather than specific entities."""
    result = graph.query("""
        MATCH (c:Community)
        WHERE toLower(c.summary) CONTAINS toLower($topic)
        RETURN c.summary, c.member_count
        ORDER BY c.member_count DESC
        LIMIT 3
    """, {"topic": topic})
    if not result:
        return f"No community summaries found for '{topic}'"
    return "\n\n".join([r["c.summary"] for r in result])
```

### Binding Tools to a Chat Model

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

# Bind all KG tools to the model
tools = [search_entity, find_path, count_entities, get_community_summary]
llm_with_tools = llm.bind_tools(tools)

# The LLM now decides when to call which tool
response = llm_with_tools.invoke("How is Geoffrey Hinton connected to DeepMind?")
# LLM will call find_path("Geoffrey Hinton", "DeepMind") automatically
```

---

## Dynamic Tool Routing

### The Problem

Not every query needs the same tool. A factual question ("Where does Alice work?")
should use a precise KG lookup. A broad question ("What are the main trends in
AI?") should use community summaries. A comparison question ("How do GCN and GAT
differ?") might need both.

### Router Pattern

```python
from langchain_core.prompts import ChatPromptTemplate

ROUTER_PROMPT = ChatPromptTemplate.from_template("""
Given the user's question, decide which tool(s) to use:

Available tools:
1. search_entity: For specific entity lookups (who, what, where)
2. find_path: For relationship questions (how is X connected to Y)
3. count_entities: For counting/aggregation questions
4. get_community_summary: For broad topic questions
5. web_search: For current events or information not in the KG
6. none: The question can be answered from general knowledge

Question: {question}

Respond with the tool name(s) to use, comma-separated.
""")


class ToolRouter:
    """Route queries to the most appropriate tool(s)."""

    def __init__(self, llm, tools: list):
        self.llm = llm
        self.tools = {t.name: t for t in tools}

    def route(self, question: str) -> list:
        """Determine which tools to use for this question."""
        # Heuristic pre-routing (saves LLM call for obvious cases)
        lower_q = question.lower()

        if "how many" in lower_q or "count" in lower_q:
            return [self.tools["count_entities"]]

        if "connected" in lower_q or "related" in lower_q or "path" in lower_q:
            return [self.tools["find_path"]]

        if any(w in lower_q for w in ["trend", "overview", "summary", "landscape"]):
            return [self.tools["get_community_summary"]]

        # Fall back to LLM routing for ambiguous queries
        response = self.llm.invoke(
            ROUTER_PROMPT.format(question=question)
        )
        tool_names = [t.strip() for t in response.content.split(",")]
        return [self.tools[n] for n in tool_names if n in self.tools]
```

### Cost-Aware Routing

```python
TOOL_COSTS = {
    "search_entity": 0.0001,       # Just a DB query
    "find_path": 0.0002,           # Slightly more expensive graph traversal
    "count_entities": 0.0001,      # Simple aggregation
    "get_community_summary": 0.0001,
    "web_search": 0.01,            # External API call
    "llm_reason": 0.05,            # Full LLM call for reasoning
}

class CostAwareRouter(ToolRouter):
    def route(self, question: str, budget: float = 0.10) -> list:
        """Route within a cost budget."""
        candidates = super().route(question)
        total_cost = sum(TOOL_COSTS.get(t.name, 0) for t in candidates)

        if total_cost > budget:
            # Prioritize cheap tools
            candidates.sort(key=lambda t: TOOL_COSTS.get(t.name, 0))
            selected = []
            running_cost = 0
            for tool in candidates:
                cost = TOOL_COSTS.get(tool.name, 0)
                if running_cost + cost <= budget:
                    selected.append(tool)
                    running_cost += cost
            return selected

        return candidates
```

---

## Tool Chaining

### KG Query -> Web Search -> KG Update -> Answer

Many real-world queries require chaining multiple tools. The KG provides context
for the web search, the web search provides new facts, and those facts are added
back to the KG.

```python
from langgraph.graph import StateGraph, END

class ChainState(TypedDict):
    question: str
    kg_context: str
    web_results: str
    new_facts: list
    final_answer: str

def query_kg(state):
    """Step 1: Query KG for existing context."""
    results = kg_tools.search_entity(state["question"])
    return {"kg_context": results}

def needs_web_search(state) -> str:
    """Decide if web search is needed (KG context insufficient)."""
    if "No entity found" in state["kg_context"] or len(state["kg_context"]) < 50:
        return "web_search"
    return "answer"

def web_search(state):
    """Step 2: Search the web for additional context."""
    results = web_search_tool.search(
        state["question"],
        context=state["kg_context"]  # KG context improves search quality
    )
    return {"web_results": results}

def update_kg(state):
    """Step 3: Extract facts from web results and add to KG."""
    new_facts = extractor.extract(state["web_results"])
    for fact in new_facts:
        kg_client.upsert(fact)
    return {"new_facts": new_facts}

def generate_answer(state):
    """Step 4: Generate final answer from all context."""
    context = f"KG Context:\n{state['kg_context']}\n"
    if state.get("web_results"):
        context += f"\nWeb Results:\n{state['web_results']}\n"
    answer = llm.invoke(f"{context}\n\nQuestion: {state['question']}")
    return {"final_answer": answer.content}

workflow = StateGraph(ChainState)
workflow.add_node("query_kg", query_kg)
workflow.add_node("web_search", web_search)
workflow.add_node("update_kg", update_kg)
workflow.add_node("answer", generate_answer)

workflow.set_entry_point("query_kg")
workflow.add_conditional_edges("query_kg", needs_web_search, {
    "web_search": "web_search",
    "answer": "answer",
})
workflow.add_edge("web_search", "update_kg")
workflow.add_edge("update_kg", "answer")
workflow.add_edge("answer", END)
```

---

## ReAct Pattern with KG Tools

### Observe-Think-Act Loops

The ReAct (Reasoning + Acting) pattern lets the agent iteratively reason about
what tool to use next based on previous observations.

```python
from langgraph.prebuilt import create_react_agent

# Define KG tools
tools = [search_entity, find_path, count_entities, get_community_summary]

# System prompt that teaches the agent about KG tools
SYSTEM_PROMPT = """You are a knowledge graph analyst. You have access to tools
that query a knowledge graph about AI/ML researchers, organizations, and technologies.

Strategy:
1. For specific entity questions, use search_entity first.
2. For relationship questions, use find_path.
3. For broad questions, start with get_community_summary then drill down.
4. If one tool gives insufficient results, try another approach.
5. Always cite the specific entities and relationships you found.
"""

# Create ReAct agent
agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=tools,
    prompt=SYSTEM_PROMPT,
)

# Run the agent
result = agent.invoke({
    "messages": [{"role": "user", "content": "Who invented GCN and what lab are they at?"}]
})
```

### Trace of a ReAct Execution

```
User: "Who invented GCN and what lab are they at?"

Thought: I need to find who invented GCN. Let me search for the GCN entity.
Action: search_entity("GCN")
Observation: GCN (TECHNOLOGY) - Graph Convolutional Network...
    Relationships: INVENTED by Thomas Kipf, INSTANCE_OF Graph Neural Networks

Thought: Thomas Kipf invented GCN. Now I need to find his affiliation.
Action: search_entity("Thomas Kipf")
Observation: Thomas Kipf (PERSON) - Research scientist at Google DeepMind...
    Relationships: AFFILIATED_WITH DeepMind, COLLABORATED_WITH Max Welling

Thought: I have enough information to answer.
Answer: Thomas Kipf invented GCN (Graph Convolutional Network). He is a
research scientist at Google DeepMind. He collaborated with Max Welling
on the foundational GCN paper.
```

---

## Function Calling with Structured Graph Outputs

### Structured Output Pattern

Instead of returning free-text from KG queries, return structured data that the
LLM can reason over precisely.

```python
from pydantic import BaseModel, Field
from typing import Optional

class EntityResult(BaseModel):
    """Structured result from a KG entity query."""
    name: str = Field(description="Entity name")
    type: str = Field(description="Entity type (PERSON, ORG, TECH, CONCEPT)")
    description: str = Field(description="Entity description")
    confidence: float = Field(description="Confidence score 0-1")
    relationships: list[dict] = Field(description="List of relationships")

class PathResult(BaseModel):
    """Structured result from a path query."""
    source: str
    target: str
    path_length: int
    nodes: list[str]
    relationships: list[str]
    exists: bool

@tool
def search_entity_structured(name: str) -> EntityResult:
    """Search for an entity and return structured results."""
    result = graph.query("""...""", {"name": name})
    if not result:
        return EntityResult(name=name, type="UNKNOWN", description="Not found",
                           confidence=0.0, relationships=[])
    r = result[0]
    return EntityResult(
        name=r["e.name"],
        type=r["e.type"],
        description=r["e.description"],
        confidence=r.get("e.confidence", 1.0),
        relationships=r["relationships"],
    )
```

### LLM with Structured Output

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

# Force the LLM to respond with a structured format
structured_llm = llm.with_structured_output(EntityResult)

# Or use function calling
response = llm.invoke(
    "Find information about Geoffrey Hinton",
    tools=[search_entity_structured],
    tool_choice={"type": "function", "function": {"name": "search_entity_structured"}},
)
```

---

## Cost Optimization

### The Two-Model Pattern

Use a cheap model to route and a powerful model to reason:

```python
class TwoModelAgent:
    """Cheap model routes, expensive model reasons."""

    def __init__(self):
        self.router = ChatOpenAI(model="gpt-4o-mini")   # $0.15/1M input tokens
        self.reasoner = ChatOpenAI(model="gpt-4o")       # $2.50/1M input tokens

    def answer(self, question: str) -> str:
        # Step 1: Cheap model decides which tools to call (fast, cheap)
        tools_to_use = self.route(question)

        # Step 2: Execute tools (DB queries, no LLM cost)
        tool_results = [tool.invoke(question) for tool in tools_to_use]

        # Step 3: Expensive model reasons over tool results (quality matters here)
        context = "\n".join(tool_results)
        answer = self.reasoner.invoke(
            f"Based on this knowledge graph data:\n{context}\n\n"
            f"Answer: {question}"
        )
        return answer.content

    def route(self, question: str) -> list:
        response = self.router.invoke(
            f"Which tools should I use for: {question}\n"
            f"Options: search_entity, find_path, count_entities, community_summary"
        )
        # Parse response into tool list
        return parse_tools(response.content)
```

### Cost Breakdown Per Query

| Component | Cost (typical) | Can Optimize? |
|-----------|---------------|---------------|
| Routing (cheap model) | $0.0001 | Already cheap |
| KG query (Neo4j) | $0.0000 | Free (self-hosted) |
| Reasoning (expensive model) | $0.005-0.02 | Cache common patterns |
| Web search (if needed) | $0.01 | Only when KG insufficient |
| **Total without web** | **$0.005-0.02** | |
| **Total with web** | **$0.015-0.03** | |

### Caching Pattern

```python
from functools import lru_cache
import hashlib

class CachedKGTools:
    """Cache KG query results to avoid redundant DB calls."""

    def __init__(self, kg_client, ttl_seconds: int = 300):
        self.kg = kg_client
        self.cache = {}
        self.ttl = ttl_seconds

    def search(self, query: str) -> str:
        key = hashlib.md5(query.encode()).hexdigest()
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["time"] < self.ttl:
                return entry["result"]

        result = self.kg.query(query)
        self.cache[key] = {"result": result, "time": time.time()}
        return result
```

---

## Key Takeaways

1. **Wrap Cypher as LangChain tools.** This lets the LLM dynamically decide when
   to query the KG rather than always querying or never querying.

2. **Route before reasoning.** Use heuristics or a cheap model to select tools.
   Only invoke the expensive model for final reasoning over tool results.

3. **Chain tools for complex queries.** KG context improves web search quality;
   web results enrich the KG. Build feedback loops.

4. **ReAct enables iterative reasoning.** The agent can query the KG, observe
   results, decide it needs more information, and query again.

5. **Structure your outputs.** Pydantic models for KG query results give the LLM
   precise data to reason over, reducing hallucination.

6. **Cache aggressively.** KG queries are deterministic within a time window.
   Cache results to avoid redundant database calls.
