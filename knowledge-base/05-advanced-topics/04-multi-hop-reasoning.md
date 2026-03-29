# Multi-Hop Reasoning Over Knowledge Graphs

Multi-hop reasoning is the process of answering a question by following **chains of relationships** through a knowledge graph. Instead of retrieving a single fact, the system traverses multiple edges to connect information that no single node or document contains.

## What Is Multi-Hop Reasoning?

A "hop" is one traversal step across a relationship. Multi-hop reasoning chains multiple hops together:

```
Question: "What university did the inventor of the telephone attend?"

Hop 1: (Telephone) ←[invented_by]— (Alexander Graham Bell)
Hop 2: (Alexander Graham Bell) —[attended]→ (University of Edinburgh)

Answer: University of Edinburgh
```

The key insight: **neither hop alone answers the question**. The answer emerges only from connecting them.

### Single-Hop vs Multi-Hop

| Aspect | Single-Hop | Multi-Hop |
|--------|-----------|-----------|
| Query complexity | "Who invented X?" | "Where was the inventor of X educated?" |
| Retrieval | One lookup | Chain of lookups |
| Context needed | One fact | Multiple connected facts |
| Vector search | Often sufficient | Usually insufficient |
| Graph traversal | Simple match | Path-based reasoning |

## Why Multi-Hop Is Hard for Standard RAG

Standard vector-based RAG retrieves chunks by semantic similarity. For multi-hop questions, this breaks down:

```
Question: "What diseases are treated by drugs developed at the university
           where the discoverer of penicillin studied?"

Required chain: Penicillin → Fleming → St Mary's Hospital Medical School
                → drugs developed there → diseases they treat

Vector search: retrieves chunks about penicillin, Fleming, various drugs
               -- but cannot reconstruct the chain
```

Problems with naive retrieval:
1. **Missing links**: relevant intermediate entities are not in the top-k results
2. **No path awareness**: retrieved chunks are independent, not connected
3. **Combinatorial explosion**: concatenating all possibly-relevant chunks exceeds context limits

## Chain-of-Thought Over Graphs

Chain-of-thought (CoT) reasoning applied to graphs means the system explicitly plans and executes a traversal strategy:

```python
def graph_chain_of_thought(question: str, graph) -> str:
    """Decompose a question into graph traversal steps."""

    # Step 1: LLM plans the reasoning chain
    plan = llm.invoke(f"""
    Given this question: "{question}"

    Decompose it into a sequence of graph lookups.
    For each step, specify:
    - What entity to start from
    - What relationship to follow
    - What you expect to find

    Format as numbered steps.
    """)

    # Step 2: Execute each step against the graph
    context = {}
    for step in parse_steps(plan):
        result = graph.query(
            start_entity=step["start"],
            relationship=step["relationship"],
            direction=step["direction"],
        )
        context[step["label"]] = result

    # Step 3: LLM synthesizes the final answer from collected context
    answer = llm.invoke(f"""
    Question: {question}

    Evidence collected by traversing the knowledge graph:
    {format_context(context)}

    Based on this evidence, provide the answer.
    """)
    return answer
```

## Path-Based Reasoning

Path-based reasoning explicitly finds and evaluates **paths** between entities in the graph:

```cypher
// Find all paths between Penicillin and any Disease (max 4 hops)
MATCH path = (start:Entity {name: "Penicillin"})-[*1..4]-(end:Entity:Disease)
RETURN path, length(path) AS hops,
       [r IN relationships(path) | type(r)] AS relationship_chain
ORDER BY hops ASC
LIMIT 10
```

### Path Scoring

Not all paths are equally informative. Score paths by:

```python
def score_path(path: list[dict]) -> float:
    """Score a graph path by relevance and reliability."""
    score = 1.0

    # Shorter paths are generally more reliable
    score *= 1.0 / len(path)

    # High-confidence edges contribute more
    for edge in path:
        score *= edge.get("confidence", 0.5)

    # Penalize deprecated or low-quality relationships
    for edge in path:
        if edge.get("valid_to") is not None:
            score *= 0.1  # heavily penalize expired facts

    return score
```

### Path Verbalization

Once a path is found, convert it to natural language for the LLM:

```python
def verbalize_path(path: list[dict]) -> str:
    """Convert a graph path to natural language."""
    segments = []
    for i in range(0, len(path) - 1, 2):
        subject = path[i]["name"]
        relation = path[i + 1]["type"].replace("_", " ")
        obj = path[i + 2]["name"]
        segments.append(f"{subject} {relation} {obj}")
    return ". ".join(segments) + "."

# Example output:
# "Penicillin discovered by Alexander Fleming.
#  Alexander Fleming studied at St Mary's Hospital Medical School."
```

## Sub-Question Decomposition

For complex multi-hop questions, decompose the question into simpler sub-questions that can each be answered independently:

```python
def decompose_question(question: str) -> list[str]:
    """Use an LLM to decompose a complex question into sub-questions."""
    response = llm.invoke(f"""
    Break this complex question into simpler sub-questions that can
    each be answered with a single graph lookup:

    Question: "{question}"

    Rules:
    - Each sub-question should be answerable with one graph traversal
    - Later sub-questions can reference answers to earlier ones
    - Use [ANSWER_N] to reference the answer to sub-question N

    Return as a numbered list.
    """)
    return parse_sub_questions(response)
```

Example decomposition:

```
Original: "What awards has the director of the highest-grossing film of 2024 won?"

Sub-questions:
1. What is the highest-grossing film of 2024?           → [ANSWER_1]
2. Who directed [ANSWER_1]?                              → [ANSWER_2]
3. What awards has [ANSWER_2] won?                       → [ANSWER_3]
```

### Executing Sub-Questions Sequentially

```python
async def answer_with_decomposition(question: str, graph) -> str:
    """Answer a multi-hop question via sub-question decomposition."""
    sub_questions = decompose_question(question)
    answers = {}

    for i, sub_q in enumerate(sub_questions):
        # Substitute previous answers into the sub-question
        resolved_q = sub_q
        for j, prev_answer in answers.items():
            resolved_q = resolved_q.replace(f"[ANSWER_{j}]", prev_answer)

        # Retrieve from graph and answer
        context = graph.query_relevant(resolved_q, top_k=5)
        answer = llm.invoke(f"""
        Question: {resolved_q}
        Context: {context}
        Answer concisely.
        """)
        answers[i + 1] = answer.strip()

    # Final synthesis
    return llm.invoke(f"""
    Original question: {question}

    Sub-questions and answers:
    {format_qa_pairs(sub_questions, answers)}

    Provide the final answer.
    """)
```

## Iterative Retrieval: Retrieve-Reason-Retrieve

Instead of planning all steps upfront, iterative retrieval lets the system **decide dynamically** what to retrieve next based on what it has learned so far:

```
Query → Retrieve initial context → Reason about gaps → Retrieve more → Reason → ... → Answer
```

```python
def iterative_retrieval(question: str, graph, max_iterations: int = 5) -> str:
    """Iteratively retrieve and reason until the question is answerable."""
    collected_context = []

    for iteration in range(max_iterations):
        response = llm.invoke(f"""
        Question: {question}

        Context collected so far:
        {format_context(collected_context)}

        Can you answer the question with the available context?
        If YES, provide the answer.
        If NO, specify what additional information you need
        (entity name and relationship to look up).

        Respond with either:
        ANSWER: <your answer>
        NEED: <entity> | <relationship> | <direction>
        """)

        if response.startswith("ANSWER:"):
            return response.replace("ANSWER:", "").strip()

        # Parse the retrieval request and execute it
        entity, relation, direction = parse_need(response)
        new_context = graph.traverse(entity, relation, direction)
        collected_context.extend(new_context)

    # Max iterations reached -- answer with what we have
    return llm.invoke(f"""
    Question: {question}
    Context: {format_context(collected_context)}
    Provide the best answer you can with the available information.
    """)
```

## LangGraph for Multi-Hop Reasoning

LangGraph provides the state management and control flow needed for multi-hop reasoning agents. Its graph-based workflow model maps naturally to the retrieve-reason-retrieve pattern.

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class ReasoningState(TypedDict):
    question: str
    sub_questions: list[str]
    current_step: int
    evidence: Annotated[list[str], operator.add]
    intermediate_answers: dict[int, str]
    final_answer: str

def plan_node(state: ReasoningState) -> dict:
    """Decompose the question into sub-questions."""
    sub_qs = decompose_question(state["question"])
    return {"sub_questions": sub_qs, "current_step": 0}

def retrieve_node(state: ReasoningState) -> dict:
    """Retrieve graph context for the current sub-question."""
    step = state["current_step"]
    sub_q = state["sub_questions"][step]

    # Resolve references to previous answers
    for idx, ans in state.get("intermediate_answers", {}).items():
        sub_q = sub_q.replace(f"[ANSWER_{idx}]", ans)

    context = graph.query_relevant(sub_q, top_k=5)
    return {"evidence": [f"Step {step + 1}: {context}"]}

def reason_node(state: ReasoningState) -> dict:
    """Answer the current sub-question using retrieved evidence."""
    step = state["current_step"]
    sub_q = state["sub_questions"][step]
    evidence = state["evidence"]

    answer = llm.invoke(f"Question: {sub_q}\nEvidence: {evidence[-1]}\nAnswer:")

    answers = state.get("intermediate_answers", {})
    answers[step + 1] = answer.strip()
    return {
        "intermediate_answers": answers,
        "current_step": step + 1,
    }

def should_continue(state: ReasoningState) -> str:
    """Check if there are more sub-questions to process."""
    if state["current_step"] >= len(state["sub_questions"]):
        return "synthesize"
    return "retrieve"

def synthesize_node(state: ReasoningState) -> dict:
    """Combine all intermediate answers into a final response."""
    final = llm.invoke(f"""
    Question: {state["question"]}
    Evidence chain: {state["intermediate_answers"]}
    Provide a comprehensive final answer.
    """)
    return {"final_answer": final}

# Build the graph
workflow = StateGraph(ReasoningState)
workflow.add_node("plan", plan_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("reason", reason_node)
workflow.add_node("synthesize", synthesize_node)

workflow.set_entry_point("plan")
workflow.add_edge("plan", "retrieve")
workflow.add_edge("retrieve", "reason")
workflow.add_conditional_edges("reason", should_continue, {
    "retrieve": "retrieve",
    "synthesize": "synthesize",
})
workflow.add_edge("synthesize", END)

agent = workflow.compile()
```

## Practical Considerations

### Controlling Traversal Depth

More hops means more context but also more noise and latency:

| Hops | Use Case | Risk |
|------|----------|------|
| 1 | Direct fact lookup | May miss connections |
| 2 | Most multi-hop questions | Good balance |
| 3 | Complex reasoning chains | Context gets large |
| 4+ | Research-grade exploration | High noise, slow |

### Handling Dead Ends

Not every traversal path leads to an answer. Build in graceful fallbacks:

- If a sub-question yields no graph results, fall back to vector search
- If iterative retrieval stalls, present partial evidence and let the LLM reason with uncertainty
- Set a maximum iteration count to prevent infinite loops

### Evaluation

Multi-hop reasoning is harder to evaluate than single-hop retrieval:

- **Path accuracy**: did the system follow the correct chain of relationships?
- **Intermediate answer quality**: are sub-question answers correct?
- **Final answer correctness**: is the synthesized answer right?
- **Efficiency**: how many retrieval steps were needed vs the minimum possible?

## Key Takeaways

- Multi-hop reasoning chains multiple graph traversals to answer questions no single lookup can solve
- Sub-question decomposition breaks complex queries into answerable steps with dependency tracking
- Iterative retrieval dynamically decides what to retrieve next based on accumulated evidence
- Path-based reasoning explicitly finds and scores paths between entities
- LangGraph provides the state management needed for multi-step reasoning agents
- Limit traversal depth to 2-3 hops for most production use cases; deeper traversal adds noise
