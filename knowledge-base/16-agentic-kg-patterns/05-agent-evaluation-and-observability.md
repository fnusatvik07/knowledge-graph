# Agent Evaluation and Observability

## Overview

Building an agent that uses a knowledge graph is only half the challenge. The other
half is knowing whether it actually works: Does the agent use the KG effectively?
Does the KG improve answer quality? Where does the system fail, and why?

This section covers evaluation metrics, trajectory logging, cost tracking, A/B testing,
and dashboard patterns for agent KG systems.

---

## Agent Trajectory Logging

### What to Log

Every agent execution produces a trajectory: the sequence of decisions, tool calls,
and results that led to the final answer. Log all of it.

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass
class ToolCall:
    """Record of a single tool call within an agent trajectory."""
    tool_name: str
    input_args: dict
    output: str
    latency_ms: float
    token_count: int
    cost_usd: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ReasoningStep:
    """Record of an agent's reasoning before a tool call."""
    thought: str
    selected_tool: str
    reasoning: str  # Why this tool was chosen
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class AgentTrajectory:
    """Complete record of an agent execution."""
    session_id: str
    query: str
    steps: list  # List of ReasoningStep and ToolCall interleaved
    final_answer: str
    total_latency_ms: float
    total_cost_usd: float
    total_tokens: int
    kg_queries_count: int
    kg_entities_found: int
    success: Optional[bool] = None  # Set during evaluation
    created_at: datetime = field(default_factory=datetime.now)
```

### LangGraph Callback-Based Logging

```python
from langchain_core.callbacks import BaseCallbackHandler
import time

class TrajectoryLogger(BaseCallbackHandler):
    """Log agent trajectory during LangGraph execution."""

    def __init__(self):
        self.trajectory = []
        self.current_step_start = None

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.current_step_start = time.time()
        self.trajectory.append({
            "type": "tool_start",
            "tool": serialized.get("name", "unknown"),
            "input": input_str,
            "timestamp": datetime.now().isoformat(),
        })

    def on_tool_end(self, output, **kwargs):
        latency = (time.time() - self.current_step_start) * 1000
        self.trajectory.append({
            "type": "tool_end",
            "output": str(output)[:500],  # Truncate large outputs
            "latency_ms": latency,
            "timestamp": datetime.now().isoformat(),
        })

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.current_step_start = time.time()

    def on_llm_end(self, response, **kwargs):
        latency = (time.time() - self.current_step_start) * 1000
        usage = response.llm_output.get("token_usage", {}) if response.llm_output else {}
        self.trajectory.append({
            "type": "llm_call",
            "latency_ms": latency,
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "timestamp": datetime.now().isoformat(),
        })

    def get_summary(self) -> dict:
        tool_calls = [s for s in self.trajectory if s["type"] == "tool_end"]
        llm_calls = [s for s in self.trajectory if s["type"] == "llm_call"]
        return {
            "total_steps": len(self.trajectory),
            "tool_calls": len(tool_calls),
            "llm_calls": len(llm_calls),
            "total_latency_ms": sum(s.get("latency_ms", 0) for s in self.trajectory),
            "total_tokens": sum(
                s.get("input_tokens", 0) + s.get("output_tokens", 0)
                for s in llm_calls
            ),
        }
```

---

## Reasoning Chain Visualization

### From Question to Answer

Visualizing the agent's reasoning path helps debug failures and understand
which KG queries contributed to the answer.

```python
def visualize_trajectory(trajectory: AgentTrajectory) -> str:
    """Generate a text-based visualization of the agent's reasoning chain."""
    lines = []
    lines.append(f"Query: {trajectory.query}")
    lines.append(f"{'='*60}")

    for i, step in enumerate(trajectory.steps):
        if isinstance(step, ReasoningStep):
            lines.append(f"\n[Step {i+1}] THINK")
            lines.append(f"  Thought: {step.thought}")
            lines.append(f"  Selected tool: {step.selected_tool}")
            lines.append(f"  Reasoning: {step.reasoning}")

        elif isinstance(step, ToolCall):
            lines.append(f"\n[Step {i+1}] ACT: {step.tool_name}")
            lines.append(f"  Input: {step.input_args}")
            lines.append(f"  Output: {step.output[:200]}...")
            lines.append(f"  Latency: {step.latency_ms:.0f}ms | "
                        f"Cost: ${step.cost_usd:.4f}")

    lines.append(f"\n{'='*60}")
    lines.append(f"Answer: {trajectory.final_answer}")
    lines.append(f"\nTotal: {trajectory.total_latency_ms:.0f}ms | "
                f"${trajectory.total_cost_usd:.4f} | "
                f"{trajectory.total_tokens} tokens | "
                f"{trajectory.kg_queries_count} KG queries")

    return "\n".join(lines)
```

### Graph-Based Trajectory Visualization

Store trajectories as graphs for pattern analysis:

```cypher
// Create trajectory graph
CREATE (q:Query {text: $query, session: $session_id})
CREATE (a:Answer {text: $answer, correct: $is_correct})

// Each tool call is a node
CREATE (t1:ToolCall {
    tool: "search_entity",
    input: "Geoffrey Hinton",
    output_summary: "Found PERSON entity with 15 relationships",
    latency_ms: 45,
    timestamp: datetime()
})
CREATE (t2:ToolCall {
    tool: "find_path",
    input: "Hinton -> DeepMind",
    output_summary: "Path: Hinton -> Sutskever -> OpenAI (no direct path)",
    latency_ms: 120
})

// Chain the trajectory
CREATE (q)-[:LED_TO]->(t1)-[:FOLLOWED_BY]->(t2)-[:PRODUCED]->(a)
```

---

## Cost Tracking

### Per-Query Cost Breakdown

```python
class CostTracker:
    """Track costs across all components of an agent query."""

    # Cost per 1M tokens (approximate, varies by provider)
    MODEL_COSTS = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    }

    def __init__(self):
        self.entries = []

    def log(self, component: str, model: str, input_tokens: int,
            output_tokens: int, latency_ms: float):
        costs = self.MODEL_COSTS.get(model, {"input": 1.0, "output": 3.0})
        cost = (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1_000_000
        self.entries.append({
            "component": component,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
            "latency_ms": latency_ms,
        })

    def summary(self) -> dict:
        total_cost = sum(e["cost_usd"] for e in self.entries)
        total_tokens = sum(e["input_tokens"] + e["output_tokens"] for e in self.entries)
        total_latency = sum(e["latency_ms"] for e in self.entries)

        by_component = {}
        for e in self.entries:
            comp = e["component"]
            if comp not in by_component:
                by_component[comp] = {"cost": 0, "tokens": 0, "calls": 0}
            by_component[comp]["cost"] += e["cost_usd"]
            by_component[comp]["tokens"] += e["input_tokens"] + e["output_tokens"]
            by_component[comp]["calls"] += 1

        return {
            "total_cost_usd": total_cost,
            "total_tokens": total_tokens,
            "total_latency_ms": total_latency,
            "by_component": by_component,
        }

    def print_report(self):
        s = self.summary()
        print(f"\nCost Report")
        print(f"{'='*50}")
        print(f"Total: ${s['total_cost_usd']:.4f} | "
              f"{s['total_tokens']:,} tokens | "
              f"{s['total_latency_ms']:.0f}ms")
        print(f"\nBreakdown:")
        for comp, data in s["by_component"].items():
            print(f"  {comp:20s}: ${data['cost']:.4f} "
                  f"({data['calls']} calls, {data['tokens']:,} tokens)")
```

---

## Quality Metrics Beyond RAG

### Standard RAG Metrics Are Insufficient

RAG evaluation typically measures retrieval precision and answer faithfulness.
Agent KG systems need additional metrics.

### Agent-Specific Metrics

**1. Tool Selection Accuracy**

Did the agent choose the right tool for the query type?

```python
def evaluate_tool_selection(trajectories: list, ground_truth: list) -> float:
    """Measure how often the agent selected the correct tool."""
    correct = 0
    total = 0
    for traj, gt in zip(trajectories, ground_truth):
        first_tool = next(
            (s for s in traj.steps if isinstance(s, ToolCall)), None
        )
        if first_tool and first_tool.tool_name == gt["expected_first_tool"]:
            correct += 1
        total += 1
    return correct / max(total, 1)
```

**2. Reasoning Depth**

How many steps did the agent take? Too few suggests shallow reasoning; too many
suggests inefficiency.

```python
def reasoning_depth_stats(trajectories: list) -> dict:
    depths = [len(t.steps) for t in trajectories]
    return {
        "mean_depth": np.mean(depths),
        "median_depth": np.median(depths),
        "min_depth": min(depths),
        "max_depth": max(depths),
        "optimal_range": (2, 5),  # Domain-specific
    }
```

**3. KG Contribution Score**

Did the KG actually help the answer? Compare the answer quality with and without
KG tool results.

```python
def kg_contribution_score(query: str, answer_with_kg: str,
                          answer_without_kg: str, ground_truth: str,
                          evaluator_llm) -> dict:
    """Measure how much the KG improved the answer."""
    score_with = evaluator_llm.score_answer(query, answer_with_kg, ground_truth)
    score_without = evaluator_llm.score_answer(query, answer_without_kg, ground_truth)

    return {
        "score_with_kg": score_with,
        "score_without_kg": score_without,
        "kg_contribution": score_with - score_without,
        "kg_helped": score_with > score_without,
    }
```

**4. KG Coverage Rate**

What fraction of the answer's facts came from the KG vs. the LLM's parametric
knowledge?

```python
def kg_coverage_rate(answer: str, kg_facts_used: list, evaluator_llm) -> float:
    """What fraction of facts in the answer are grounded in KG data?"""
    facts_in_answer = evaluator_llm.extract_facts(answer)
    grounded = sum(1 for fact in facts_in_answer
                   if any(is_grounded(fact, kg_fact) for kg_fact in kg_facts_used))
    return grounded / max(len(facts_in_answer), 1)
```

---

## A/B Testing: Does the KG Actually Help?

### Experiment Design

The most important question: does adding a KG to your agent actually improve
performance compared to a vanilla RAG or pure LLM approach?

```python
class ABTest:
    """Compare agent with and without KG access."""

    def __init__(self, agent_with_kg, agent_without_kg, evaluator):
        self.with_kg = agent_with_kg
        self.without_kg = agent_without_kg
        self.evaluator = evaluator
        self.results = []

    def run(self, test_queries: list) -> dict:
        for query_data in test_queries:
            query = query_data["question"]
            ground_truth = query_data["expected_answer"]

            # Run both variants
            answer_a = self.with_kg.answer(query)
            answer_b = self.without_kg.answer(query)

            # Evaluate both
            score_a = self.evaluator.score(query, answer_a, ground_truth)
            score_b = self.evaluator.score(query, answer_b, ground_truth)

            self.results.append({
                "query": query,
                "with_kg_score": score_a,
                "without_kg_score": score_b,
                "kg_helped": score_a > score_b,
                "with_kg_latency": answer_a.latency_ms,
                "without_kg_latency": answer_b.latency_ms,
                "with_kg_cost": answer_a.cost_usd,
                "without_kg_cost": answer_b.cost_usd,
            })

        return self.summarize()

    def summarize(self) -> dict:
        n = len(self.results)
        kg_wins = sum(1 for r in self.results if r["kg_helped"])
        avg_improvement = np.mean([
            r["with_kg_score"] - r["without_kg_score"] for r in self.results
        ])
        avg_latency_overhead = np.mean([
            r["with_kg_latency"] - r["without_kg_latency"] for r in self.results
        ])
        avg_cost_overhead = np.mean([
            r["with_kg_cost"] - r["without_kg_cost"] for r in self.results
        ])

        return {
            "total_queries": n,
            "kg_wins": kg_wins,
            "kg_win_rate": kg_wins / n,
            "avg_score_improvement": avg_improvement,
            "avg_latency_overhead_ms": avg_latency_overhead,
            "avg_cost_overhead_usd": avg_cost_overhead,
            "recommendation": (
                "KEEP KG" if kg_wins / n > 0.6
                else "KG NOT WORTH IT" if kg_wins / n < 0.3
                else "MIXED — consider query-type routing"
            ),
        }
```

### Query Types Where KGs Excel

Based on published benchmarks and practical experience:

| Query Type | KG Advantage | Why |
|-----------|-------------|-----|
| Multi-hop relationships | High | LLMs hallucinate paths; KGs traverse them |
| Aggregation (count, list) | High | LLMs cannot count reliably |
| Temporal questions | High | KGs track when facts were true |
| Contradiction detection | High | KGs flag conflicting facts |
| Simple factual lookup | Low | LLMs know most common facts |
| Creative/opinion questions | None | KGs do not store opinions |
| Current events | Low | KGs lag behind real-time sources |

---

## LangSmith Integration for Tracing

Reference: https://docs.smith.langchain.com/

### Setup

```python
import os

# Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls__..."
os.environ["LANGCHAIN_PROJECT"] = "kg-agent-production"
```

### Custom Run Metadata

```python
from langsmith import RunTree

# Add KG-specific metadata to traces
with RunTree(
    name="kg_agent_query",
    run_type="chain",
    extra={
        "kg_entities_searched": 3,
        "kg_paths_found": 1,
        "kg_query_latency_ms": 45,
        "model_used": "gpt-4o",
    }
) as rt:
    result = agent.invoke(query)
    rt.end(outputs={"answer": result})
```

### Evaluation Datasets in LangSmith

```python
from langsmith import Client

client = Client()

# Create evaluation dataset
dataset = client.create_dataset("kg-agent-eval-v1")
for example in test_examples:
    client.create_example(
        inputs={"question": example["question"]},
        outputs={"answer": example["expected_answer"]},
        dataset_id=dataset.id,
    )

# Run evaluation
from langsmith.evaluation import evaluate

results = evaluate(
    agent.invoke,
    data="kg-agent-eval-v1",
    evaluators=[correctness_evaluator, faithfulness_evaluator],
)
```

---

## Dashboard Patterns

### Real-Time Monitoring Metrics

A production agent KG system should display:

**Health Metrics:**
- Agent uptime and error rate
- Neo4j connection pool utilization
- Average query latency (P50, P95, P99)
- KG size (nodes, edges, growth rate)

**Quality Metrics:**
- Answer correctness rate (from user feedback or automated evaluation)
- KG contribution rate (fraction of queries where KG improved the answer)
- Tool selection accuracy
- Average reasoning depth

**Cost Metrics:**
- Cost per query (by model, by tool)
- Daily/weekly token consumption
- Cost trend over time
- Cost per successful answer vs. failed answer

### Dashboard Implementation Pattern

```python
class AgentDashboard:
    """Collect and expose metrics for monitoring."""

    def __init__(self):
        self.metrics = {
            "queries_total": 0,
            "queries_successful": 0,
            "kg_queries_total": 0,
            "total_cost_usd": 0.0,
            "total_tokens": 0,
            "latencies": [],      # Ring buffer of recent latencies
            "error_count": 0,
        }

    def record_query(self, trajectory: AgentTrajectory):
        self.metrics["queries_total"] += 1
        if trajectory.success:
            self.metrics["queries_successful"] += 1
        self.metrics["kg_queries_total"] += trajectory.kg_queries_count
        self.metrics["total_cost_usd"] += trajectory.total_cost_usd
        self.metrics["total_tokens"] += trajectory.total_tokens
        self.metrics["latencies"].append(trajectory.total_latency_ms)

        # Keep only last 1000 latencies for percentile calculation
        if len(self.metrics["latencies"]) > 1000:
            self.metrics["latencies"] = self.metrics["latencies"][-1000:]

    def get_dashboard_data(self) -> dict:
        latencies = sorted(self.metrics["latencies"])
        n = len(latencies)
        return {
            "queries_total": self.metrics["queries_total"],
            "success_rate": (self.metrics["queries_successful"] /
                           max(self.metrics["queries_total"], 1)),
            "avg_kg_queries_per_request": (self.metrics["kg_queries_total"] /
                                          max(self.metrics["queries_total"], 1)),
            "total_cost_usd": self.metrics["total_cost_usd"],
            "cost_per_query": (self.metrics["total_cost_usd"] /
                              max(self.metrics["queries_total"], 1)),
            "latency_p50": latencies[n // 2] if n else 0,
            "latency_p95": latencies[int(n * 0.95)] if n else 0,
            "latency_p99": latencies[int(n * 0.99)] if n else 0,
        }
```

### Alerting Rules

```python
ALERT_RULES = {
    "high_error_rate": {
        "condition": lambda d: (1 - d["success_rate"]) > 0.1,
        "message": "Error rate exceeded 10%",
        "severity": "critical",
    },
    "high_latency": {
        "condition": lambda d: d["latency_p95"] > 5000,
        "message": "P95 latency exceeded 5 seconds",
        "severity": "warning",
    },
    "cost_spike": {
        "condition": lambda d: d["cost_per_query"] > 0.10,
        "message": "Average cost per query exceeded $0.10",
        "severity": "warning",
    },
    "low_kg_usage": {
        "condition": lambda d: d["avg_kg_queries_per_request"] < 0.5,
        "message": "Agent is not using KG for most queries — check tool routing",
        "severity": "info",
    },
}
```

---

## Key Takeaways

1. **Log everything.** Every tool call, LLM invocation, and decision should be
   recorded. You cannot debug what you cannot observe.

2. **Measure KG contribution explicitly.** Run A/B tests comparing agent with
   and without KG access. If the KG does not help, simplify your architecture.

3. **Track costs per component.** Know exactly how much each tool call and LLM
   invocation costs. Optimize the expensive parts first.

4. **Tool selection accuracy matters as much as answer quality.** An agent that
   calls the wrong tool wastes tokens and latency even if it eventually gets
   the right answer.

5. **Use LangSmith for production tracing.** It provides the infrastructure for
   trajectory logging, evaluation datasets, and A/B testing without building
   everything from scratch.

6. **Set alerting rules early.** Do not wait for a cost spike or latency degradation
   to be noticed by users. Monitor proactively.
