"""Supervisor Agent -- the main orchestrator that delegates tasks to
specialized agents (Extractor, Validator, Enricher, Researcher).

Uses LangGraph's supervisor multi-agent pattern with parallel execution
support for ingesting multiple documents simultaneously.

Reference: https://langchain-ai.github.io/langgraph/concepts/multi_agent/
"""

import sys
import json
from pathlib import Path
from typing import TypedDict, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
from shared.llm_clients import chat_completion_structured, chat_completion

from src.agents.extractor import run_extractor
from src.agents.validator import run_validator
from src.agents.enricher import run_enricher
from src.agents.researcher import run_researcher


# ── Schemas ──────────────────────────────────────────────────────────

class TaskPlan(BaseModel):
    """The supervisor's plan for handling a task."""
    task_type: Literal["ingest", "validate", "enrich", "research", "full_pipeline"] = Field(
        description="Type of task to perform"
    )
    agents_to_invoke: list[str] = Field(
        description="Ordered list of agents to invoke: extractor, validator, enricher, researcher"
    )
    parallel: bool = Field(
        description="Whether the first agent can run in parallel on multiple inputs"
    )
    reasoning: str = Field(description="Why this plan was chosen")


# ── LangGraph state ──────────────────────────────────────────────────

class SupervisorState(TypedDict):
    task: str
    documents: list[dict]  # List of {id, title, text}
    questions: list[str]
    plan: dict | None
    agent_results: list[dict]
    final_report: str
    current_step: int


# ── Graph nodes ──────────────────────────────────────────────────────

def plan_task(state: SupervisorState) -> dict:
    """Analyze the task and create an execution plan."""
    task = state["task"]
    has_docs = bool(state.get("documents"))
    has_questions = bool(state.get("questions"))

    plan = chat_completion_structured(
        prompt=f"""You are a supervisor agent managing a knowledge graph system.
Given the following task, decide which agents to invoke and in what order.

Task: {task}
Has documents to ingest: {has_docs} ({len(state.get('documents', []))} documents)
Has questions to answer: {has_questions} ({len(state.get('questions', []))} questions)

Available agents:
- extractor: Extracts entities/relationships from documents and writes to KG
- validator: Checks KG quality, fixes duplicates and type inconsistencies
- enricher: Finds gaps in the KG and fills them using web search
- researcher: Answers questions using the KG with dynamic tool selection

Plan the execution. For ingestion tasks, extraction can run in parallel.""",
        output_schema=TaskPlan,
        system="You are a multi-agent supervisor. Create efficient execution plans.",
    )

    return {
        "plan": plan.model_dump(),
        "current_step": 0,
        "agent_results": [],
    }


def execute_extraction(state: SupervisorState) -> dict:
    """Run the Extractor Agent, potentially in parallel for multiple docs."""
    documents = state.get("documents", [])
    results = list(state.get("agent_results") or [])

    if not documents:
        results.append({"agent": "extractor", "result": "No documents to extract."})
        return {"agent_results": results, "current_step": state["current_step"] + 1}

    plan = state.get("plan", {})
    use_parallel = plan.get("parallel", False) and len(documents) > 1

    if use_parallel:
        # Parallel extraction using ThreadPoolExecutor
        extraction_results = []
        with ThreadPoolExecutor(max_workers=min(4, len(documents))) as executor:
            futures = {
                executor.submit(
                    run_extractor, doc["id"], doc["title"], doc["text"]
                ): doc["id"]
                for doc in documents
            }
            for future in as_completed(futures):
                doc_id = futures[future]
                try:
                    summary = future.result()
                    extraction_results.append(f"[{doc_id}] {summary}")
                except Exception as e:
                    extraction_results.append(f"[{doc_id}] ERROR: {e}")

        results.append({
            "agent": "extractor",
            "result": f"Parallel extraction of {len(documents)} documents:\n" + "\n".join(extraction_results),
        })
    else:
        # Sequential extraction
        for doc in documents:
            try:
                summary = run_extractor(doc["id"], doc["title"], doc["text"])
                results.append({"agent": "extractor", "result": summary})
            except Exception as e:
                results.append({"agent": "extractor", "result": f"ERROR on {doc['id']}: {e}"})

    return {"agent_results": results, "current_step": state["current_step"] + 1}


def execute_validation(state: SupervisorState) -> dict:
    """Run the Validator Agent."""
    results = list(state.get("agent_results") or [])

    try:
        report = run_validator()
        results.append({"agent": "validator", "result": report})
    except Exception as e:
        results.append({"agent": "validator", "result": f"ERROR: {e}"})

    return {"agent_results": results, "current_step": state["current_step"] + 1}


def execute_enrichment(state: SupervisorState) -> dict:
    """Run the Enricher Agent."""
    results = list(state.get("agent_results") or [])

    try:
        report = run_enricher()
        results.append({"agent": "enricher", "result": report})
    except Exception as e:
        results.append({"agent": "enricher", "result": f"ERROR: {e}"})

    return {"agent_results": results, "current_step": state["current_step"] + 1}


def execute_research(state: SupervisorState) -> dict:
    """Run the Researcher Agent on all questions."""
    results = list(state.get("agent_results") or [])
    questions = state.get("questions", [])

    for question in questions:
        try:
            research_result = run_researcher(question)
            results.append({
                "agent": "researcher",
                "question": question,
                "result": research_result["answer"],
                "tools_used": research_result["tools_used"],
            })
        except Exception as e:
            results.append({
                "agent": "researcher",
                "question": question,
                "result": f"ERROR: {e}",
            })

    return {"agent_results": results, "current_step": state["current_step"] + 1}


def generate_final_report(state: SupervisorState) -> dict:
    """Generate a comprehensive report of all agent activities."""
    results = state["agent_results"]
    plan = state.get("plan", {})

    report_lines = [
        "=" * 60,
        "SUPERVISOR REPORT",
        "=" * 60,
        f"Task: {state['task']}",
        f"Plan: {plan.get('reasoning', 'N/A')}",
        f"Agents invoked: {plan.get('agents_to_invoke', [])}",
        "",
    ]

    for i, result in enumerate(results, 1):
        report_lines.append(f"--- Agent #{i}: {result['agent'].upper()} ---")
        if "question" in result:
            report_lines.append(f"Question: {result['question']}")
        report_lines.append(f"Result: {result['result']}")
        if "tools_used" in result:
            report_lines.append(f"Tools used: {', '.join(result['tools_used'])}")
        report_lines.append("")

    report_lines.append("=" * 60)
    return {"final_report": "\n".join(report_lines)}


# ── Routing logic ────────────────────────────────────────────────────

def route_next_agent(state: SupervisorState) -> str:
    """Determine which agent to execute next based on the plan."""
    plan = state.get("plan", {})
    agents = plan.get("agents_to_invoke", [])
    step = state.get("current_step", 0)

    if step >= len(agents):
        return "report"

    agent = agents[step]
    agent_map = {
        "extractor": "extract",
        "validator": "validate",
        "enricher": "enrich",
        "researcher": "research",
    }
    return agent_map.get(agent, "report")


# ── Build the graph ──────────────────────────────────────────────────

def build_supervisor_graph() -> StateGraph:
    """Build and compile the Supervisor Agent graph."""
    graph = StateGraph(SupervisorState)

    graph.add_node("plan", plan_task)
    graph.add_node("extract", execute_extraction)
    graph.add_node("validate", execute_validation)
    graph.add_node("enrich", execute_enrichment)
    graph.add_node("research", execute_research)
    graph.add_node("report", generate_final_report)

    graph.set_entry_point("plan")

    # After planning, route to the first agent
    graph.add_conditional_edges("plan", route_next_agent, {
        "extract": "extract",
        "validate": "validate",
        "enrich": "enrich",
        "research": "research",
        "report": "report",
    })

    # Each agent routes to the next one in the plan
    for node in ["extract", "validate", "enrich", "research"]:
        graph.add_conditional_edges(node, route_next_agent, {
            "extract": "extract",
            "validate": "validate",
            "enrich": "enrich",
            "research": "research",
            "report": "report",
        })

    graph.add_edge("report", END)

    return graph.compile()


def run_supervisor(
    task: str,
    documents: list[dict] | None = None,
    questions: list[str] | None = None,
) -> str:
    """Run the supervisor with a task.

    Args:
        task: Natural language description of what to do
        documents: List of dicts with keys: id, title, text
        questions: List of questions for the researcher

    Returns:
        Final report string
    """
    app = build_supervisor_graph()
    result = app.invoke({
        "task": task,
        "documents": documents or [],
        "questions": questions or [],
        "plan": None,
        "agent_results": [],
        "final_report": "",
        "current_step": 0,
    })
    return result["final_report"]


if __name__ == "__main__":
    report = run_supervisor(
        task="Ingest a test document and validate the results",
        documents=[{
            "id": "test_001",
            "title": "Test",
            "text": "OpenAI developed GPT-4 using the Transformer architecture.",
        }],
    )
    print(report)
