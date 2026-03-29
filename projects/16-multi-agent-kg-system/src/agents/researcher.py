"""Researcher Agent -- answers questions using the shared KG with DYNAMIC
tool selection. The LLM decides which tools to use (Cypher query, vector
search, multi-hop traversal, web search) based on the question type.

Uses LangGraph with LangChain tool-calling for flexible reasoning.
"""

import sys
from pathlib import Path
from typing import TypedDict, Annotated

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
from shared.llm_clients import get_chat_model

# Import the tools
from src.tools.kg_tools import (
    CypherQueryTool,
    EntitySearchTool,
    NeighborhoodTool,
    GapDetectionTool,
    KGStatsTool,
)
from src.tools.web_tools import TavilySearchTool, EntityResearchTool


# ── LangGraph state ──────────────────────────────────────────────────

class ResearcherState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    answer: str
    tools_used: list[str]
    reasoning_steps: list[str]


# ── Available tools ──────────────────────────────────────────────────

ALL_TOOLS = [
    CypherQueryTool(),
    EntitySearchTool(),
    NeighborhoodTool(),
    GapDetectionTool(),
    KGStatsTool(),
    TavilySearchTool(),
    EntityResearchTool(),
]

TOOL_MAP = {tool.name: tool for tool in ALL_TOOLS}


SYSTEM_PROMPT = """You are an expert research agent with access to a knowledge graph and web search.
Your job is to answer questions by dynamically selecting the right tools.

Strategy guidelines:
- For factual questions about entities: use entity_search first, then neighborhood_tool for context
- For relationship questions: use cypher_query with a targeted Cypher query
- For statistical questions: use kg_stats
- For questions requiring external information: use tavily_search or entity_research
- For complex multi-hop questions: combine multiple tools in sequence

Always explain your reasoning before using a tool. After gathering information,
synthesize a clear, comprehensive answer."""


# ── Graph nodes ──────────────────────────────────────────────────────

def call_model(state: ResearcherState) -> dict:
    """Call the LLM with tools available for dynamic selection."""
    llm = get_chat_model(provider="openai", model="gpt-4o-mini")
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    messages = state["messages"]
    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}


def execute_tools(state: ResearcherState) -> dict:
    """Execute whatever tools the LLM decided to call."""
    last_message = state["messages"][-1]
    tool_messages = []
    tools_used = list(state.get("tools_used") or [])
    reasoning_steps = list(state.get("reasoning_steps") or [])

    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {"messages": [], "tools_used": tools_used, "reasoning_steps": reasoning_steps}

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        tools_used.append(tool_name)
        reasoning_steps.append(f"Called {tool_name} with args: {tool_args}")

        tool = TOOL_MAP.get(tool_name)
        if tool:
            try:
                result = tool.invoke(tool_args)
                tool_messages.append(
                    ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                )
                reasoning_steps.append(f"Result from {tool_name}: {str(result)[:200]}...")
            except Exception as e:
                tool_messages.append(
                    ToolMessage(content=f"Error: {e}", tool_call_id=tool_call["id"])
                )
                reasoning_steps.append(f"Error from {tool_name}: {e}")
        else:
            tool_messages.append(
                ToolMessage(content=f"Unknown tool: {tool_name}", tool_call_id=tool_call["id"])
            )

    return {
        "messages": tool_messages,
        "tools_used": tools_used,
        "reasoning_steps": reasoning_steps,
    }


def format_answer(state: ResearcherState) -> dict:
    """Extract the final answer from the conversation."""
    last_message = state["messages"][-1]
    answer = last_message.content if hasattr(last_message, "content") else str(last_message)
    return {"answer": answer}


def should_continue(state: ResearcherState) -> str:
    """Check if the LLM wants to call more tools or is done."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "done"


# ── Build the graph ──────────────────────────────────────────────────

def build_researcher_graph() -> StateGraph:
    """Build and compile the Researcher Agent graph."""
    graph = StateGraph(ResearcherState)

    graph.add_node("model", call_model)
    graph.add_node("tools", execute_tools)
    graph.add_node("format", format_answer)

    graph.set_entry_point("model")
    graph.add_conditional_edges("model", should_continue, {
        "tools": "tools",
        "done": "format",
    })
    graph.add_edge("tools", "model")  # Loop back for multi-step reasoning
    graph.add_edge("format", END)

    return graph.compile()


def run_researcher(question: str) -> dict:
    """Run the researcher agent on a question.

    Returns:
        dict with keys: answer, tools_used, reasoning_steps
    """
    app = build_researcher_graph()
    result = app.invoke({
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=question),
        ],
        "question": question,
        "answer": "",
        "tools_used": [],
        "reasoning_steps": [],
    })
    return {
        "answer": result["answer"],
        "tools_used": result["tools_used"],
        "reasoning_steps": result["reasoning_steps"],
    }


if __name__ == "__main__":
    result = run_researcher("What technologies are related to knowledge graphs?")
    print(f"Answer: {result['answer']}")
    print(f"\nTools used: {result['tools_used']}")
    print(f"\nReasoning steps:")
    for step in result["reasoning_steps"]:
        print(f"  - {step}")
