"""Web search tools for the multi-agent KG system.

LangChain BaseTool subclasses that wrap Tavily for web search,
with descriptions optimized for LLM tool selection.
"""

import sys
from pathlib import Path
from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
from shared.config import get_tavily_api_key


# ── Tool Input Schemas ───────────────────────────────────────────────

class TavilySearchInput(BaseModel):
    query: str = Field(description="The search query to find information on the web")
    max_results: int = Field(default=5, description="Maximum number of results to return (1-10)")


class EntityResearchInput(BaseModel):
    entity_name: str = Field(description="The name of the entity to research")
    context: str = Field(default="", description="Additional context about the entity to narrow the search")


# ── Tools ────────────────────────────────────────────────────────────

class TavilySearchTool(BaseTool):
    """Search the web for information using Tavily."""

    name: str = "tavily_search"
    description: str = (
        "Search the web for general information using Tavily. "
        "Use this when the knowledge graph does not contain enough information "
        "to answer a question, or when you need up-to-date information. "
        "Returns a list of search results with titles, URLs, and content snippets."
    )
    args_schema: Type[BaseModel] = TavilySearchInput

    def _run(self, query: str, max_results: int = 5) -> str:
        from tavily import TavilyClient

        try:
            client = TavilyClient(api_key=get_tavily_api_key())
            result = client.search(
                query=query,
                max_results=min(max_results, 10),
                search_depth="basic",
            )

            formatted_results = []
            for r in result.get("results", []):
                formatted_results.append(
                    f"Title: {r.get('title', 'N/A')}\n"
                    f"URL: {r.get('url', 'N/A')}\n"
                    f"Content: {r.get('content', 'N/A')}\n"
                )

            if not formatted_results:
                return f"No results found for query: {query}"

            return "\n---\n".join(formatted_results)
        except Exception as e:
            return f"Search error: {e}"


class EntityResearchTool(BaseTool):
    """Research a specific entity by searching for detailed information."""

    name: str = "entity_research"
    description: str = (
        "Research a specific entity (person, organization, technology, concept) "
        "by searching for detailed information about it. Use this when you need "
        "in-depth information about a particular entity that is not fully covered "
        "in the knowledge graph. Returns detailed search results."
    )
    args_schema: Type[BaseModel] = EntityResearchInput

    def _run(self, entity_name: str, context: str = "") -> str:
        from tavily import TavilyClient

        try:
            client = TavilyClient(api_key=get_tavily_api_key())

            # Build a targeted search query
            search_query = entity_name
            if context:
                search_query = f"{entity_name} {context}"

            result = client.search(
                query=search_query,
                max_results=5,
                search_depth="advanced",
            )

            formatted_results = []
            for r in result.get("results", []):
                formatted_results.append(
                    f"Title: {r.get('title', 'N/A')}\n"
                    f"URL: {r.get('url', 'N/A')}\n"
                    f"Content: {r.get('content', 'N/A')}\n"
                )

            if not formatted_results:
                return f"No detailed information found for entity: {entity_name}"

            return (
                f"Research results for '{entity_name}':\n\n"
                + "\n---\n".join(formatted_results)
            )
        except Exception as e:
            return f"Entity research error: {e}"
