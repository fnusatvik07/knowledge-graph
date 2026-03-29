"""LangChain-compatible tool for Tavily web search.

Wraps the Tavily API to provide a tool interface for LangGraph agents
to search the web for entity enrichment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from langchain_core.tools import BaseTool

from shared.config import get_tavily_api_key


class WebSearchTool(BaseTool):
    """Tool for searching the web via Tavily."""

    name: str = "web_search"
    description: str = (
        "Search the web for current information about a topic. "
        "Useful for finding the latest news, facts, or details about "
        "people, organizations, technologies, or events. "
        "Input should be a search query string."
    )

    def _run(self, query: str) -> str:
        """Execute a web search and return formatted results."""
        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=get_tavily_api_key())
            response = client.search(
                query=query,
                max_results=5,
                search_depth="basic",
            )

            results = []
            for item in response.get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "content": item.get("content", ""),
                    "url": item.get("url", ""),
                    "score": item.get("score", 0.0),
                })

            return json.dumps(results, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def _arun(self, query: str) -> str:
        """Async not implemented; falls back to sync."""
        return self._run(query)


class WebSearchEntityTool(BaseTool):
    """Tool for searching the web specifically about a named entity."""

    name: str = "web_search_entity"
    description: str = (
        "Search the web for information about a specific entity "
        "(person, company, technology, etc.). Input should be a JSON "
        "string with 'entity_name' and optional 'context' fields."
    )

    def _run(self, input_str: str) -> str:
        """Search for entity-specific information."""
        try:
            params = json.loads(input_str)
            entity_name = params["entity_name"]
            context = params.get("context", "")
        except (json.JSONDecodeError, KeyError):
            entity_name = input_str.strip()
            context = ""

        query = entity_name
        if context:
            query = f"{entity_name} {context}"

        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=get_tavily_api_key())
            response = client.search(
                query=query,
                max_results=3,
                search_depth="basic",
            )

            results = {
                "entity": entity_name,
                "results": [
                    {
                        "title": item.get("title", ""),
                        "content": item.get("content", "")[:500],
                        "url": item.get("url", ""),
                    }
                    for item in response.get("results", [])
                ],
            }
            return json.dumps(results, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def _arun(self, input_str: str) -> str:
        return self._run(input_str)
