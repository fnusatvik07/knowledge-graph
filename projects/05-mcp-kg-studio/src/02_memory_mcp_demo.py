"""Demonstrate building a knowledge graph using the MCP Memory protocol.

The MCP Memory Server stores a simple knowledge graph as JSONL with:
- Entities (with observations/facts)
- Relations (typed edges between entities)

MCP Memory Server: https://github.com/modelcontextprotocol/servers/tree/main/src/memory
MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk

This script communicates with the Memory MCP server programmatically
using the MCP Python SDK, simulating what Claude does when it uses
the memory tools conversationally.

Prerequisites:
    pip install mcp httpx
    npm install -g @modelcontextprotocol/server-memory
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


async def run_memory_mcp_demo():
    """Demonstrate the MCP Memory server's knowledge graph capabilities.

    The MCP SDK provides a client that connects to MCP servers via stdio.
    Docs: https://modelcontextprotocol.io/docs/concepts/architecture
    """
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:
        print("MCP Python SDK not installed. Run: pip install mcp")
        print("\nFalling back to manual JSONL demo...\n")
        run_jsonl_fallback()
        return

    # Configure the Memory MCP server to launch via stdio
    # Docs: https://modelcontextprotocol.io/docs/concepts/transports
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-memory"],
        env={"MEMORY_FILE_PATH": str(
            Path(__file__).resolve().parent.parent / "data" / "memory_kg.jsonl"
        )},
    )

    print("=" * 70)
    print("MCP MEMORY SERVER — Knowledge Graph Demo")
    print("=" * 70)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the MCP session
            await session.initialize()

            # List available tools
            tools = await session.list_tools()
            print("\nAvailable MCP Tools:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description[:80]}...")

            # ── Step 1: Create entities ──────────────────────────────
            print("\n--- Creating Entities ---")
            entities_to_create = [
                {
                    "name": "OpenAI",
                    "entityType": "Company",
                    "observations": [
                        "Founded in 2015 in San Francisco",
                        "Develops GPT series of language models",
                        "Partnered with Microsoft for cloud computing",
                        "Created ChatGPT in November 2022",
                    ],
                },
                {
                    "name": "Anthropic",
                    "entityType": "Company",
                    "observations": [
                        "Founded in 2021 by former OpenAI researchers",
                        "Develops Claude AI assistant",
                        "Pioneered Constitutional AI approach",
                        "Headquartered in San Francisco",
                    ],
                },
                {
                    "name": "GPT-4",
                    "entityType": "Model",
                    "observations": [
                        "Released in March 2023",
                        "Multimodal model supporting text and images",
                        "Uses Transformer architecture",
                        "Trained with RLHF",
                    ],
                },
                {
                    "name": "Claude 3",
                    "entityType": "Model",
                    "observations": [
                        "Released in March 2024",
                        "Available in Opus, Sonnet, and Haiku variants",
                        "Trained with Constitutional AI",
                        "Supports 200K context window",
                    ],
                },
                {
                    "name": "Transformer",
                    "entityType": "Architecture",
                    "observations": [
                        "Introduced in 'Attention Is All You Need' (2017)",
                        "Foundation of modern large language models",
                        "Uses self-attention mechanism",
                    ],
                },
            ]

            result = await session.call_tool(
                "create_entities",
                arguments={"entities": entities_to_create},
            )
            print(f"  Created {len(entities_to_create)} entities")
            print(f"  Response: {result.content[0].text[:200]}")

            # ── Step 2: Create relations ─────────────────────────────
            print("\n--- Creating Relations ---")
            relations_to_create = [
                {"from": "OpenAI", "to": "GPT-4", "relationType": "developed"},
                {"from": "Anthropic", "to": "Claude 3", "relationType": "developed"},
                {"from": "GPT-4", "to": "Transformer", "relationType": "uses_architecture"},
                {"from": "Claude 3", "to": "Transformer", "relationType": "uses_architecture"},
                {"from": "Anthropic", "to": "OpenAI", "relationType": "founded_by_alumni_of"},
            ]

            result = await session.call_tool(
                "create_relations",
                arguments={"relations": relations_to_create},
            )
            print(f"  Created {len(relations_to_create)} relations")
            print(f"  Response: {result.content[0].text[:200]}")

            # ── Step 3: Add observations to existing entities ────────
            print("\n--- Adding Observations ---")
            result = await session.call_tool(
                "add_observations",
                arguments={
                    "observations": [
                        {
                            "entityName": "OpenAI",
                            "contents": [
                                "Valued at $157 billion as of 2024",
                                "Sam Altman serves as CEO",
                            ],
                        }
                    ]
                },
            )
            print(f"  Added observations to OpenAI")
            print(f"  Response: {result.content[0].text[:200]}")

            # ── Step 4: Search the knowledge graph ───────────────────
            print("\n--- Searching Knowledge Graph ---")
            search_queries = [
                "AI companies",
                "Transformer architecture",
                "language models",
            ]

            for query in search_queries:
                result = await session.call_tool(
                    "search_nodes",
                    arguments={"query": query},
                )
                print(f"\n  Search: '{query}'")
                print(f"  Results: {result.content[0].text[:300]}")

            # ── Step 5: Open specific nodes ──────────────────────────
            print("\n--- Opening Specific Nodes ---")
            result = await session.call_tool(
                "open_nodes",
                arguments={"names": ["OpenAI", "GPT-4"]},
            )
            print(f"  Nodes: OpenAI, GPT-4")
            print(f"  Details: {result.content[0].text[:400]}")

            print("\n" + "=" * 70)
            print("Memory MCP demo complete!")
            print(f"KG stored at: data/memory_kg.jsonl")
            print("=" * 70)


def run_jsonl_fallback():
    """Fallback: demonstrate the JSONL format used by Memory MCP.

    If the MCP SDK isn't installed, we can still show the data format
    and build a compatible JSONL file manually.
    """
    print("=" * 70)
    print("MCP MEMORY FORMAT — JSONL Fallback Demo")
    print("=" * 70)
    print()
    print("The MCP Memory Server stores KGs as JSONL with this format:")
    print()

    # The Memory server uses a simple JSONL format
    kg_data = {
        "entities": [
            {
                "name": "OpenAI",
                "entityType": "Company",
                "observations": [
                    "Founded in 2015",
                    "Develops GPT models",
                    "Partnered with Microsoft",
                ],
            },
            {
                "name": "Anthropic",
                "entityType": "Company",
                "observations": [
                    "Founded in 2021",
                    "Develops Claude AI",
                    "Pioneered Constitutional AI",
                ],
            },
            {
                "name": "GPT-4",
                "entityType": "Model",
                "observations": [
                    "Released March 2023",
                    "Multimodal",
                    "Transformer-based",
                ],
            },
        ],
        "relations": [
            {"from": "OpenAI", "to": "GPT-4", "relationType": "developed"},
            {"from": "Anthropic", "to": "OpenAI", "relationType": "founded_by_alumni_of"},
        ],
    }

    # Write as JSONL (one JSON object per line — the Memory server format)
    output_path = Path(__file__).resolve().parent.parent / "data" / "memory_kg.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(json.dumps(kg_data) + "\n")

    print(f"  Written to: {output_path}")
    print()
    print("  Format structure:")
    print(json.dumps(kg_data, indent=2))
    print()
    print("  To use with the Memory MCP server:")
    print(f"    MEMORY_FILE_PATH={output_path}")
    print("    npx -y @modelcontextprotocol/server-memory")


def main():
    asyncio.run(run_memory_mcp_demo())


if __name__ == "__main__":
    main()
