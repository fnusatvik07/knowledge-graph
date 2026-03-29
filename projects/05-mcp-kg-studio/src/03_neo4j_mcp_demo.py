"""Demonstrate querying Neo4j through the MCP protocol.

The Neo4j MCP server (https://github.com/neo4j/mcp-neo4j) exposes tools for:
- read_neo4j_cypher: Execute read-only Cypher queries
- write_neo4j_cypher: Execute write Cypher queries
- get_neo4j_schema: Get the database schema

This script uses the MCP Python SDK to communicate with the Neo4j MCP server
programmatically, demonstrating the same interaction pattern that Claude uses
when it has the Neo4j MCP server configured.

MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk

Prerequisites:
    pip install mcp neo4j
    Neo4j running on bolt://localhost:7687 (with data from Project 4)
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


async def run_neo4j_mcp_demo():
    """Connect to Neo4j MCP server and execute graph operations.

    The MCP client connects to the server via stdio transport.
    Docs: https://modelcontextprotocol.io/docs/concepts/transports
    """
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:
        print("MCP Python SDK not installed. Run: pip install mcp")
        print("\nFalling back to direct Neo4j demo...\n")
        run_direct_neo4j_fallback()
        return

    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@neo4j/mcp-neo4j"],
        env={
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USERNAME": "neo4j",
            "NEO4J_PASSWORD": "password123",
        },
    )

    print("=" * 70)
    print("NEO4J MCP SERVER — Graph Query Demo")
    print("=" * 70)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List available tools from the Neo4j MCP server
            tools = await session.list_tools()
            print("\nAvailable Neo4j MCP Tools:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description[:80]}...")

            # ── Step 1: Get the database schema ─────────────────────
            print("\n--- Database Schema ---")
            result = await session.call_tool("get_neo4j_schema", arguments={})
            schema_text = result.content[0].text
            print(f"  Schema:\n{schema_text[:500]}")

            # ── Step 2: Read queries ─────────────────────────────────
            print("\n--- Read Queries via MCP ---")

            read_queries = [
                {
                    "name": "Count all nodes",
                    "cypher": "MATCH (n) RETURN count(n) AS total_nodes",
                },
                {
                    "name": "List companies",
                    "cypher": "MATCH (c:Company) RETURN c.name AS company ORDER BY c.name",
                },
                {
                    "name": "Models and their creators",
                    "cypher": """
                        MATCH (c:Company)-[:DEVELOPED]->(m:Model)
                        RETURN c.name AS company, m.name AS model
                        ORDER BY c.name
                    """,
                },
                {
                    "name": "2-hop: Companies sharing architectures",
                    "cypher": """
                        MATCH (c1:Company)-[:DEVELOPED]->(:Model)-[:USES_ARCHITECTURE]->(a:Architecture)
                              <-[:USES_ARCHITECTURE]-(:Model)<-[:DEVELOPED]-(c2:Company)
                        WHERE c1.name < c2.name
                        RETURN DISTINCT c1.name AS company1, c2.name AS company2, a.name AS architecture
                    """,
                },
                {
                    "name": "Investment relationships",
                    "cypher": """
                        MATCH (investor:Company)-[r:INVESTED_IN]->(target:Company)
                        RETURN investor.name AS investor, target.name AS target, r.amount AS amount
                    """,
                },
            ]

            for q in read_queries:
                print(f"\n  Query: {q['name']}")
                print(f"  Cypher: {q['cypher'].strip()[:100]}...")
                try:
                    result = await session.call_tool(
                        "read_neo4j_cypher",
                        arguments={"query": q["cypher"]},
                    )
                    print(f"  Result: {result.content[0].text[:300]}")
                except Exception as e:
                    print(f"  Error: {e}")

            # ── Step 3: Write query (add a new entity) ───────────────
            print("\n--- Write Query via MCP ---")
            write_cypher = """
                MERGE (t:Technique {id: 'dpo'})
                SET t.name = 'DPO',
                    t.year_introduced = 2023,
                    t.paper = 'Direct Preference Optimization'
                RETURN t.name AS created
            """
            print(f"  Creating new entity: DPO technique")
            try:
                result = await session.call_tool(
                    "write_neo4j_cypher",
                    arguments={"query": write_cypher},
                )
                print(f"  Result: {result.content[0].text[:200]}")
            except Exception as e:
                print(f"  Error: {e}")

            # ── Step 4: Verify the write ─────────────────────────────
            print("\n--- Verify Write ---")
            verify_cypher = "MATCH (t:Technique {id: 'dpo'}) RETURN t.name, t.year_introduced"
            try:
                result = await session.call_tool(
                    "read_neo4j_cypher",
                    arguments={"query": verify_cypher},
                )
                print(f"  Result: {result.content[0].text[:200]}")
            except Exception as e:
                print(f"  Error: {e}")

            print("\n" + "=" * 70)
            print("Neo4j MCP demo complete!")
            print("=" * 70)


def run_direct_neo4j_fallback():
    """Fallback: query Neo4j directly to show what MCP does behind the scenes."""
    print("=" * 70)
    print("NEO4J DIRECT QUERIES (MCP Fallback)")
    print("=" * 70)
    print()
    print("Without the MCP SDK, here is what the Neo4j MCP server does internally:")
    print()

    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            "bolt://localhost:7687", auth=("neo4j", "password123")
        )
        driver.verify_connectivity()

        queries = [
            ("Node count", "MATCH (n) RETURN count(n) AS total"),
            ("Companies", "MATCH (c:Company) RETURN c.name ORDER BY c.name"),
            ("Models", "MATCH (c:Company)-[:DEVELOPED]->(m:Model) RETURN c.name, m.name"),
        ]

        with driver.session() as session:
            for name, cypher in queries:
                result = session.run(cypher)
                records = [dict(r) for r in result]
                print(f"  {name}: {records[:5]}")

        driver.close()
        print()
        print("  The Neo4j MCP server wraps these same queries as MCP tools,")
        print("  allowing Claude to execute them conversationally.")

    except Exception as e:
        print(f"  Neo4j not available: {e}")
        print("  Start Neo4j: docker compose up -d neo4j")


def main():
    asyncio.run(run_neo4j_mcp_demo())


if __name__ == "__main__":
    main()
