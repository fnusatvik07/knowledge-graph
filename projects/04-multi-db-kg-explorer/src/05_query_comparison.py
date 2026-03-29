"""Run the SAME queries across all 4 graph databases and compare results/timing.

This script connects to Neo4j, ArangoDB, Memgraph, and FalkorDB,
runs equivalent queries in each database's query language, and
displays a formatted comparison table.

Prerequisites:
    - All 4 databases running (docker compose up -d)
    - Data loaded into all databases (run scripts 01-04 first)
    pip install neo4j python-arango gqlalchemy falkordb tabulate
"""

import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tabulate import tabulate

# ── Database connections ─────────────────────────────────────────────────

def get_neo4j():
    from neo4j import GraphDatabase
    return GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password123"))

def get_arangodb():
    from arango import ArangoClient
    client = ArangoClient(hosts="http://localhost:8529")
    return client.db("kg_explorer", username="root", password="password123")

def get_memgraph():
    from gqlalchemy import Memgraph
    return Memgraph(host="localhost", port=7688)

def get_falkordb():
    from falkordb import FalkorDB
    db = FalkorDB(host="localhost", port=6379)
    return db.select_graph("kg_explorer")


# ── Query definitions ────────────────────────────────────────────────────
# Each query is defined in all 4 query languages.

QUERIES = [
    {
        "name": "Simple Lookup: Find OpenAI",
        "description": "Find a single entity by name",
        "neo4j": "MATCH (c:Company {name: 'OpenAI'}) RETURN c.name AS name, c.founded AS founded, c.sector AS sector",
        "memgraph": "MATCH (c:Company {name: 'OpenAI'}) RETURN c.name AS name, c.founded AS founded, c.sector AS sector",
        "falkordb": "MATCH (c:Company {name: 'OpenAI'}) RETURN c.name AS name, c.founded AS founded, c.sector AS sector",
        "arangodb": "FOR c IN Company FILTER c.name == 'OpenAI' RETURN {name: c.name, founded: c.founded, sector: c.sector}",
    },
    {
        "name": "Neighbor Traversal: OpenAI's models",
        "description": "1-hop: find all models developed by OpenAI",
        "neo4j": "MATCH (c:Company {name: 'OpenAI'})-[:DEVELOPED]->(m:Model) RETURN m.name AS model ORDER BY m.name",
        "memgraph": "MATCH (c:Company {name: 'OpenAI'})-[:DEVELOPED]->(m:Model) RETURN m.name AS model ORDER BY m.name",
        "falkordb": "MATCH (c:Company {name: 'OpenAI'})-[:DEVELOPED]->(m:Model) RETURN m.name AS model ORDER BY m.name",
        "arangodb": """
            FOR e IN relationships
                FILTER e.type == 'DEVELOPED' AND e._from == 'Company/openai'
                LET model = DOCUMENT(e._to)
                SORT model.name
                RETURN {model: model.name}
        """,
    },
    {
        "name": "2-Hop Path: Company -> Model -> Architecture",
        "description": "Which architectures are used by which company's models?",
        "neo4j": """
            MATCH (c:Company)-[:DEVELOPED]->(m:Model)-[:USES_ARCHITECTURE]->(a:Architecture)
            RETURN c.name AS company, m.name AS model, a.name AS architecture
            ORDER BY c.name, m.name
        """,
        "memgraph": """
            MATCH (c:Company)-[:DEVELOPED]->(m:Model)-[:USES_ARCHITECTURE]->(a:Architecture)
            RETURN c.name AS company, m.name AS model, a.name AS architecture
            ORDER BY c.name, m.name
        """,
        "falkordb": """
            MATCH (c:Company)-[:DEVELOPED]->(m:Model)-[:USES_ARCHITECTURE]->(a:Architecture)
            RETURN c.name AS company, m.name AS model, a.name AS architecture
            ORDER BY c.name, m.name
        """,
        "arangodb": """
            FOR e1 IN relationships
                FILTER e1.type == 'DEVELOPED'
                LET company = DOCUMENT(e1._from)
                LET model = DOCUMENT(e1._to)
                FOR e2 IN relationships
                    FILTER e2.type == 'USES_ARCHITECTURE' AND e2._from == e1._to
                    LET arch = DOCUMENT(e2._to)
                    SORT company.name, model.name
                    RETURN {company: company.name, model: model.name, architecture: arch.name}
        """,
    },
    {
        "name": "Aggregation: Models per company",
        "description": "Count how many models each company has developed",
        "neo4j": """
            MATCH (c:Company)-[:DEVELOPED]->(m:Model)
            RETURN c.name AS company, count(m) AS model_count
            ORDER BY model_count DESC
        """,
        "memgraph": """
            MATCH (c:Company)-[:DEVELOPED]->(m:Model)
            RETURN c.name AS company, count(m) AS model_count
            ORDER BY model_count DESC
        """,
        "falkordb": """
            MATCH (c:Company)-[:DEVELOPED]->(m:Model)
            RETURN c.name AS company, count(m) AS model_count
            ORDER BY model_count DESC
        """,
        "arangodb": """
            FOR e IN relationships
                FILTER e.type == 'DEVELOPED'
                LET company = DOCUMENT(e._from)
                COLLECT comp = company.name WITH COUNT INTO model_count
                SORT model_count DESC
                RETURN {company: comp, model_count: model_count}
        """,
    },
    {
        "name": "Pattern Matching: Companies connected via shared architecture",
        "description": "Find pairs of companies whose models use the same architecture",
        "neo4j": """
            MATCH (c1:Company)-[:DEVELOPED]->(:Model)-[:USES_ARCHITECTURE]->(a:Architecture)
                  <-[:USES_ARCHITECTURE]-(:Model)<-[:DEVELOPED]-(c2:Company)
            WHERE c1.name < c2.name
            RETURN DISTINCT c1.name AS company1, c2.name AS company2, a.name AS shared_arch
            ORDER BY c1.name
        """,
        "memgraph": """
            MATCH (c1:Company)-[:DEVELOPED]->(:Model)-[:USES_ARCHITECTURE]->(a:Architecture)
                  <-[:USES_ARCHITECTURE]-(:Model)<-[:DEVELOPED]-(c2:Company)
            WHERE c1.name < c2.name
            RETURN DISTINCT c1.name AS company1, c2.name AS company2, a.name AS shared_arch
            ORDER BY c1.name
        """,
        "falkordb": """
            MATCH (c1:Company)-[:DEVELOPED]->(:Model)-[:USES_ARCHITECTURE]->(a:Architecture)
                  <-[:USES_ARCHITECTURE]-(:Model)<-[:DEVELOPED]-(c2:Company)
            WHERE c1.name < c2.name
            RETURN DISTINCT c1.name AS company1, c2.name AS company2, a.name AS shared_arch
            ORDER BY c1.name
        """,
        "arangodb": """
            FOR e1 IN relationships
                FILTER e1.type == 'USES_ARCHITECTURE'
                FOR e2 IN relationships
                    FILTER e2.type == 'USES_ARCHITECTURE' AND e2._to == e1._to AND e2._from != e1._from
                    FOR d1 IN relationships
                        FILTER d1.type == 'DEVELOPED' AND d1._to == e1._from
                        FOR d2 IN relationships
                            FILTER d2.type == 'DEVELOPED' AND d2._to == e2._from
                            LET c1 = DOCUMENT(d1._from)
                            LET c2 = DOCUMENT(d2._from)
                            LET arch = DOCUMENT(e1._to)
                            FILTER c1.name < c2.name
                            RETURN DISTINCT {company1: c1.name, company2: c2.name, shared_arch: arch.name}
        """,
    },
]


# ── Query execution helpers ──────────────────────────────────────────────

def time_query(func) -> tuple[list[dict], float]:
    """Execute a query function and return (results, elapsed_ms)."""
    start = time.perf_counter()
    results = func()
    elapsed = (time.perf_counter() - start) * 1000
    return results, elapsed


def run_neo4j(driver, cypher: str) -> list[dict]:
    with driver.session() as session:
        result = session.run(cypher)
        return [dict(r) for r in result]


def run_arangodb(db, aql: str) -> list[dict]:
    cursor = db.aql.execute(aql)
    return list(cursor)


def run_memgraph(mg, cypher: str) -> list[dict]:
    results = list(mg.execute_and_fetch(cypher))
    return [dict(r) for r in results]


def run_falkordb(graph, cypher: str) -> list[dict]:
    result = graph.query(cypher)
    # FalkorDB returns result_set as list of lists with header
    if result.result_set:
        headers = result.header
        return [dict(zip(headers, row)) for row in result.result_set]
    return []


# ── Main comparison ──────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("MULTI-DATABASE QUERY COMPARISON")
    print("=" * 80)
    print()

    # Connect to all databases
    connections = {}
    db_names = ["Neo4j", "ArangoDB", "Memgraph", "FalkorDB"]

    try:
        connections["Neo4j"] = get_neo4j()
        print("[+] Connected to Neo4j")
    except Exception as e:
        print(f"[-] Neo4j unavailable: {e}")

    try:
        connections["ArangoDB"] = get_arangodb()
        print("[+] Connected to ArangoDB")
    except Exception as e:
        print(f"[-] ArangoDB unavailable: {e}")

    try:
        connections["Memgraph"] = get_memgraph()
        print("[+] Connected to Memgraph")
    except Exception as e:
        print(f"[-] Memgraph unavailable: {e}")

    try:
        connections["FalkorDB"] = get_falkordb()
        print("[+] Connected to FalkorDB")
    except Exception as e:
        print(f"[-] FalkorDB unavailable: {e}")

    if not connections:
        print("\nNo databases available. Run: docker compose up -d")
        return

    # Run each query across all available databases
    timing_summary = []

    for query_def in QUERIES:
        print(f"\n{'─' * 80}")
        print(f"Query: {query_def['name']}")
        print(f"Description: {query_def['description']}")
        print(f"{'─' * 80}")

        timing_row = {"Query": query_def["name"]}

        for db_name in db_names:
            if db_name not in connections:
                timing_row[db_name] = "N/A"
                continue

            query_key = db_name.lower().replace("falkordb", "falkordb")
            query_text = query_def.get(query_key, query_def.get(db_name.lower()))

            try:
                conn = connections[db_name]
                if db_name == "Neo4j":
                    results, elapsed = time_query(lambda c=conn, q=query_text: run_neo4j(c, q))
                elif db_name == "ArangoDB":
                    results, elapsed = time_query(lambda c=conn, q=query_text: run_arangodb(c, q))
                elif db_name == "Memgraph":
                    results, elapsed = time_query(lambda c=conn, q=query_text: run_memgraph(c, q))
                elif db_name == "FalkorDB":
                    results, elapsed = time_query(lambda c=conn, q=query_text: run_falkordb(c, q))

                timing_row[db_name] = f"{elapsed:.2f}ms"
                print(f"\n  [{db_name}] ({elapsed:.2f}ms) — {len(results)} results")
                for r in results[:5]:  # Show first 5 results
                    print(f"    {r}")
                if len(results) > 5:
                    print(f"    ... and {len(results) - 5} more")

            except Exception as e:
                timing_row[db_name] = "ERROR"
                print(f"\n  [{db_name}] ERROR: {e}")

        timing_summary.append(timing_row)

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n\n{'=' * 80}")
    print("TIMING COMPARISON SUMMARY")
    print("=" * 80)
    print()
    print(tabulate(timing_summary, headers="keys", tablefmt="grid"))

    # ── Query language comparison ────────────────────────────────────────
    print(f"\n\n{'=' * 80}")
    print("QUERY LANGUAGE COMPARISON")
    print("=" * 80)

    comparison = [
        ["Feature", "Neo4j (Cypher)", "ArangoDB (AQL)", "Memgraph (Cypher)", "FalkorDB (OpenCypher)"],
        ["Pattern matching", "MATCH (a)-[r]->(b)", "FOR v,e,p IN ... GRAPH", "MATCH (a)-[r]->(b)", "MATCH (a)-[r]->(b)"],
        ["Filtering", "WHERE a.x = 1", "FILTER doc.x == 1", "WHERE a.x = 1", "WHERE a.x = 1"],
        ["Aggregation", "count(), collect()", "COLLECT, AGGREGATE", "count(), collect()", "count(), collect()"],
        ["Path queries", "shortestPath()", "SHORTEST_PATH", "BFS/DFS built-in", "shortestPath()"],
        ["Indexes", "CREATE INDEX", "ensureIndex()", "CREATE INDEX", "CREATE INDEX"],
        ["Full-text search", "APOC / db.index.fulltext", "ANALYZER + VIEW", "Not built-in", "Not built-in"],
        ["Stored procedures", "APOC library", "UDFs (JavaScript)", "Query modules (C/Python)", "Not supported"],
    ]

    print()
    print(tabulate(comparison, headers="firstrow", tablefmt="grid"))

    # Cleanup Neo4j driver
    if "Neo4j" in connections:
        connections["Neo4j"].close()

    print("\nDone.")


if __name__ == "__main__":
    main()
