"""LangChain graph integrations for each database.

Demonstrates how to use LangChain's graph connectors to enable
natural-language queries against graph databases.

LangChain Graph docs:
- Neo4j: https://python.langchain.com/docs/integrations/graphs/neo4j_cypher/
- GraphCypherQAChain: https://python.langchain.com/docs/how_to/graph_cypher_qa/
- ArangoDB: https://python.langchain.com/docs/integrations/graphs/arangodb/
- FalkorDB: https://python.langchain.com/docs/integrations/graphs/falkordb/
- Memgraph: https://python.langchain.com/docs/integrations/graphs/memgraph/

Prerequisites:
    pip install langchain-neo4j langchain-community langchain-openai
    All databases running with data loaded.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared.llm_clients import get_chat_model


# ── Natural language questions to test across all databases ──────────────
NL_QUESTIONS = [
    "Which companies have developed AI models?",
    "What architecture does GPT-4 use?",
    "Which companies are connected to Google through investments or partnerships?",
    "How many models has each company developed?",
    "What hardware was used to train GPT-4?",
]


def demo_neo4j_langchain():
    """Use LangChain's Neo4jGraph + GraphCypherQAChain for NL-to-Cypher.

    Docs: https://python.langchain.com/docs/integrations/graphs/neo4j_cypher/
    The GraphCypherQAChain converts natural language to Cypher queries,
    executes them against Neo4j, and returns natural language answers.
    """
    print("=" * 70)
    print("NEO4J + LangChain GraphCypherQAChain")
    print("=" * 70)

    try:
        # Neo4jGraph extracts the schema automatically for the LLM
        # Docs: https://python.langchain.com/docs/integrations/graphs/neo4j_cypher/
        from langchain_neo4j import Neo4jGraph, GraphCypherQAChain

        graph = Neo4jGraph(
            url="bolt://localhost:7687",
            username="neo4j",
            password="password123",
        )

        # Print the auto-detected schema (node labels, relationship types, properties)
        print("\nDetected Graph Schema:")
        print(graph.schema)

        # Create the QA chain using our shared LLM client
        # Docs: https://python.langchain.com/docs/how_to/graph_cypher_qa/
        llm = get_chat_model(provider="openai", model="gpt-4o-mini")

        chain = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=graph,
            verbose=True,         # Show generated Cypher
            allow_dangerous_requests=True,  # Required for write queries
        )

        for question in NL_QUESTIONS:
            print(f"\nQ: {question}")
            try:
                result = chain.invoke({"query": question})
                print(f"A: {result['result']}")
            except Exception as e:
                print(f"Error: {e}")

        graph._driver.close()

    except ImportError:
        print("Install: pip install langchain-neo4j")
    except Exception as e:
        print(f"Neo4j unavailable: {e}")


def demo_arangodb_langchain():
    """Use LangChain's ArangoDB graph integration.

    Docs: https://python.langchain.com/docs/integrations/graphs/arangodb/
    """
    print("\n" + "=" * 70)
    print("ARANGODB + LangChain ArangoGraph")
    print("=" * 70)

    try:
        from langchain_community.graphs import ArangoGraph
        from langchain_community.chains.graph_qa.arangodb import ArangoGraphQAChain

        graph = ArangoGraph(
            url="http://localhost:8529",
            username="root",
            password="password123",
            db_name="kg_explorer",
        )

        print("\nDetected Graph Schema:")
        print(graph.schema)

        llm = get_chat_model(provider="openai", model="gpt-4o-mini")

        chain = ArangoGraphQAChain.from_llm(
            llm=llm,
            graph=graph,
            verbose=True,
            allow_dangerous_requests=True,
        )

        for question in NL_QUESTIONS[:3]:  # Run a subset
            print(f"\nQ: {question}")
            try:
                result = chain.invoke({"query": question})
                print(f"A: {result['result']}")
            except Exception as e:
                print(f"Error: {e}")

    except ImportError:
        print("Install: pip install langchain-community python-arango")
    except Exception as e:
        print(f"ArangoDB unavailable: {e}")


def demo_memgraph_langchain():
    """Use LangChain's Memgraph graph integration.

    Docs: https://python.langchain.com/docs/integrations/graphs/memgraph/
    Memgraph uses Cypher, so GraphCypherQAChain works with it too.
    """
    print("\n" + "=" * 70)
    print("MEMGRAPH + LangChain MemgraphGraph")
    print("=" * 70)

    try:
        from langchain_community.graphs import MemgraphGraph
        from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

        graph = MemgraphGraph(
            url="bolt://localhost:7688",
            username="",
            password="",
        )

        print("\nDetected Graph Schema:")
        print(graph.schema)

        llm = get_chat_model(provider="openai", model="gpt-4o-mini")

        chain = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=graph,
            verbose=True,
            allow_dangerous_requests=True,
        )

        for question in NL_QUESTIONS[:3]:
            print(f"\nQ: {question}")
            try:
                result = chain.invoke({"query": question})
                print(f"A: {result['result']}")
            except Exception as e:
                print(f"Error: {e}")

    except ImportError:
        print("Install: pip install langchain-community gqlalchemy")
    except Exception as e:
        print(f"Memgraph unavailable: {e}")


def demo_falkordb_langchain():
    """Use LangChain's FalkorDB graph integration.

    Docs: https://python.langchain.com/docs/integrations/graphs/falkordb/
    """
    print("\n" + "=" * 70)
    print("FALKORDB + LangChain FalkorDBGraph")
    print("=" * 70)

    try:
        from langchain_community.graphs import FalkorDBGraph
        from langchain_community.chains.graph_qa.falkordb import FalkorDBQAChain

        graph = FalkorDBGraph(
            database="kg_explorer",
            host="localhost",
            port=6379,
        )

        print("\nDetected Graph Schema:")
        print(graph.schema)

        llm = get_chat_model(provider="openai", model="gpt-4o-mini")

        chain = FalkorDBQAChain.from_llm(
            llm=llm,
            graph=graph,
            verbose=True,
            allow_dangerous_requests=True,
        )

        for question in NL_QUESTIONS[:3]:
            print(f"\nQ: {question}")
            try:
                result = chain.invoke({"query": question})
                print(f"A: {result['result']}")
            except Exception as e:
                print(f"Error: {e}")

    except ImportError:
        print("Install: pip install langchain-community falkordb")
    except Exception as e:
        print(f"FalkorDB unavailable: {e}")


def main():
    print("=" * 70)
    print("LANGCHAIN GRAPH DATABASE INTEGRATIONS")
    print("Natural Language -> Graph Query -> Natural Language Answer")
    print("=" * 70)
    print()
    print("This demo shows how LangChain can convert natural language questions")
    print("into database-specific query languages (Cypher, AQL, OpenCypher),")
    print("execute them, and return natural language answers.")
    print()

    demo_neo4j_langchain()
    demo_arangodb_langchain()
    demo_memgraph_langchain()
    demo_falkordb_langchain()

    print("\n" + "=" * 70)
    print("COMPARISON NOTES")
    print("=" * 70)
    print("""
    - Neo4j has the most mature LangChain integration (langchain-neo4j package)
    - ArangoDB, Memgraph, FalkorDB use langchain-community integrations
    - All support natural language to query translation via LLM
    - Schema auto-detection quality varies by database
    - Neo4j's GraphCypherQAChain is the most battle-tested
    - For production use, consider adding query validation and result caching
    """)


if __name__ == "__main__":
    main()
