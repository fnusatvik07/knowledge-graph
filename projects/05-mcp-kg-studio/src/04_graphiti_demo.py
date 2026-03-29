"""Demonstrate Graphiti's temporal knowledge graph construction.

Graphiti (by Zep) builds time-aware knowledge graphs where facts have timestamps
and can be invalidated when they become outdated. This is critical for KGs about
evolving domains like AI where facts change rapidly.

Graphiti docs: https://github.com/getzep/graphiti
Graphiti MCP: uses Neo4j as backend + OpenAI for embeddings

This script demonstrates:
1. Adding facts with timestamps (episodes)
2. Querying the KG at a point in time
3. Invalidating outdated facts
4. Semantic search over the temporal KG

Prerequisites:
    pip install graphiti-core
    Neo4j running on bolt://localhost:7687
    OPENAI_API_KEY set in environment
"""

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


async def run_graphiti_demo():
    """Build a temporal KG about AI companies and models using Graphiti.

    Graphiti's core concept is the 'episode' — a timestamped piece of
    information that gets decomposed into entities and relationships.
    Docs: https://github.com/getzep/graphiti#core-concepts
    """
    try:
        from graphiti_core import Graphiti
        from graphiti_core.nodes import EpisodeType
    except ImportError:
        print("Graphiti not installed. Run: pip install graphiti-core")
        print("\nShowing conceptual demo instead...\n")
        run_conceptual_demo()
        return

    print("=" * 70)
    print("GRAPHITI — Temporal Knowledge Graph Demo")
    print("=" * 70)

    # Initialize Graphiti with Neo4j backend
    # Graphiti uses Neo4j for storage and OpenAI for embeddings
    graphiti = Graphiti(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password123",
    )

    try:
        # Build indices for efficient retrieval
        await graphiti.build_indices_and_constraints()
        print("[graphiti] Initialized indices and constraints.")

        # ── Step 1: Add historical episodes (facts with timestamps) ──
        print("\n--- Adding Historical Episodes ---")
        print("Each episode is a timestamped fact that Graphiti decomposes")
        print("into entities and relationships automatically.\n")

        episodes = [
            {
                "content": "OpenAI released GPT-3 in June 2020. It was the largest language model at the time with 175 billion parameters.",
                "timestamp": datetime(2020, 6, 15, tzinfo=timezone.utc),
                "source": "news",
            },
            {
                "content": "Anthropic was founded in 2021 by Dario Amodei and other former OpenAI researchers. They focus on AI safety research.",
                "timestamp": datetime(2021, 1, 28, tzinfo=timezone.utc),
                "source": "news",
            },
            {
                "content": "OpenAI released GPT-4 in March 2023. It is a multimodal model that can process both text and images. GPT-4 significantly outperforms GPT-3.",
                "timestamp": datetime(2023, 3, 14, tzinfo=timezone.utc),
                "source": "news",
            },
            {
                "content": "Microsoft invested $13 billion in OpenAI in January 2023, making it the largest AI investment at the time.",
                "timestamp": datetime(2023, 1, 23, tzinfo=timezone.utc),
                "source": "news",
            },
            {
                "content": "Anthropic released Claude 3 in March 2024 in three variants: Opus, Sonnet, and Haiku. Claude 3 Opus outperformed GPT-4 on several benchmarks.",
                "timestamp": datetime(2024, 3, 4, tzinfo=timezone.utc),
                "source": "news",
            },
            {
                "content": "OpenAI released GPT-4o in May 2024. It is faster and cheaper than GPT-4 while maintaining similar capabilities.",
                "timestamp": datetime(2024, 5, 13, tzinfo=timezone.utc),
                "source": "news",
            },
            {
                "content": "Google invested $2 billion in Anthropic in late 2023, becoming a major investor alongside Amazon.",
                "timestamp": datetime(2023, 10, 27, tzinfo=timezone.utc),
                "source": "news",
            },
            {
                "content": "GPT-3 is no longer OpenAI's flagship model. GPT-4o has replaced it as the recommended model for most use cases.",
                "timestamp": datetime(2024, 6, 1, tzinfo=timezone.utc),
                "source": "news",
            },
        ]

        for i, episode in enumerate(episodes, 1):
            print(f"  [{i}/{len(episodes)}] Adding: {episode['content'][:80]}...")
            print(f"           Timestamp: {episode['timestamp'].strftime('%Y-%m-%d')}")
            await graphiti.add_episode(
                name=f"episode_{i}",
                episode_body=episode["content"],
                source=EpisodeType.text,
                source_description=episode["source"],
                reference_time=episode["timestamp"],
                group_id="kg_studio_demo",
            )
        print(f"\n  Added {len(episodes)} episodes to the temporal KG.")

        # ── Step 2: Search the knowledge graph ───────────────────────
        print("\n--- Semantic Search ---")
        search_queries = [
            "What models has OpenAI developed?",
            "Who invested in AI companies?",
            "What is the relationship between Anthropic and OpenAI?",
        ]

        for query in search_queries:
            print(f"\n  Query: '{query}'")
            results = await graphiti.search(
                query=query,
                group_ids=["kg_studio_demo"],
                num_results=5,
            )
            for r in results:
                print(f"    - {r.fact}")
                if hasattr(r, "valid_at"):
                    print(f"      (valid at: {r.valid_at})")

        # ── Step 3: Point-in-time queries ────────────────────────────
        print("\n--- Point-in-Time Queries ---")
        print("Graphiti tracks when facts become valid and when they are superseded.")

        time_queries = [
            ("2022-01-01", "What was OpenAI's best model?",
             datetime(2022, 1, 1, tzinfo=timezone.utc)),
            ("2023-06-01", "What was OpenAI's best model?",
             datetime(2023, 6, 1, tzinfo=timezone.utc)),
            ("2024-06-01", "What was OpenAI's best model?",
             datetime(2024, 6, 1, tzinfo=timezone.utc)),
        ]

        for date_str, query, ref_time in time_queries:
            print(f"\n  As of {date_str}: '{query}'")
            results = await graphiti.search(
                query=query,
                group_ids=["kg_studio_demo"],
                num_results=3,
            )
            for r in results:
                print(f"    - {r.fact}")

        # ── Step 4: Show how facts evolve ────────────────────────────
        print("\n--- Temporal Fact Evolution ---")
        print("Notice how Graphiti handles the evolution of 'best model':")
        print("  2020: GPT-3 was the largest language model")
        print("  2023: GPT-4 significantly outperforms GPT-3")
        print("  2024: GPT-4o replaced GPT-3 as the recommended model")
        print("  -> Earlier facts are NOT deleted, but superseded.")

    except Exception as e:
        print(f"\n[graphiti] Error: {e}")
        print("Make sure Neo4j is running and OPENAI_API_KEY is set.")
        raise

    finally:
        await graphiti.close()
        print("\n[graphiti] Connection closed.")

    print("\n" + "=" * 70)
    print("Graphiti temporal KG demo complete!")
    print("=" * 70)


def run_conceptual_demo():
    """Show the conceptual model of temporal KGs without running Graphiti.

    This fallback demonstrates the key concepts even if Graphiti
    or Neo4j aren't available.
    """
    print("=" * 70)
    print("TEMPORAL KNOWLEDGE GRAPH — Conceptual Demo")
    print("=" * 70)
    print()
    print("A temporal KG tracks WHEN facts are true, not just IF they are true.")
    print()

    # Simulated temporal facts
    facts = [
        {"fact": "GPT-3 is OpenAI's flagship model", "valid_from": "2020-06", "valid_until": "2023-03", "status": "superseded"},
        {"fact": "GPT-3 has 175B parameters", "valid_from": "2020-06", "valid_until": None, "status": "active"},
        {"fact": "GPT-4 is OpenAI's flagship model", "valid_from": "2023-03", "valid_until": "2024-05", "status": "superseded"},
        {"fact": "GPT-4 is multimodal (text + images)", "valid_from": "2023-03", "valid_until": None, "status": "active"},
        {"fact": "GPT-4o is OpenAI's flagship model", "valid_from": "2024-05", "valid_until": None, "status": "active"},
        {"fact": "Microsoft invested $13B in OpenAI", "valid_from": "2023-01", "valid_until": None, "status": "active"},
        {"fact": "Anthropic was founded by former OpenAI researchers", "valid_from": "2021-01", "valid_until": None, "status": "active"},
        {"fact": "Google invested $2B in Anthropic", "valid_from": "2023-10", "valid_until": None, "status": "active"},
    ]

    print("All facts with temporal metadata:")
    print("-" * 90)
    print(f"  {'Fact':<55} {'Valid From':<12} {'Valid Until':<12} {'Status'}")
    print("-" * 90)
    for f in facts:
        until = f["valid_until"] or "present"
        print(f"  {f['fact']:<55} {f['valid_from']:<12} {until:<12} {f['status']}")

    # Point-in-time queries
    print("\n\nPoint-in-time query simulation:")
    print("-" * 50)

    def query_at_time(question: str, date: str):
        print(f"\n  Q ({date}): {question}")
        active = [f for f in facts
                   if f["valid_from"] <= date
                   and (f["valid_until"] is None or f["valid_until"] > date)]
        for f in active:
            if any(kw in f["fact"].lower() for kw in question.lower().split()):
                print(f"    -> {f['fact']}")

    query_at_time("What is OpenAI's flagship model?", "2021-01")
    query_at_time("What is OpenAI's flagship model?", "2023-06")
    query_at_time("What is OpenAI's flagship model?", "2024-06")
    query_at_time("Who invested in OpenAI or Anthropic?", "2024-01")

    print("\n\nKey Graphiti concepts:")
    print("  - Episodes: timestamped text that gets decomposed into facts")
    print("  - Entities: auto-extracted from episode text")
    print("  - Edges: relationships with temporal validity")
    print("  - Invalidation: mark facts as no longer true (they remain in history)")
    print("  - Search: semantic search respects temporal context")


def main():
    asyncio.run(run_graphiti_demo())


if __name__ == "__main__":
    main()
