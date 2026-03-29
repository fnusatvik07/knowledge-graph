"""Step 5: Global search with GraphRAG.

Performs community-summary based global search queries against the GraphRAG
index. Global search leverages hierarchical community summaries to answer
broad, thematic questions that require synthesizing information across the
entire corpus.
"""
# LangChain docs: https://python.langchain.com/docs/integrations/chat/

import argparse
import os
import sys
from pathlib import Path

# Add projects dir to path for shared imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared.config import get_openai_api_key
from shared.llm_clients import chat_completion

PROJECT_DIR = Path(__file__).resolve().parent.parent
GRAPHRAG_DIR = PROJECT_DIR / "output" / "graphrag"

# Example questions suited for global search (thematic, broad)
GLOBAL_SEARCH_QUESTIONS = [
    "What are the main strategies for reducing global carbon emissions?",
    "How are different sectors of the economy interconnected in addressing climate change?",
    "What role does government policy play in the clean energy transition?",
    "What are the major technological innovations driving sustainability?",
    "How do agriculture and energy systems interact in the context of climate solutions?",
]


def search_with_graphrag(question: str) -> str:
    """Run a global search query using GraphRAG CLI."""
    import subprocess

    env = os.environ.copy()
    env["GRAPHRAG_API_KEY"] = get_openai_api_key()

    try:
        result = subprocess.run(
            [
                "graphrag", "query",
                "--root", str(GRAPHRAG_DIR),
                "--method", "global",
                "--query", question,
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"Error: {result.stderr.strip()}"
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        return "Error: Query timed out after 180 seconds"


def search_with_fallback(question: str, community_summaries: list[str], provider: str = "openai") -> str:
    """Fallback global search using map-reduce over document summaries.

    Simulates GraphRAG's global search by:
    1. Generating a summary for each corpus document (community analog)
    2. MAP: Extracting relevant points from each summary
    3. REDUCE: Synthesizing all points into a comprehensive answer
    """
    # MAP phase: extract relevant information from each community/summary
    print("  MAP phase: analyzing each document community...")
    map_results = []

    for i, summary in enumerate(community_summaries):
        map_prompt = f"""Analyze the following document summary and extract key points
that are relevant to answering this question. If nothing is relevant, respond with "NO RELEVANT INFORMATION".

Question: {question}

Document Summary:
{summary}

Relevant key points (bullet list):"""

        result = chat_completion(map_prompt, provider=provider)
        if "NO RELEVANT INFORMATION" not in result.upper():
            map_results.append(result)
            print(f"    Community {i}: found relevant information")
        else:
            print(f"    Community {i}: no relevant information")

    if not map_results:
        return "No relevant information found across document communities."

    # REDUCE phase: synthesize all map results
    print("  REDUCE phase: synthesizing across communities...")
    combined_points = "\n\n".join(
        f"--- Community {i} ---\n{r}" for i, r in enumerate(map_results)
    )

    reduce_prompt = f"""You are synthesizing information gathered from multiple document
communities (topic clusters) to answer a broad thematic question.

Question: {question}

Information gathered from communities:
{combined_points}

Provide a comprehensive, well-structured answer that synthesizes information
across all communities. Highlight connections between different topics and
provide a holistic view.

Answer:"""

    answer = chat_completion(
        reduce_prompt,
        system="You are an expert analyst synthesizing information from a knowledge graph's "
               "community summaries. Provide comprehensive, well-organized answers that "
               "show connections across different topic areas.",
        provider=provider,
    )

    return answer


def generate_community_summaries(provider: str = "openai") -> list[str]:
    """Generate summaries for each corpus document to simulate communities."""
    from shared.document_loader import load_text_files

    corpus_dir = PROJECT_DIR / "data" / "corpus"
    documents = load_text_files(corpus_dir)

    summaries = []
    for doc in documents:
        summary = chat_completion(
            prompt=f"""Summarize the following document, focusing on key entities,
relationships between entities, and main themes. Keep the summary under 300 words.

Document: {doc['title']}

{doc['content']}

Summary:""",
            provider=provider,
        )
        summaries.append(f"[{doc['title']}]\n{summary}")

    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GraphRAG global search")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "ollama"],
        default="openai",
        help="LLM provider to use (default: openai)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    provider = args.provider

    print("=" * 60)
    print("GraphRAG Global Search")
    print("=" * 60)
    print()
    print(f"Using LLM provider: {provider}")
    print("Global search uses community summaries to answer broad, thematic")
    print("questions that require synthesizing information across the entire")
    print("corpus. It uses a map-reduce approach over community summaries.")
    print()

    api_key = get_openai_api_key()

    # Try GraphRAG CLI first
    graphrag_available = False
    try:
        import graphrag  # noqa: F401
        parquet_files = list(GRAPHRAG_DIR.rglob("*.parquet"))
        if parquet_files:
            graphrag_available = True
            print("GraphRAG index found. Using GraphRAG global search.")
        else:
            print("GraphRAG index not found. Using fallback map-reduce search.")
    except ImportError:
        print("GraphRAG not installed. Using fallback map-reduce search.")

    # Generate community summaries for fallback
    community_summaries = None
    if not graphrag_available:
        print("\nGenerating document community summaries...")
        community_summaries = generate_community_summaries(provider=provider)
        print(f"Generated {len(community_summaries)} community summaries.\n")

    results = {}

    for i, question in enumerate(GLOBAL_SEARCH_QUESTIONS, 1):
        print(f"\n{'='*60}")
        print(f"Question {i}: {question}")
        print("-" * 60)

        if graphrag_available:
            answer = search_with_graphrag(question)
            if answer is None:
                if community_summaries is None:
                    print("  Generating community summaries for fallback...")
                    community_summaries = generate_community_summaries(provider=provider)
                answer = search_with_fallback(question, community_summaries, provider=provider)
        else:
            answer = search_with_fallback(question, community_summaries, provider=provider)

        print(f"\nAnswer:\n{answer}")
        results[question] = answer

    # Summary
    print(f"\n\n{'='*60}")
    print("Global Search Summary")
    print("=" * 60)
    print(f"Questions answered: {len(results)}")
    print(f"Method: {'GraphRAG' if graphrag_available else 'Fallback map-reduce'}")
    print(f"\nGlobal search excels at:")
    print("  - Broad thematic questions ('What are the main strategies for X?')")
    print("  - Cross-cutting analysis ('How are sectors interconnected?')")
    print("  - Summarization across multiple topics")
    print("  - Questions requiring holistic corpus understanding")


if __name__ == "__main__":
    main()
