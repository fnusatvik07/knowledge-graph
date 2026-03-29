"""Step 4: Local search with GraphRAG.

Performs entity-focused local search queries against the GraphRAG index.
Local search retrieves information from specific entities and their
neighborhoods in the knowledge graph, ideal for targeted questions
about particular concepts, organizations, or technologies.
"""

import os
import sys
from pathlib import Path

# Add projects dir to path for shared imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared.config import get_openai_api_key
from shared.llm_clients import chat_completion

PROJECT_DIR = Path(__file__).resolve().parent.parent
GRAPHRAG_DIR = PROJECT_DIR / "output" / "graphrag"

# Example questions suited for local search (entity-focused)
LOCAL_SEARCH_QUESTIONS = [
    "What is the role of Tesla in the electric vehicle industry?",
    "How does the Inflation Reduction Act support clean energy?",
    "What is Climeworks and what technology do they use?",
    "What are lithium-ion batteries used for in renewable energy?",
    "How does the Paris Agreement address climate change?",
]


def search_with_graphrag(question: str) -> str:
    """Run a local search query using GraphRAG CLI."""
    import subprocess

    env = os.environ.copy()
    env["GRAPHRAG_API_KEY"] = get_openai_api_key()

    try:
        result = subprocess.run(
            [
                "graphrag", "query",
                "--root", str(GRAPHRAG_DIR),
                "--method", "local",
                "--query", question,
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"Error: {result.stderr.strip()}"
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        return "Error: Query timed out after 120 seconds"


def search_with_fallback(question: str, corpus_context: str) -> str:
    """Fallback local search using entity extraction + targeted retrieval.

    When GraphRAG CLI is not available, simulates local search by:
    1. Extracting key entities from the question
    2. Finding relevant passages mentioning those entities
    3. Generating an answer from the entity-focused context
    """
    # Step 1: Extract entities from the question
    entity_prompt = f"""Extract the key entities (people, organizations, technologies,
events, concepts) from this question. Return them as a comma-separated list.

Question: {question}

Entities:"""
    entities_str = chat_completion(entity_prompt, model="gpt-4o-mini")
    entities = [e.strip() for e in entities_str.split(",")]

    print(f"  Extracted entities: {entities}")

    # Step 2: Find relevant passages containing these entities
    relevant_passages = []
    corpus_lower = corpus_context.lower()
    for entity in entities:
        entity_lower = entity.lower()
        # Find sentences containing the entity
        for sentence in corpus_context.split(". "):
            if entity_lower in sentence.lower() and sentence not in relevant_passages:
                relevant_passages.append(sentence.strip())

    # Limit context size
    context = ". ".join(relevant_passages[:20])
    if not context:
        context = corpus_context[:3000]

    # Step 3: Generate answer from entity-focused context
    answer = chat_completion(
        prompt=f"""Based on the following entity-focused context, answer the question.
Focus on the specific entities mentioned and their relationships.

Context (entity-focused passages):
{context}

Question: {question}

Answer:""",
        system="You are a knowledge graph analyst. Answer questions by focusing on "
               "specific entities and their relationships. Be precise and cite "
               "specific facts from the context.",
        model="gpt-4o-mini",
    )

    return answer


def load_corpus() -> str:
    """Load all corpus documents into a single string."""
    from shared.document_loader import load_text_files

    corpus_dir = PROJECT_DIR / "data" / "corpus"
    documents = load_text_files(corpus_dir)
    return "\n\n".join(doc["content"] for doc in documents)


def main():
    print("=" * 60)
    print("GraphRAG Local Search")
    print("=" * 60)
    print()
    print("Local search focuses on specific entities and their immediate")
    print("neighborhood in the knowledge graph. Best for targeted questions")
    print("about particular concepts, organizations, or technologies.")
    print()

    api_key = get_openai_api_key()

    # Try GraphRAG CLI first
    graphrag_available = False
    try:
        import graphrag  # noqa: F401
        # Check if index exists
        parquet_files = list(GRAPHRAG_DIR.rglob("*.parquet"))
        if parquet_files:
            graphrag_available = True
            print("GraphRAG index found. Using GraphRAG local search.")
        else:
            print("GraphRAG index not found. Using fallback entity-focused search.")
    except ImportError:
        print("GraphRAG not installed. Using fallback entity-focused search.")

    # Load corpus for fallback
    corpus_context = load_corpus()

    print()
    results = {}

    for i, question in enumerate(LOCAL_SEARCH_QUESTIONS, 1):
        print(f"\n{'='*60}")
        print(f"Question {i}: {question}")
        print("-" * 60)

        if graphrag_available:
            answer = search_with_graphrag(question)
            if answer is None:
                # CLI not found, switch to fallback
                print("  GraphRAG CLI not available, using fallback...")
                answer = search_with_fallback(question, corpus_context)
        else:
            answer = search_with_fallback(question, corpus_context)

        print(f"\nAnswer:\n{answer}")
        results[question] = answer

    # Summary
    print(f"\n\n{'='*60}")
    print("Local Search Summary")
    print("=" * 60)
    print(f"Questions answered: {len(results)}")
    print(f"Method: {'GraphRAG' if graphrag_available else 'Fallback entity-focused'}")
    print(f"\nLocal search excels at:")
    print("  - Questions about specific entities (e.g., 'What is Tesla?')")
    print("  - Relationship queries (e.g., 'How does X relate to Y?')")
    print("  - Fact-based questions with clear entity references")


if __name__ == "__main__":
    main()
