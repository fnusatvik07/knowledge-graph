"""Step 2: Index the corpus using LightRAG.

Initializes LightRAG with OpenAI LLM and embedding models, inserts all
corpus documents, and saves the resulting knowledge graph index.
"""

import asyncio
import sys
from pathlib import Path

# Add projects dir to path for shared imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared.config import get_openai_api_key
from shared.document_loader import load_text_files

PROJECT_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_DIR / "data" / "corpus"
OUTPUT_DIR = PROJECT_DIR / "output" / "lightrag"


def check_lightrag_installed():
    """Verify that the lightrag package is available."""
    try:
        import lightrag  # noqa: F401
        return True
    except ImportError:
        print("=" * 60)
        print("ERROR: LightRAG is not installed.")
        print()
        print("Install it with:")
        print("  pip install lightrag-hku")
        print()
        print("For more information, see:")
        print("  https://github.com/HKUDS/LightRAG")
        print("=" * 60)
        return False


async def build_index():
    """Initialize LightRAG and insert all corpus documents."""
    from lightrag import LightRAG, QueryParam
    from lightrag.llm import openai_complete_if_cache, openai_embedding

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    api_key = get_openai_api_key()

    # Define LLM wrapper for LightRAG
    async def llm_model_func(
        prompt, system_prompt=None, history_messages=[], **kwargs
    ) -> str:
        return await openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            **kwargs,
        )

    # Define embedding wrapper for LightRAG
    async def embedding_func(texts: list[str]) -> list[list[float]]:
        return await openai_embedding(
            texts,
            model="text-embedding-3-small",
            api_key=api_key,
        )

    # Initialize LightRAG
    print("Initializing LightRAG...")
    rag = LightRAG(
        working_dir=str(OUTPUT_DIR),
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
    )

    # Load corpus documents
    documents = load_text_files(CORPUS_DIR)
    if not documents:
        print(f"No documents found in {CORPUS_DIR}")
        sys.exit(1)

    print(f"Loaded {len(documents)} documents from corpus\n")

    # Insert documents into LightRAG
    for i, doc in enumerate(documents, 1):
        print(f"[{i}/{len(documents)}] Inserting: {doc['title']}")
        print(f"  Length: {len(doc['content'])} characters")
        await rag.ainsert(doc["content"])
        print(f"  Done.")

    print(f"\nAll documents inserted successfully!")

    # Verify the index was created
    index_files = list(OUTPUT_DIR.iterdir())
    print(f"\nIndex files in {OUTPUT_DIR}:")
    for f in sorted(index_files):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name} ({size_kb:.1f} KB)")

    return rag


def test_query(rag):
    """Run a quick test query to verify the index works."""
    from lightrag import QueryParam

    print("\n" + "=" * 60)
    print("Test Query")
    print("=" * 60)

    test_question = "How do renewable energy and electric vehicles relate to climate change?"
    print(f"\nQuestion: {test_question}")

    # Try different search modes
    for mode in ["naive", "local", "global", "hybrid"]:
        try:
            print(f"\n--- {mode.upper()} search ---")
            result = asyncio.get_event_loop().run_until_complete(
                rag.aquery(test_question, param=QueryParam(mode=mode))
            )
            # Print first 300 chars of each result
            preview = result[:300] + "..." if len(result) > 300 else result
            print(preview)
        except Exception as e:
            print(f"  Error with {mode} search: {e}")


async def main_async():
    """Async main entry point."""
    print("=" * 60)
    print("LightRAG Indexing Pipeline")
    print("=" * 60)

    if not check_lightrag_installed():
        sys.exit(1)

    api_key = get_openai_api_key()
    print(f"API key loaded (ends with ...{api_key[-4:]})")
    print(f"\nCorpus directory: {CORPUS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}\n")

    rag = await build_index()

    # Run a quick test query
    print("\n" + "=" * 60)
    print("Test Query")
    print("=" * 60)

    from lightrag import QueryParam

    test_question = "How do renewable energy and electric vehicles relate to climate change?"
    print(f"\nQuestion: {test_question}")

    for mode in ["naive", "local", "global", "hybrid"]:
        try:
            print(f"\n--- {mode.upper()} search ---")
            result = await rag.aquery(
                test_question, param=QueryParam(mode=mode)
            )
            preview = result[:300] + "..." if len(result) > 300 else result
            print(preview)
        except Exception as e:
            print(f"  Error with {mode} search: {e}")

    print("\n" + "=" * 60)
    print("LightRAG indexing complete!")
    print(f"Index stored in: {OUTPUT_DIR}")
    print("=" * 60)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
