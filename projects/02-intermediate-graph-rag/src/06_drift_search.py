"""Step 6: DRIFT search - combining local and global search strategies.

DRIFT (Dynamic Reasoning and Inference with Flexible Traversal) search
combines local entity-focused retrieval with global community-level
understanding. This script demonstrates a simplified DRIFT-like approach:

1. Start with the query and identify relevant entities (local)
2. Expand to their communities for broader context (global)
3. Iteratively refine the search based on intermediate findings
4. Synthesize a comprehensive answer from both levels
"""
# LangChain docs: https://python.langchain.com/docs/integrations/chat/

import argparse
import json
import sys
from pathlib import Path

# Add projects dir to path for shared imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared.llm_clients import chat_completion
from shared.document_loader import load_text_files

PROJECT_DIR = Path(__file__).resolve().parent.parent
GRAPHRAG_DIR = PROJECT_DIR / "output" / "graphrag"
CORPUS_DIR = PROJECT_DIR / "data" / "corpus"

# Questions that benefit from DRIFT search (require both local detail
# and global context)
DRIFT_QUESTIONS = [
    "How do battery technologies connect the renewable energy and electric vehicle sectors, and what policy incentives accelerate their development?",
    "Trace the path of carbon dioxide from emission sources through capture technologies to permanent storage or agricultural sequestration.",
    "What is the relationship between the Inflation Reduction Act, green hydrogen, and the broader clean energy transition?",
]


def load_corpus_documents() -> list[dict]:
    """Load all corpus documents."""
    return load_text_files(CORPUS_DIR)


def extract_query_entities(question: str, provider: str = "openai") -> list[str]:
    """Extract key entities from the query using LLM."""
    response = chat_completion(
        prompt=f"""Extract the key entities from this question. Return as a JSON list of strings.
Include specific names, technologies, organizations, concepts, and policies mentioned.

Question: {question}

Return ONLY a JSON list, e.g., ["entity1", "entity2"]""",
        provider=provider,
        response_format={"type": "json_object"},
    )
    try:
        data = json.loads(response)
        # Handle various JSON structures
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("entities", "items", "list", "result"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            # Return all values if they are lists
            for v in data.values():
                if isinstance(v, list):
                    return v
        return []
    except json.JSONDecodeError:
        return []


def local_entity_search(entities: list[str], documents: list[dict]) -> dict:
    """Find passages related to specific entities (local search phase)."""
    entity_passages = {}

    for entity in entities:
        passages = []
        entity_lower = entity.lower()
        for doc in documents:
            sentences = doc["content"].replace("\n", " ").split(". ")
            for sentence in sentences:
                if entity_lower in sentence.lower():
                    passages.append({
                        "text": sentence.strip() + ".",
                        "source": doc["title"],
                    })
        entity_passages[entity] = passages[:5]  # Top 5 passages per entity

    return entity_passages


def global_community_context(question: str, documents: list[dict], provider: str = "openai") -> str:
    """Generate community-level context (global search phase).

    Summarizes each document's relevance to the question, then
    synthesizes cross-document themes.
    """
    doc_relevance = []

    for doc in documents:
        relevance = chat_completion(
            prompt=f"""Rate how relevant this document is to the question on a scale
of 0-10, and extract the most relevant themes. Respond in JSON format:
{{"relevance": <0-10>, "themes": ["theme1", "theme2"], "key_facts": ["fact1", "fact2"]}}

Question: {question}

Document ({doc['title']}):
{doc['content'][:1500]}""",
            provider=provider,
            response_format={"type": "json_object"},
        )
        try:
            data = json.loads(relevance)
            data["source"] = doc["title"]
            doc_relevance.append(data)
        except json.JSONDecodeError:
            pass

    # Sort by relevance and create community context
    doc_relevance.sort(key=lambda x: x.get("relevance", 0), reverse=True)

    context_parts = []
    for dr in doc_relevance:
        if dr.get("relevance", 0) >= 3:
            themes = ", ".join(dr.get("themes", []))
            facts = "; ".join(dr.get("key_facts", []))
            context_parts.append(
                f"[{dr['source']} (relevance: {dr.get('relevance', '?')}/10)]\n"
                f"  Themes: {themes}\n"
                f"  Key facts: {facts}"
            )

    return "\n\n".join(context_parts)


def drift_search(question: str, documents: list[dict], provider: str = "openai") -> str:
    """Execute DRIFT search: iterative local-to-global retrieval.

    Phase 1: LOCAL - Extract entities and find specific passages
    Phase 2: GLOBAL - Get community-level thematic context
    Phase 3: EXPAND - Identify follow-up entities from Phase 1 & 2
    Phase 4: SYNTHESIZE - Combine all levels into final answer
    """
    print("\n  Phase 1: LOCAL - Entity extraction and passage retrieval")
    entities = extract_query_entities(question, provider=provider)
    print(f"    Extracted entities: {entities}")

    entity_passages = local_entity_search(entities, documents)
    local_context = ""
    for entity, passages in entity_passages.items():
        if passages:
            local_context += f"\n[Entity: {entity}]\n"
            for p in passages:
                local_context += f"  - {p['text']} (from {p['source']})\n"
            print(f"    {entity}: {len(passages)} passages found")
        else:
            print(f"    {entity}: no direct passages found")

    print("\n  Phase 2: GLOBAL - Community-level thematic analysis")
    global_context = global_community_context(question, documents, provider=provider)
    print(f"    Analyzed {len(documents)} document communities")

    print("\n  Phase 3: EXPAND - Identifying connected entities")
    expansion_prompt = f"""Based on the following entity-level findings and community-level
themes, identify 3-5 additional related entities or concepts that should be explored
to give a more complete answer. Return as a JSON list.

Question: {question}

Entity findings:
{local_context[:1500]}

Community themes:
{global_context[:1500]}

Return ONLY a JSON list of additional entities to explore."""

    expansion_response = chat_completion(
        expansion_prompt,
        provider=provider,
        response_format={"type": "json_object"},
    )

    try:
        expansion_data = json.loads(expansion_response)
        if isinstance(expansion_data, dict):
            for v in expansion_data.values():
                if isinstance(v, list):
                    expanded_entities = v
                    break
            else:
                expanded_entities = []
        else:
            expanded_entities = expansion_data if isinstance(expansion_data, list) else []
    except json.JSONDecodeError:
        expanded_entities = []

    print(f"    Expanded entities: {expanded_entities}")

    # Search for expanded entities
    expanded_passages = local_entity_search(expanded_entities, documents)
    expanded_context = ""
    for entity, passages in expanded_passages.items():
        if passages:
            expanded_context += f"\n[Expanded entity: {entity}]\n"
            for p in passages[:3]:
                expanded_context += f"  - {p['text']} (from {p['source']})\n"

    print("\n  Phase 4: SYNTHESIZE - Combining local + global + expanded context")
    synthesis_prompt = f"""You are answering a complex question using a DRIFT search strategy
that combines entity-level detail with community-level thematic understanding.

Question: {question}

=== LOCAL CONTEXT (entity-specific passages) ===
{local_context[:2000]}

=== GLOBAL CONTEXT (community-level themes) ===
{global_context[:2000]}

=== EXPANDED CONTEXT (related entities discovered during search) ===
{expanded_context[:1500]}

Provide a comprehensive answer that:
1. Addresses specific entities and their roles (from local search)
2. Shows how themes connect across different topic areas (from global search)
3. Incorporates insights from the expanded search
4. Highlights cross-cutting relationships and dependencies

Answer:"""

    answer = chat_completion(
        synthesis_prompt,
        system="You are an expert analyst using a knowledge graph with both local "
               "entity-level detail and global community-level understanding. "
               "Provide rich, interconnected answers that demonstrate deep "
               "cross-domain understanding.",
        provider=provider,
    )

    return answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DRIFT search")
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
    print("DRIFT Search: Dynamic Local-Global Hybrid Retrieval")
    print("=" * 60)
    print()
    print(f"Using LLM provider: {provider}")
    print("DRIFT search combines local entity-focused retrieval with global")
    print("community-level understanding, then iteratively expands the search")
    print("to discover connected concepts across the knowledge graph.")
    print()

    documents = load_corpus_documents()
    print(f"Loaded {len(documents)} corpus documents\n")

    results = {}

    for i, question in enumerate(DRIFT_QUESTIONS, 1):
        print(f"\n{'='*60}")
        print(f"Question {i}: {question}")
        print("-" * 60)

        answer = drift_search(question, documents, provider=provider)
        print(f"\nFinal Answer:\n{answer}")
        results[question] = answer

    # Summary
    print(f"\n\n{'='*60}")
    print("DRIFT Search Summary")
    print("=" * 60)
    print(f"Questions answered: {len(results)}")
    print(f"\nDRIFT search advantages:")
    print("  - Combines entity precision with thematic breadth")
    print("  - Discovers non-obvious connections through expansion")
    print("  - Handles complex, multi-faceted questions")
    print("  - Provides answers grounded in both specific facts and broad themes")
    print(f"\nDRIFT search phases:")
    print("  1. LOCAL:      Extract entities -> find specific passages")
    print("  2. GLOBAL:     Analyze document communities for themes")
    print("  3. EXPAND:     Discover related entities from initial findings")
    print("  4. SYNTHESIZE: Combine all levels into comprehensive answer")


if __name__ == "__main__":
    main()
