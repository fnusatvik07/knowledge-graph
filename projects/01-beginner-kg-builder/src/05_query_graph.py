"""Step 5: Query the knowledge graph using natural language.

Loads the graph, finds relevant entities and their neighborhoods,
and uses an LLM to generate answers grounded in graph data.
"""
# LangChain docs: https://python.langchain.com/docs/integrations/chat/

import argparse
import json
import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared.llm_clients import chat_completion, get_embedding
from prompts import QUERY_SYSTEM, QUERY_USER

OUTPUT_DIR = Path(__file__).parent.parent / "output"


def find_relevant_entities(G: nx.DiGraph, question: str, top_k: int = 5) -> list[str]:
    """Find entities most relevant to the question using keyword matching.

    For a production system, you'd use embedding similarity. Here we use
    simple keyword overlap for clarity.
    """
    question_words = set(question.lower().split())
    scores = {}

    for node, data in G.nodes(data=True):
        node_words = set(node.lower().split())
        desc_words = set(data.get("description", "").lower().split())
        all_words = node_words | desc_words

        # Score = number of overlapping words
        overlap = len(question_words & all_words)
        if overlap > 0:
            scores[node] = overlap

    # Sort by score and return top_k
    sorted_entities = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in sorted_entities[:top_k]]


def get_subgraph_context(G: nx.DiGraph, seed_entities: list[str], hops: int = 2) -> str:
    """Extract a text description of the subgraph around seed entities."""
    context_parts = []

    visited = set()
    for entity in seed_entities:
        if entity not in G:
            continue

        # Entity info
        data = G.nodes[entity]
        context_parts.append(f"Entity: {entity} (Type: {data.get('type', 'UNKNOWN')})")
        context_parts.append(f"  Description: {data.get('description', 'N/A')}")

        # Outgoing relationships
        for _, target, edge_data in G.out_edges(entity, data=True):
            rel = edge_data.get("relation_type", "RELATED_TO")
            desc = edge_data.get("description", "")
            context_parts.append(f"  → {entity} --[{rel}]--> {target}: {desc}")
            visited.add(target)

        # Incoming relationships
        for source, _, edge_data in G.in_edges(entity, data=True):
            rel = edge_data.get("relation_type", "RELATED_TO")
            desc = edge_data.get("description", "")
            context_parts.append(f"  ← {source} --[{rel}]--> {entity}: {desc}")
            visited.add(source)

    # Add 1-hop neighbors' descriptions for additional context
    if hops >= 2:
        for neighbor in list(visited)[:10]:  # Limit to avoid huge context
            if neighbor not in seed_entities and neighbor in G:
                data = G.nodes[neighbor]
                context_parts.append(f"\nNearby entity: {neighbor} (Type: {data.get('type', 'UNKNOWN')})")
                context_parts.append(f"  Description: {data.get('description', 'N/A')}")

    return "\n".join(context_parts)


def answer_question(G: nx.DiGraph, question: str, provider: str = "openai") -> str:
    """Answer a question using the knowledge graph."""
    # Find relevant entities
    relevant = find_relevant_entities(G, question)
    if not relevant:
        return "I couldn't find any relevant entities in the knowledge graph for this question."

    # Get subgraph context
    context = get_subgraph_context(G, relevant)

    # Generate answer using LLM
    answer = chat_completion(
        prompt=QUERY_USER.format(question=question, context=context),
        system=QUERY_SYSTEM,
        provider=provider,
    )
    return answer


SAMPLE_QUESTIONS = [
    "Who founded OpenAI and what products did they develop?",
    "What is the relationship between Graph RAG and traditional RAG?",
    "Which people are considered the godfathers of deep learning?",
    "What graph database is most popular and what query language does it use?",
    "How are LangChain and LlamaIndex related to knowledge graphs?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the knowledge graph")
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
    print(f"Using LLM provider: {provider}")

    graphml_path = OUTPUT_DIR / "knowledge_graph.graphml"
    if not graphml_path.exists():
        print("Error: Run 03_build_graph.py first!")
        return

    G = nx.read_graphml(str(graphml_path))
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"{'='*60}\n")

    for question in SAMPLE_QUESTIONS:
        print(f"Q: {question}")
        answer = answer_question(G, question, provider=provider)
        print(f"A: {answer}")
        print(f"\n{'-'*60}\n")

    # Interactive mode
    print("\n\nEnter your own questions (type 'quit' to exit):\n")
    while True:
        question = input("Q: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
        answer = answer_question(G, question, provider=provider)
        print(f"A: {answer}\n")


if __name__ == "__main__":
    main()
