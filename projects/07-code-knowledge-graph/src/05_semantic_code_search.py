"""Semantic search over code entities using LLM embeddings.

Embeds descriptions of all functions, classes, and modules in the code KG,
then finds nearest matches for natural language queries like
"Find functions related to error handling" or "Where is logging done?".

Usage:
    python src/05_semantic_code_search.py --query "error handling"
    python src/05_semantic_code_search.py --query "mathematical operations"
    python src/05_semantic_code_search.py --query "data transformation" --top_k 5

References:
    - LangChain embeddings: https://python.langchain.com/docs/integrations/text_embedding/
    - LangChain vector similarity: https://python.langchain.com/docs/how_to/vectorstores/
    - Cosine similarity: https://en.wikipedia.org/wiki/Cosine_similarity
"""

import argparse
import json
import sys
from pathlib import Path

# ── Shared imports ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared.llm_clients import get_embedding, get_embedding_batch

import networkx as nx
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"


def build_entity_descriptions(G: nx.DiGraph) -> list[dict]:
    """Build natural language descriptions for each code entity.

    Creates a searchable description from the entity's metadata:
    name, type, docstring, parameters, module, relationships.

    Args:
        G: The code knowledge graph.

    Returns:
        List of dicts with 'node_id', 'description', and entity metadata.
    """
    entities = []

    for node_id, data in G.nodes(data=True):
        entity_type = data.get("entity_type", "Unknown")
        name = data.get("name", "")
        module = data.get("module", "")
        docstring = data.get("docstring", "") or ""

        # Skip external/placeholder nodes with no useful info
        if module == "external" and not docstring:
            continue

        # Build a rich description
        parts = []

        if entity_type == "Function":
            params = data.get("params", [])
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except (json.JSONDecodeError, TypeError):
                    params = [params]
            is_method = data.get("is_method", False)
            kind = "method" if is_method else "function"
            parts.append(f"Python {kind} named '{name}'")
            if module:
                parts.append(f"in module '{module}'")
            if params:
                param_str = ", ".join(str(p) for p in params[:6])
                parts.append(f"with parameters: {param_str}")
            if docstring:
                parts.append(f"Description: {docstring[:200]}")

            # What does it call?
            callees = [
                G.nodes[target].get("name", target)
                for _, target, edata in G.out_edges(node_id, data=True)
                if edata.get("relationship") == "CALLS"
            ]
            if callees:
                parts.append(f"Calls: {', '.join(callees[:5])}")

        elif entity_type == "Class":
            parts.append(f"Python class named '{name}'")
            if module:
                parts.append(f"in module '{module}'")
            if docstring:
                parts.append(f"Description: {docstring[:200]}")

            # List methods
            methods = [
                G.nodes[target].get("name", target)
                for _, target, edata in G.out_edges(node_id, data=True)
                if edata.get("relationship") == "DEFINES"
                and G.nodes[target].get("entity_type") == "Function"
            ]
            if methods:
                parts.append(f"Methods: {', '.join(methods)}")

            # Inheritance
            bases = [
                G.nodes[target].get("name", target)
                for _, target, edata in G.out_edges(node_id, data=True)
                if edata.get("relationship") == "INHERITS_FROM"
            ]
            if bases:
                parts.append(f"Inherits from: {', '.join(bases)}")

        elif entity_type == "Module":
            language = data.get("language", "")
            parts.append(f"{language} module named '{name}'")
            if docstring:
                parts.append(f"Description: {docstring[:200]}")

            # What does it contain?
            contents = [
                f"{G.nodes[target].get('entity_type', '?')} {G.nodes[target].get('name', target)}"
                for _, target, edata in G.out_edges(node_id, data=True)
                if edata.get("relationship") == "CONTAINS"
            ]
            if contents:
                parts.append(f"Contains: {', '.join(contents[:8])}")

        else:
            parts.append(f"{entity_type} named '{name}'")

        description = ". ".join(parts)

        entities.append({
            "node_id": node_id,
            "name": name,
            "entity_type": entity_type,
            "module": module,
            "description": description,
            "line_start": data.get("line_start", 0),
            "line_end": data.get("line_end", 0),
        })

    return entities


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


def semantic_search(
    query: str,
    entities: list[dict],
    embeddings: list[list[float]],
    top_k: int = 5,
    provider: str = "openai",
    model: str | None = None,
) -> list[dict]:
    """Find code entities most semantically similar to the query.

    Args:
        query: Natural language search query.
        entities: List of entity dicts with descriptions.
        embeddings: Pre-computed embeddings for each entity's description.
        top_k: Number of results to return.
        provider: Embedding provider.
        model: Embedding model name.

    Returns:
        Top-k matching entities with similarity scores.
    """
    # Embed the query
    query_embedding = get_embedding(query, provider=provider, model=model)

    # Compute similarities
    scores = []
    for i, entity_embedding in enumerate(embeddings):
        sim = cosine_similarity(query_embedding, entity_embedding)
        scores.append((i, sim))

    # Sort by similarity (descending)
    scores.sort(key=lambda x: -x[1])

    # Return top-k results
    results = []
    for idx, sim in scores[:top_k]:
        entity = entities[idx].copy()
        entity["similarity_score"] = round(sim, 4)
        results.append(entity)

    return results


def main():
    parser = argparse.ArgumentParser(description="Semantic code search using LLM embeddings.")
    parser.add_argument("--query", type=str, required=True,
                        help="Natural language search query")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of results to return")
    parser.add_argument("--provider", type=str, default="openai",
                        choices=["openai", "ollama"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--input", type=str, default=None,
                        help="Path to code_kg.json")
    args = parser.parse_args()

    # Load the graph
    input_file = Path(args.input) if args.input else OUTPUT_DIR / "code_kg.json"
    if not input_file.exists():
        print(f"ERROR: Graph file not found at {input_file}")
        print("Run 03_build_code_kg.py first.")
        sys.exit(1)

    print(f"Loading code knowledge graph from {input_file}...")
    with open(input_file) as f:
        json_data = json.load(f)
    G = nx.node_link_graph(json_data)

    # Build entity descriptions
    print("Building entity descriptions...")
    entities = build_entity_descriptions(G)
    print(f"Found {len(entities)} searchable entities")

    # Check for cached embeddings
    cache_file = OUTPUT_DIR / "entity_embeddings.json"
    if cache_file.exists():
        print(f"Loading cached embeddings from {cache_file}...")
        with open(cache_file) as f:
            cache = json.load(f)
        # Validate cache matches current entities
        if len(cache.get("embeddings", [])) == len(entities):
            embeddings = cache["embeddings"]
        else:
            print("Cache mismatch, recomputing embeddings...")
            embeddings = None
    else:
        embeddings = None

    if embeddings is None:
        # Compute embeddings for all entity descriptions
        descriptions = [e["description"] for e in entities]
        print(f"Computing embeddings for {len(descriptions)} entities...")
        embeddings = get_embedding_batch(descriptions, provider=args.provider, model=args.model)

        # Cache embeddings
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump({"embeddings": embeddings}, f)
        print(f"Cached embeddings to {cache_file}")

    # Perform semantic search
    print(f"\nSearching for: \"{args.query}\"")
    print(f"{'='*60}")

    results = semantic_search(
        query=args.query,
        entities=entities,
        embeddings=embeddings,
        top_k=args.top_k,
        provider=args.provider,
        model=args.model,
    )

    for i, result in enumerate(results, 1):
        print(f"\n[{i}] {result['name']} ({result['entity_type']})")
        print(f"    Module: {result['module']}")
        print(f"    Similarity: {result['similarity_score']:.4f}")
        if result.get("line_start"):
            print(f"    Lines: {result['line_start']}-{result.get('line_end', '?')}")
        print(f"    Description: {result['description'][:120]}...")

    # Save search results
    search_output = OUTPUT_DIR / "search_results.json"
    with open(search_output, "w") as f:
        json.dump({
            "query": args.query,
            "top_k": args.top_k,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {search_output}")


if __name__ == "__main__":
    main()
