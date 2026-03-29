"""Step 5: Semantic search across your knowledge base.

Embed all notes and concepts, then search with natural language queries.
Enhances results with graph context (connected notes within 2 hops).
Uses LLM to synthesize answers from relevant notes.

Inputs:  output/notes_full.json, output/personal_kg.graphml
Outputs: Interactive search (prints results to console)
"""

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np

# Add projects dir to path for shared imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared.llm_clients import (
    chat_completion,
    get_embedding,
    get_embedding_batch,
)

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"


class KnowledgeBaseSearcher:
    """Semantic search engine for personal knowledge base."""

    def __init__(self, notes: list[dict], graph: nx.DiGraph, provider: str, model: str | None):
        self.notes = {note["id"]: note for note in notes}
        self.graph = graph
        self.provider = provider
        self.model = model
        self.note_ids: list[str] = []
        self.note_embeddings: np.ndarray | None = None

    def build_index(self):
        """Embed all notes to build the search index."""
        print("Building search index...")
        texts = []
        self.note_ids = []

        for note_id, note in self.notes.items():
            text = f"{note['title']}\n\n{note['content']}"
            texts.append(text[:2000])  # Truncate for embedding
            self.note_ids.append(note_id)

        embeddings = get_embedding_batch(texts, provider=self.provider, model=self.model)
        self.note_embeddings = np.array(embeddings)
        print(f"  Indexed {len(self.note_ids)} notes.")

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Search for notes matching a natural language query."""
        if self.note_embeddings is None:
            self.build_index()

        # Embed the query
        query_embedding = np.array(get_embedding(query, provider=self.provider, model=self.model))

        # Cosine similarity
        norms = np.linalg.norm(self.note_embeddings, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        similarities = (self.note_embeddings @ query_embedding) / (norms * query_norm + 1e-10)

        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            note_id = self.note_ids[idx]
            results.append({
                "note_id": note_id,
                "title": self.notes[note_id]["title"],
                "similarity": float(similarities[idx]),
                "content_preview": self.notes[note_id]["content"][:300],
            })

        return results

    def graph_enhanced_search(self, query: str, top_k: int = 3, hops: int = 2) -> dict:
        """Search with graph context: find related notes within N hops."""
        # Get initial search results
        direct_results = self.search(query, top_k=top_k)

        # Expand results with graph neighbors
        expanded_notes = set()
        for result in direct_results:
            node_id = f"note:{result['note_id']}"
            if node_id in self.graph:
                # BFS to find notes within N hops
                visited = set()
                frontier = {node_id}
                for _ in range(hops):
                    next_frontier = set()
                    for node in frontier:
                        if node not in visited:
                            visited.add(node)
                            neighbors = set(self.graph.successors(node)) | set(self.graph.predecessors(node))
                            next_frontier |= neighbors
                    frontier = next_frontier - visited

                # Collect connected notes
                for node in visited | frontier:
                    if self.graph.nodes[node].get("node_type") == "NOTE":
                        note_id = node.replace("note:", "")
                        if note_id in self.notes:
                            expanded_notes.add(note_id)

        # Build context from all relevant notes
        context_notes = []
        direct_ids = {r["note_id"] for r in direct_results}
        connected_ids = expanded_notes - direct_ids

        for result in direct_results:
            context_notes.append({
                "note_id": result["note_id"],
                "title": result["title"],
                "source": "direct_match",
                "similarity": result["similarity"],
            })

        for note_id in connected_ids:
            context_notes.append({
                "note_id": note_id,
                "title": self.notes[note_id]["title"],
                "source": "graph_connected",
                "similarity": 0.0,
            })

        return {
            "query": query,
            "direct_results": direct_results,
            "graph_expanded_notes": context_notes,
        }

    def synthesize_answer(self, query: str, top_k: int = 3) -> str:
        """Use LLM to synthesize an answer from relevant notes."""
        search_result = self.graph_enhanced_search(query, top_k=top_k)

        # Build context from relevant notes
        context_parts = []
        for note_info in search_result["graph_expanded_notes"]:
            note = self.notes[note_info["note_id"]]
            context_parts.append(f"--- {note['title']} ---\n{note['content'][:800]}")

        context = "\n\n".join(context_parts)

        system = """You are a helpful assistant answering questions about the user's
personal knowledge base. Use ONLY the provided note content to answer.
If the notes don't contain enough information, say so. Reference which
notes your answer comes from."""

        prompt = f"""Based on these notes from my knowledge base:

{context}

Question: {query}

Provide a comprehensive answer based on my notes."""

        answer = chat_completion(
            prompt=prompt,
            system=system,
            provider=self.provider,
            model=self.model,
        )

        return answer


def main():
    parser = argparse.ArgumentParser(description="Search your knowledge base")
    parser.add_argument("--query", "-q", type=str, help="Search query")
    parser.add_argument("--provider", default="openai", help="LLM/embedding provider")
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--top-k", type=int, default=3, help="Number of results")
    parser.add_argument("--synthesize", action="store_true", help="Use LLM to synthesize an answer")
    args = parser.parse_args()

    # Load data
    notes_path = OUTPUT_DIR / "notes_full.json"
    graph_path = OUTPUT_DIR / "personal_kg.graphml"

    if not notes_path.exists():
        print("Run 01_parse_notes.py first.")
        return
    if not graph_path.exists():
        print("Run 03_build_personal_kg.py first.")
        return

    notes = json.loads(notes_path.read_text(encoding="utf-8"))
    G = nx.read_graphml(str(graph_path))

    searcher = KnowledgeBaseSearcher(notes, G, args.provider, args.model)
    searcher.build_index()

    query = args.query
    if not query:
        # Interactive mode
        print("\nPersonal Knowledge Base Search")
        print("Type a query and press Enter. Type 'quit' to exit.\n")
        while True:
            query = input("Search: ").strip()
            if query.lower() in ("quit", "exit", "q"):
                break
            if not query:
                continue

            # Search
            results = searcher.graph_enhanced_search(query, top_k=args.top_k)

            print(f"\n  Direct matches:")
            for r in results["direct_results"]:
                print(f"    [{r['similarity']:.3f}] {r['title']}")
                print(f"           {r['content_preview'][:100]}...")

            connected = [n for n in results["graph_expanded_notes"] if n["source"] == "graph_connected"]
            if connected:
                print(f"\n  Connected notes (via graph):")
                for n in connected:
                    print(f"    -> {n['title']}")

            if args.synthesize:
                print(f"\n  Synthesized answer:")
                answer = searcher.synthesize_answer(query, top_k=args.top_k)
                print(f"    {answer}")

            print()
    else:
        # Single query mode
        print(f"\nSearching for: {query}\n")

        results = searcher.graph_enhanced_search(query, top_k=args.top_k)

        print("Direct matches:")
        for r in results["direct_results"]:
            print(f"  [{r['similarity']:.3f}] {r['title']}")
            print(f"         {r['content_preview'][:150]}...")
            print()

        connected = [n for n in results["graph_expanded_notes"] if n["source"] == "graph_connected"]
        if connected:
            print("Connected notes (via graph):")
            for n in connected:
                print(f"  -> {n['title']}")
            print()

        if args.synthesize:
            print("Synthesized answer:")
            answer = searcher.synthesize_answer(query, top_k=args.top_k)
            print(f"  {answer}")


if __name__ == "__main__":
    main()
