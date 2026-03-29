"""Step 4: Discover hidden connections between notes.

Analyzes the personal knowledge graph to find:
- Notes that share concepts but don't explicitly link to each other
- Concept clusters (groups of related ideas spanning multiple notes)
- Knowledge gaps (concepts mentioned but with no dedicated note)
- Semantically similar notes via embeddings

Inputs:  output/personal_kg.graphml, output/note_concepts.json, output/notes_full.json
Outputs: output/discoveries.json
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import networkx as nx
import numpy as np

# Add projects dir to path for shared imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared.llm_clients import get_embedding_batch

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"


def suggest_missing_links(G: nx.DiGraph) -> list[dict]:
    """Find notes that share concepts but don't have explicit wiki-links.

    These are candidates for adding [[wiki-links]] between notes.
    """
    note_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "NOTE"]
    suggestions = []

    for i, note_a in enumerate(note_nodes):
        for note_b in note_nodes[i + 1:]:
            # Check if they already link to each other
            has_direct_link = (
                G.has_edge(note_a, note_b)
                and G[note_a][note_b].get("edge_type") == "NOTE_LINKS_TO"
            ) or (
                G.has_edge(note_b, note_a)
                and G[note_b][note_a].get("edge_type") == "NOTE_LINKS_TO"
            )

            if has_direct_link:
                continue

            # Find shared concepts (notes -> concept neighbors)
            concepts_a = {
                n for n in G.successors(note_a)
                if G.nodes[n].get("node_type") == "CONCEPT"
            }
            concepts_b = {
                n for n in G.successors(note_b)
                if G.nodes[n].get("node_type") == "CONCEPT"
            }
            shared = concepts_a & concepts_b

            if shared:
                suggestions.append({
                    "note_a": note_a.replace("note:", ""),
                    "note_b": note_b.replace("note:", ""),
                    "shared_concepts": [c.replace("concept:", "") for c in shared],
                    "strength": len(shared),
                    "reason": f"Share {len(shared)} concepts but no direct link",
                })

    return sorted(suggestions, key=lambda x: x["strength"], reverse=True)


def find_concept_clusters(G: nx.DiGraph) -> list[dict]:
    """Find clusters of related concepts using community detection."""
    # Build a subgraph of concept nodes and their relationships
    concept_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "CONCEPT"]

    if len(concept_nodes) < 2:
        return []

    concept_subgraph = G.subgraph(concept_nodes).to_undirected()

    # Use greedy modularity if the graph is connected enough
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(concept_subgraph))
    except Exception:
        # Fallback: use connected components
        communities = list(nx.connected_components(concept_subgraph))

    clusters = []
    for i, community in enumerate(communities):
        concepts = [n.replace("concept:", "") for n in community]
        if len(concepts) >= 2:
            clusters.append({
                "cluster_id": i,
                "concepts": sorted(concepts),
                "size": len(concepts),
            })

    return sorted(clusters, key=lambda x: x["size"], reverse=True)


def find_knowledge_gaps(notes: list[dict], concepts_data: dict) -> list[dict]:
    """Find concepts that are mentioned but have no dedicated note.

    These represent potential notes you could write to fill gaps in your
    knowledge base.
    """
    note_ids = {note["id"] for note in notes}
    all_concepts = concepts_data.get("all_concepts", [])

    gaps = []
    for concept in all_concepts:
        name = concept["name"]
        # Check if there's a note with this name (or similar)
        has_note = any(
            name.replace(" ", "_") == nid or name == nid
            for nid in note_ids
        )
        if not has_note and len(concept.get("mentioned_in", [])) >= 2:
            gaps.append({
                "concept": name,
                "category": concept.get("category", "other"),
                "mentioned_in": concept["mentioned_in"],
                "mention_count": len(concept["mentioned_in"]),
                "suggestion": f"Consider creating a note about '{name}' — mentioned in {len(concept['mentioned_in'])} notes",
            })

    return sorted(gaps, key=lambda x: x["mention_count"], reverse=True)


def find_similar_notes_by_embedding(
    notes: list[dict],
    provider: str = "openai",
    model: str | None = None,
    top_k: int = 5,
) -> list[dict]:
    """Find semantically similar notes using embeddings."""
    # Prepare texts for embedding
    texts = []
    note_ids = []
    for note in notes:
        texts.append(f"{note['title']}\n\n{note['content'][:1000]}")
        note_ids.append(note["id"])

    print("  Computing embeddings for all notes...")
    embeddings = get_embedding_batch(texts, provider=provider, model=model)
    embeddings_np = np.array(embeddings)

    # Compute cosine similarity matrix
    norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = embeddings_np / norms
    similarity_matrix = normalized @ normalized.T

    # Find top similar pairs
    pairs = []
    for i in range(len(note_ids)):
        for j in range(i + 1, len(note_ids)):
            pairs.append({
                "note_a": note_ids[i],
                "note_b": note_ids[j],
                "similarity": float(similarity_matrix[i, j]),
            })

    pairs.sort(key=lambda x: x["similarity"], reverse=True)
    return pairs[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Discover hidden connections")
    parser.add_argument("--provider", default="openai", help="Embedding provider")
    parser.add_argument("--model", default=None, help="Embedding model")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding-based similarity")
    args = parser.parse_args()

    # Load data
    graph_path = OUTPUT_DIR / "personal_kg.graphml"
    if not graph_path.exists():
        print("Run 03_build_personal_kg.py first.")
        return
    G = nx.read_graphml(str(graph_path))

    notes = json.loads((OUTPUT_DIR / "notes_full.json").read_text(encoding="utf-8"))

    concepts_data = None
    concepts_path = OUTPUT_DIR / "note_concepts.json"
    if concepts_path.exists():
        concepts_data = json.loads(concepts_path.read_text(encoding="utf-8"))

    discoveries = {}

    # 1. Suggest missing links
    print("Finding suggested links...")
    suggestions = suggest_missing_links(G)
    discoveries["suggested_links"] = suggestions

    print(f"\n  Found {len(suggestions)} potential missing links:")
    for s in suggestions[:10]:
        print(f"    {s['note_a']} <-> {s['note_b']}  (shared: {', '.join(s['shared_concepts'][:3])})")
    print()

    # 2. Concept clusters
    print("Finding concept clusters...")
    clusters = find_concept_clusters(G)
    discoveries["concept_clusters"] = clusters

    for cluster in clusters[:5]:
        print(f"  Cluster {cluster['cluster_id']}: {', '.join(cluster['concepts'][:5])}")
    print()

    # 3. Knowledge gaps
    if concepts_data:
        print("Finding knowledge gaps...")
        gaps = find_knowledge_gaps(notes, concepts_data)
        discoveries["knowledge_gaps"] = gaps

        print(f"\n  Found {len(gaps)} knowledge gaps:")
        for gap in gaps[:10]:
            print(f"    {gap['suggestion']}")
        print()

    # 4. Semantic similarity
    if not args.skip_embeddings:
        print("Computing semantic similarity...")
        similar_pairs = find_similar_notes_by_embedding(
            notes, provider=args.provider, model=args.model
        )
        discoveries["similar_notes"] = similar_pairs

        print("\n  Most similar note pairs:")
        for pair in similar_pairs:
            print(f"    {pair['note_a']} <-> {pair['note_b']}  (similarity: {pair['similarity']:.3f})")
    else:
        print("Skipping embedding-based similarity (use --skip-embeddings=false to enable)")

    # Save discoveries
    output_path = OUTPUT_DIR / "discoveries.json"
    output_path.write_text(json.dumps(discoveries, indent=2), encoding="utf-8")
    print(f"\nSaved discoveries to {output_path}")


if __name__ == "__main__":
    main()
