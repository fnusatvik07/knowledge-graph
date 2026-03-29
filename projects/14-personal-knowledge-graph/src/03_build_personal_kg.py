"""Step 3: Build the personal knowledge graph with NetworkX.

Combines parsed note metadata and extracted concepts into a unified
knowledge graph with note nodes, concept nodes, and typed edges.

Node types:
  - NOTE: A markdown note (with content, tags, word count)
  - CONCEPT: An extracted entity/concept

Edge types:
  - NOTE_LINKS_TO: Explicit [[wiki-link]] between notes
  - MENTIONS_CONCEPT: A note mentions a concept
  - CONCEPT_RELATED_TO: Two concepts co-occur in the same note

Inputs:  output/notes_full.json, output/note_concepts.json
Outputs: output/personal_kg.graphml, output/kg_stats.json
"""

import json
import sys
from pathlib import Path
from collections import Counter

import networkx as nx

# Add projects dir to path for shared imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"


def build_graph(notes: list[dict], concepts_data: dict) -> nx.DiGraph:
    """Build the personal knowledge graph."""
    G = nx.DiGraph()

    # ── Add note nodes ───────────────────────────────────────────────
    for note in notes:
        G.add_node(
            f"note:{note['id']}",
            node_type="NOTE",
            title=note["title"],
            tags=", ".join(note.get("tags", [])),
            word_count=note.get("word_count", 0),
            label=note["title"],
        )

    # ── Add explicit wiki-link edges ─────────────────────────────────
    for note in notes:
        source = f"note:{note['id']}"
        for link_info in note.get("resolved_links", []):
            if link_info["resolved"]:
                target = f"note:{link_info['target']}"
                if target in G:
                    G.add_edge(source, target, edge_type="NOTE_LINKS_TO", weight=1.0)

    # ── Add concept nodes and edges ──────────────────────────────────
    if concepts_data:
        all_concepts = concepts_data.get("all_concepts", [])
        note_concepts = concepts_data.get("note_concepts", [])

        # Add concept nodes
        for concept in all_concepts:
            node_id = f"concept:{concept['name']}"
            G.add_node(
                node_id,
                node_type="CONCEPT",
                category=concept.get("category", "other"),
                mentioned_count=len(concept.get("mentioned_in", [])),
                label=concept["name"],
            )

        # Add MENTIONS_CONCEPT edges
        for nc in note_concepts:
            note_node = f"note:{nc['note_id']}"
            for concept in nc["concepts"]:
                concept_node = f"concept:{concept['name']}"
                if concept_node in G:
                    G.add_edge(
                        note_node,
                        concept_node,
                        edge_type="MENTIONS_CONCEPT",
                        relevance=concept.get("relevance", "mentioned"),
                        weight=1.0 if concept.get("relevance") == "primary" else 0.5,
                    )

        # Add CONCEPT_RELATED_TO edges (concepts co-occurring in same note)
        for nc in note_concepts:
            concept_names = [c["name"] for c in nc["concepts"]]
            for i, c1 in enumerate(concept_names):
                for c2 in concept_names[i + 1:]:
                    node1 = f"concept:{c1}"
                    node2 = f"concept:{c2}"
                    if node1 in G and node2 in G:
                        if not G.has_edge(node1, node2):
                            G.add_edge(
                                node1, node2,
                                edge_type="CONCEPT_RELATED_TO",
                                weight=0.3,
                            )

    return G


def compute_stats(G: nx.DiGraph) -> dict:
    """Compute graph statistics."""
    note_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "NOTE"]
    concept_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "CONCEPT"]

    edge_types = Counter(d.get("edge_type", "unknown") for _, _, d in G.edges(data=True))

    # Degree analysis for notes
    note_degrees = {n: G.degree(n) for n in note_nodes}
    sorted_notes = sorted(note_degrees.items(), key=lambda x: x[1], reverse=True)

    # Find orphan notes (no connections)
    orphan_notes = [n for n in note_nodes if G.degree(n) == 0]

    # Most connected concepts
    concept_degrees = {n: G.degree(n) for n in concept_nodes}
    sorted_concepts = sorted(concept_degrees.items(), key=lambda x: x[1], reverse=True)

    stats = {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "note_count": len(note_nodes),
        "concept_count": len(concept_nodes),
        "edge_type_counts": dict(edge_types),
        "orphan_notes": [n.replace("note:", "") for n in orphan_notes],
        "most_connected_notes": [
            {"note": n.replace("note:", ""), "connections": d}
            for n, d in sorted_notes[:10]
        ],
        "most_connected_concepts": [
            {"concept": n.replace("concept:", ""), "connections": d}
            for n, d in sorted_concepts[:10]
        ],
    }

    return stats


def main():
    # Load notes
    notes_path = OUTPUT_DIR / "notes_full.json"
    if not notes_path.exists():
        print("Run 01_parse_notes.py first.")
        return
    notes = json.loads(notes_path.read_text(encoding="utf-8"))

    # Load concepts (optional — graph works without them)
    concepts_path = OUTPUT_DIR / "note_concepts.json"
    concepts_data = None
    if concepts_path.exists():
        concepts_data = json.loads(concepts_path.read_text(encoding="utf-8"))
        print("Loaded extracted concepts.")
    else:
        print("No concepts file found. Building graph with notes and wiki-links only.")
        print("Run 02_extract_concepts.py for a richer graph.\n")

    # Build graph
    G = build_graph(notes, concepts_data)

    # Compute and display stats
    stats = compute_stats(G)

    print("=" * 60)
    print("Personal Knowledge Graph Stats")
    print("=" * 60)
    print(f"  Nodes:    {stats['total_nodes']} ({stats['note_count']} notes, {stats['concept_count']} concepts)")
    print(f"  Edges:    {stats['total_edges']}")
    print()
    print("  Edge types:")
    for etype, count in stats["edge_type_counts"].items():
        print(f"    {etype}: {count}")
    print()

    if stats["orphan_notes"]:
        print(f"  Orphan notes (no connections): {stats['orphan_notes']}")
    else:
        print("  No orphan notes — all notes are connected!")
    print()

    print("  Most connected notes:")
    for item in stats["most_connected_notes"][:5]:
        print(f"    {item['note']:30s}  {item['connections']} connections")
    print()

    if stats["most_connected_concepts"]:
        print("  Most connected concepts:")
        for item in stats["most_connected_concepts"][:5]:
            print(f"    {item['concept']:30s}  {item['connections']} connections")

    # Save graph
    graph_path = OUTPUT_DIR / "personal_kg.graphml"
    nx.write_graphml(G, str(graph_path))
    print(f"\nSaved graph to {graph_path}")

    stats_path = OUTPUT_DIR / "kg_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Saved stats to {stats_path}")


if __name__ == "__main__":
    main()
