"""Step 2: Extract concepts and entities from notes using LLM.

Goes beyond explicit wiki-links to find implicit concepts mentioned in each
note. Identifies connections between notes that share concepts even when
they don't directly link to each other.

Inputs:  output/notes_full.json
Outputs: output/note_concepts.json
"""

import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel, Field

# Add projects dir to path for shared imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared.llm_clients import chat_completion_structured

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"

# ── Schema ───────────────────────────────────────────────────────────────

class Concept(BaseModel):
    """A concept or entity extracted from a note."""
    name: str = Field(description="Canonical name of the concept (lowercase, concise)")
    category: str = Field(description="Category: person, algorithm, technique, field, tool, theory, or other")
    relevance: str = Field(description="How this concept relates to the note: 'primary' (main topic) or 'mentioned' (referenced)")


class NoteConceptsResult(BaseModel):
    """All concepts extracted from a single note."""
    concepts: list[Concept] = Field(description="List of concepts found in the note")
    summary: str = Field(description="One-sentence summary of the note's main topic")


# ── Prompts ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a knowledge extraction assistant. Given a markdown note,
extract all meaningful concepts, entities, techniques, tools, theories, and people
mentioned in the note. Focus on concepts that could connect this note to other notes
in a personal knowledge base.

Be thorough but precise. Include both explicitly discussed topics and implicitly
referenced concepts. Use lowercase canonical names."""

USER_PROMPT = """Extract all concepts from this note:

Title: {title}
Content:
{content}

Return every concept mentioned, including implicit ones. For each, classify its
category and whether it's a primary topic or just mentioned."""


def extract_concepts_from_note(note: dict, provider: str, model: str | None) -> dict:
    """Extract concepts from a single note using the LLM."""
    prompt = USER_PROMPT.format(title=note["title"], content=note["content"])

    result = chat_completion_structured(
        prompt=prompt,
        output_schema=NoteConceptsResult,
        system=SYSTEM_PROMPT,
        provider=provider,
        model=model,
    )

    return {
        "note_id": note["id"],
        "title": note["title"],
        "summary": result.summary,
        "concepts": [c.model_dump() for c in result.concepts],
    }


def find_implicit_connections(note_concepts: list[dict]) -> list[dict]:
    """Find notes that share concepts but don't have explicit wiki-links."""
    # Build concept -> notes mapping
    concept_to_notes: dict[str, list[str]] = {}
    for nc in note_concepts:
        for concept in nc["concepts"]:
            name = concept["name"]
            if name not in concept_to_notes:
                concept_to_notes[name] = []
            concept_to_notes[name].append(nc["note_id"])

    # Find implicit connections (shared concepts)
    connections = []
    seen_pairs = set()
    for concept_name, note_ids in concept_to_notes.items():
        if len(note_ids) < 2:
            continue
        for i, note_a in enumerate(note_ids):
            for note_b in note_ids[i + 1:]:
                pair = tuple(sorted([note_a, note_b]))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    # Find all shared concepts for this pair
                    shared = [
                        cn for cn, nids in concept_to_notes.items()
                        if note_a in nids and note_b in nids
                    ]
                    connections.append({
                        "note_a": pair[0],
                        "note_b": pair[1],
                        "shared_concepts": shared,
                        "strength": len(shared),
                    })

    return sorted(connections, key=lambda x: x["strength"], reverse=True)


def main():
    parser = argparse.ArgumentParser(description="Extract concepts from notes using LLM")
    parser.add_argument("--provider", default="openai", help="LLM provider")
    parser.add_argument("--model", default=None, help="Model name")
    args = parser.parse_args()

    # Load parsed notes
    notes_path = OUTPUT_DIR / "notes_full.json"
    if not notes_path.exists():
        print("Run 01_parse_notes.py first to generate notes_full.json")
        return

    notes = json.loads(notes_path.read_text(encoding="utf-8"))
    print(f"Extracting concepts from {len(notes)} notes...\n")

    # Extract concepts from each note
    all_note_concepts = []
    for note in notes:
        print(f"  Processing: {note['title']}...")
        result = extract_concepts_from_note(note, args.provider, args.model)
        all_note_concepts.append(result)
        print(f"    Found {len(result['concepts'])} concepts")
        for c in result["concepts"]:
            marker = "*" if c["relevance"] == "primary" else " "
            print(f"      {marker} {c['name']} ({c['category']})")
        print()

    # Find implicit connections
    connections = find_implicit_connections(all_note_concepts)

    print("=" * 60)
    print("Implicit connections (notes sharing concepts):\n")
    for conn in connections:
        print(f"  {conn['note_a']} <-> {conn['note_b']}")
        print(f"    Shared concepts ({conn['strength']}): {', '.join(conn['shared_concepts'])}")
    print()

    # Collect unique concepts across all notes
    all_concepts = {}
    for nc in all_note_concepts:
        for c in nc["concepts"]:
            name = c["name"]
            if name not in all_concepts:
                all_concepts[name] = {
                    "name": name,
                    "category": c["category"],
                    "mentioned_in": [],
                }
            all_concepts[name]["mentioned_in"].append(nc["note_id"])

    print(f"Total unique concepts: {len(all_concepts)}")

    # Save results
    output = {
        "note_concepts": all_note_concepts,
        "implicit_connections": connections,
        "all_concepts": list(all_concepts.values()),
    }
    output_path = OUTPUT_DIR / "note_concepts.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved concepts to {output_path}")


if __name__ == "__main__":
    main()
