"""Step 1: Parse markdown notes and extract metadata.

Reads all markdown files from data/notes/, extracts titles, content,
[[wiki-links]], #tags, and headings. Builds a note metadata index and
resolves wiki-links to actual note files.

Outputs: output/note_index.json
"""

import json
import re
import sys
from pathlib import Path

# Add projects dir to path for shared imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
NOTES_DIR = PROJECT_DIR / "data" / "notes"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def extract_title(content: str, filename: str) -> str:
    """Extract the title from markdown content (first # heading) or filename."""
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return filename.replace("_", " ").replace(".md", "").title()


def extract_wiki_links(content: str) -> list[str]:
    """Extract all [[wiki-style links]] from content."""
    return re.findall(r"\[\[(\w[\w_-]*)\]\]", content)


def extract_tags(content: str) -> list[str]:
    """Extract all #tags from content."""
    return re.findall(r"(?:^|\s)#([\w-]+)", content)


def extract_headings(content: str) -> list[dict]:
    """Extract all headings with their level."""
    headings = []
    for match in re.finditer(r"^(#{1,6})\s+(.+)$", content, re.MULTILINE):
        headings.append({
            "level": len(match.group(1)),
            "text": match.group(2).strip(),
        })
    return headings


def resolve_links(wiki_links: list[str], available_notes: dict[str, str]) -> list[dict]:
    """Resolve wiki-links to actual note files.

    Returns a list of dicts with link target and whether it was resolved.
    """
    resolved = []
    for link in wiki_links:
        normalized = link.lower().replace("-", "_")
        if normalized in available_notes:
            resolved.append({
                "target": normalized,
                "resolved": True,
                "target_file": available_notes[normalized],
            })
        else:
            resolved.append({
                "target": normalized,
                "resolved": False,
                "target_file": None,
            })
    return resolved


def parse_note(filepath: Path, available_notes: dict[str, str]) -> dict:
    """Parse a single markdown note file into structured metadata."""
    content = filepath.read_text(encoding="utf-8")
    stem = filepath.stem.lower()

    wiki_links = extract_wiki_links(content)
    resolved = resolve_links(wiki_links, available_notes)

    return {
        "id": stem,
        "filename": filepath.name,
        "title": extract_title(content, filepath.name),
        "content": content,
        "word_count": len(content.split()),
        "wiki_links": wiki_links,
        "resolved_links": resolved,
        "tags": extract_tags(content),
        "headings": extract_headings(content),
    }


def main():
    """Parse all notes and build the metadata index."""
    note_files = sorted(NOTES_DIR.glob("*.md"))
    if not note_files:
        print(f"No markdown files found in {NOTES_DIR}")
        return

    print(f"Found {len(note_files)} notes in {NOTES_DIR}\n")

    # Build lookup of available notes: stem -> filename
    available_notes = {f.stem.lower(): f.name for f in note_files}

    # Parse all notes
    note_index = []
    for filepath in note_files:
        note = parse_note(filepath, available_notes)
        note_index.append(note)
        print(f"  Parsed: {note['title']}")
        print(f"    Links: {note['wiki_links']}")
        print(f"    Tags:  {note['tags']}")
        resolved_count = sum(1 for r in note["resolved_links"] if r["resolved"])
        unresolved_count = sum(1 for r in note["resolved_links"] if not r["resolved"])
        print(f"    Resolved: {resolved_count}, Unresolved: {unresolved_count}")
        print()

    # Summary
    all_links = [link for note in note_index for link in note["wiki_links"]]
    all_tags = list({tag for note in note_index for tag in note["tags"]})
    unresolved = [
        r["target"]
        for note in note_index
        for r in note["resolved_links"]
        if not r["resolved"]
    ]

    print("=" * 60)
    print(f"Total notes:      {len(note_index)}")
    print(f"Total links:      {len(all_links)}")
    print(f"Unique tags:      {len(all_tags)} — {sorted(all_tags)}")
    print(f"Unresolved links: {len(unresolved)} — {sorted(set(unresolved))}")

    # Save index (without full content for readability)
    output_index = []
    for note in note_index:
        entry = {k: v for k, v in note.items() if k != "content"}
        entry["content_preview"] = note["content"][:200] + "..."
        output_index.append(entry)

    output_path = OUTPUT_DIR / "note_index.json"
    output_path.write_text(json.dumps(output_index, indent=2), encoding="utf-8")
    print(f"\nSaved note index to {output_path}")

    # Also save full content for downstream use
    full_path = OUTPUT_DIR / "notes_full.json"
    full_path.write_text(json.dumps(note_index, indent=2), encoding="utf-8")
    print(f"Saved full notes to {full_path}")


if __name__ == "__main__":
    main()
