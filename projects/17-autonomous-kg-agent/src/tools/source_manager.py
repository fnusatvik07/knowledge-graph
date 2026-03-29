"""Source Manager -- manages document sources for the autonomous agent.

Tracks which sources have been processed, their status, and provides
functions for evaluating and prioritizing candidate sources.
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

# ── State tracking ───────────────────────────────────────────────────
# We use a JSON file to track processing state (simple, no extra DB needed).

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_DIR / "data"
STATE_FILE = PROJECT_DIR / "output" / "source_state.json"


def _load_state() -> dict:
    """Load source processing state from disk."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"processed": {}, "history": []}


def _save_state(state: dict) -> None:
    """Save source processing state to disk."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def _load_candidate_sources() -> list[dict]:
    """Load candidate sources from data/candidate_sources.json."""
    sources_path = DATA_DIR / "candidate_sources.json"
    if not sources_path.exists():
        return []
    with open(sources_path) as f:
        return json.load(f)


def list_pending_sources() -> list[dict]:
    """Get sources that have not yet been processed.

    Returns:
        List of source dicts with id, title, text, status='pending'
    """
    state = _load_state()
    candidates = _load_candidate_sources()

    pending = []
    for source in candidates:
        source_id = source.get("id", "")
        if source_id not in state.get("processed", {}):
            source_copy = dict(source)
            source_copy["status"] = "pending"
            pending.append(source_copy)

    return pending


def evaluate_source(source: dict) -> dict:
    """Get basic evaluation metadata for a source.

    Args:
        source: dict with id, title, text

    Returns:
        dict with word_count, has_title, estimated_entities
    """
    text = source.get("text", "")
    title = source.get("title", "")

    word_count = len(text.split())
    # Simple heuristic for entity detection (capitalized words)
    words = text.split()
    potential_entities = set()
    for i, word in enumerate(words):
        if word and word[0].isupper() and len(word) > 2:
            # Check if it's not at the start of a sentence
            if i > 0 and not words[i - 1].endswith((".","!","?")):
                potential_entities.add(word.strip(".,;:()\"'"))

    return {
        "source_id": source.get("id", ""),
        "word_count": word_count,
        "has_title": bool(title),
        "estimated_entities": len(potential_entities),
        "potential_entities": sorted(list(potential_entities))[:10],
    }


def mark_processed(source_id: str, status: str = "ingested") -> None:
    """Mark a source as processed.

    Args:
        source_id: The source identifier
        status: Processing status (ingested, skipped, error)
    """
    state = _load_state()
    now = datetime.now(timezone.utc).isoformat()

    state["processed"][source_id] = {
        "status": status,
        "processed_at": now,
    }
    state["history"].append({
        "source_id": source_id,
        "status": status,
        "timestamp": now,
    })

    _save_state(state)


def get_processing_history() -> list[dict]:
    """Get the full processing history.

    Returns:
        List of dicts with source_id, status, timestamp
    """
    state = _load_state()
    return state.get("history", [])


def get_source_by_id(source_id: str) -> dict | None:
    """Get a specific source by its ID.

    Args:
        source_id: The source identifier

    Returns:
        Source dict or None if not found
    """
    candidates = _load_candidate_sources()
    for source in candidates:
        if source.get("id") == source_id:
            return source
    return None


def reset_processing_state() -> None:
    """Reset all processing state (for re-running the demo)."""
    if STATE_FILE.exists():
        STATE_FILE.unlink()


if __name__ == "__main__":
    print("Pending sources:")
    for s in list_pending_sources():
        eval_result = evaluate_source(s)
        print(f"  [{s['id']}] {s['title']} - {eval_result['word_count']} words, ~{eval_result['estimated_entities']} entities")

    print(f"\nProcessing history: {get_processing_history()}")
