"""Step 1: Extract entities from sample articles using LLM.

Reads text files from data/sample_articles/, sends each to the configured LLM
for entity extraction, and saves the combined results to output/entities.json.
"""
# LangChain docs: https://python.langchain.com/docs/integrations/chat/

import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel, Field

# Add projects dir to path for shared imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared.llm_clients import chat_completion_structured
from shared.document_loader import load_text_files
from prompts import ENTITY_EXTRACTION_SYSTEM, ENTITY_EXTRACTION_USER

DATA_DIR = Path(__file__).parent.parent / "data" / "sample_articles"
OUTPUT_DIR = Path(__file__).parent.parent / "output"


# -- Pydantic models for structured extraction --------------------------------
# LangChain structured output: https://python.langchain.com/docs/how_to/structured_output/

class ExtractedEntity(BaseModel):
    """A single entity extracted from text."""
    name: str = Field(description="Canonical entity name")
    type: str = Field(description="Entity type (PERSON, ORGANIZATION, TECHNOLOGY, CONCEPT, EVENT)")
    description: str = Field(description="Brief description based on the text")


class EntityList(BaseModel):
    """List of entities extracted from a document."""
    entities: list[ExtractedEntity] = Field(default_factory=list)


def extract_entities(text: str, provider: str = "openai") -> list[dict]:
    """Extract entities from a text using structured LLM output."""
    result: EntityList = chat_completion_structured(
        prompt=ENTITY_EXTRACTION_USER.format(text=text),
        output_schema=EntityList,
        system=ENTITY_EXTRACTION_SYSTEM,
        provider=provider,
    )
    return [e.model_dump() for e in result.entities]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract entities from sample articles")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "ollama"],
        default="openai",
        help="LLM provider to use (default: openai)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Using LLM provider: {args.provider}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    documents = load_text_files(DATA_DIR)
    print(f"Loaded {len(documents)} documents")

    all_entities = []
    for doc in documents:
        print(f"\nProcessing: {doc['title']}")
        entities = extract_entities(doc["content"], provider=args.provider)
        for entity in entities:
            entity["source_document"] = doc["title"]
        all_entities.extend(entities)
        print(f"  Found {len(entities)} entities")

    # Deduplicate by name (keep richest description)
    merged = {}
    for entity in all_entities:
        key = entity["name"].strip().lower()
        if key in merged:
            existing = merged[key]
            if len(entity["description"]) > len(existing["description"]):
                existing["description"] = entity["description"]
            if entity["source_document"] not in existing.get("sources", []):
                existing.setdefault("sources", []).append(entity["source_document"])
        else:
            entity["sources"] = [entity.pop("source_document")]
            merged[key] = entity

    unique_entities = list(merged.values())

    output_path = OUTPUT_DIR / "entities.json"
    with open(output_path, "w") as f:
        json.dump(unique_entities, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Total unique entities: {len(unique_entities)}")
    print(f"Saved to: {output_path}")
    print(f"\nEntity types:")
    type_counts = {}
    for e in unique_entities:
        type_counts[e["type"]] = type_counts.get(e["type"], 0) + 1
    for t, count in sorted(type_counts.items()):
        print(f"  {t}: {count}")


if __name__ == "__main__":
    main()
