"""Step 2: Extract relationships between entities using LLM.

Reads entities from output/entities.json, processes each source document
to find relationships, and saves results to output/relationships.json.
"""
# LangChain docs: https://python.langchain.com/docs/integrations/chat/

import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared.llm_clients import chat_completion_structured
from shared.document_loader import load_text_files
from prompts import RELATIONSHIP_EXTRACTION_SYSTEM, RELATIONSHIP_EXTRACTION_USER

DATA_DIR = Path(__file__).parent.parent / "data" / "sample_articles"
OUTPUT_DIR = Path(__file__).parent.parent / "output"


# -- Pydantic models for structured extraction --------------------------------
# LangChain structured output: https://python.langchain.com/docs/how_to/structured_output/

class ExtractedRelationship(BaseModel):
    """A single relationship extracted from text."""
    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    relation_type: str = Field(description="Relationship type (e.g. DEVELOPED, FOUNDED, WORKS_AT)")
    description: str = Field(description="Sentence describing the relationship")


class RelationshipList(BaseModel):
    """List of relationships extracted from a document."""
    relationships: list[ExtractedRelationship] = Field(default_factory=list)


def extract_relationships(text: str, entity_names: list[str], provider: str = "openai") -> list[dict]:
    """Extract relationships from text given a list of known entities."""
    entities_str = "\n".join(f"- {name}" for name in entity_names)
    result: RelationshipList = chat_completion_structured(
        prompt=RELATIONSHIP_EXTRACTION_USER.format(entities=entities_str, text=text),
        output_schema=RelationshipList,
        system=RELATIONSHIP_EXTRACTION_SYSTEM,
        provider=provider,
    )
    return [r.model_dump() for r in result.relationships]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract relationships between entities")
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

    # Load previously extracted entities
    entities_path = OUTPUT_DIR / "entities.json"
    if not entities_path.exists():
        print("Error: Run 01_extract_entities.py first!")
        return

    with open(entities_path) as f:
        entities = json.load(f)

    entity_names = [e["name"] for e in entities]
    print(f"Loaded {len(entity_names)} entities")

    # Process each document
    documents = load_text_files(DATA_DIR)
    all_relationships = []

    for doc in documents:
        print(f"\nProcessing: {doc['title']}")
        relationships = extract_relationships(doc["content"], entity_names, provider=args.provider)
        for rel in relationships:
            rel["source_document"] = doc["title"]
        all_relationships.extend(relationships)
        print(f"  Found {len(relationships)} relationships")

    # Deduplicate relationships
    seen = set()
    unique_relationships = []
    for rel in all_relationships:
        key = (rel["source"].lower(), rel["target"].lower(), rel["relation_type"])
        if key not in seen:
            seen.add(key)
            unique_relationships.append(rel)

    output_path = OUTPUT_DIR / "relationships.json"
    with open(output_path, "w") as f:
        json.dump(unique_relationships, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Total unique relationships: {len(unique_relationships)}")
    print(f"Saved to: {output_path}")
    print(f"\nRelationship types:")
    type_counts = {}
    for r in unique_relationships:
        type_counts[r["relation_type"]] = type_counts.get(r["relation_type"], 0) + 1
    for t, count in sorted(type_counts.items()):
        print(f"  {t}: {count}")


if __name__ == "__main__":
    main()
