"""Extract biomedical relationships from PubMed abstracts using ontology constraints.

Uses the biomedical ontology to guide relationship extraction, ensuring that
extracted relationships conform to valid source/target type pairs.

Usage:
    python src/03_extract_relations.py
    python src/03_extract_relations.py --provider anthropic

References:
    - LangChain structured output: https://python.langchain.com/docs/how_to/structured_output/
    - LangChain chat models: https://python.langchain.com/docs/integrations/chat/
"""

import argparse
import json
import sys
from pathlib import Path

# ── Shared imports ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared.llm_clients import chat_completion_structured

from ontology import (
    EntityType,
    RelationshipType,
    RELATIONSHIP_CONSTRAINTS,
    ExtractedRelationships,
    BiomedicalRelationship,
    validate_relationship,
)

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"


def build_constraints_description() -> str:
    """Build a human-readable description of valid relationship constraints."""
    lines = []
    for rel_type, constraints in RELATIONSHIP_CONSTRAINTS.items():
        sources = ", ".join(t.value for t in constraints["source"])
        targets = ", ".join(t.value for t in constraints["target"])
        lines.append(f"  - {rel_type.value}: ({sources}) -> ({targets})")
    return "\n".join(lines)


SYSTEM_PROMPT = f"""You are a biomedical relationship extraction expert.
Given a PubMed abstract and a list of entities found in it, extract all
relationships between the entities.

Valid relationship types and their source/target constraints:
{build_constraints_description()}

Guidelines:
- Only extract relationships that are explicitly stated or strongly implied in the text
- Use the EXACT entity names from the provided entity list
- Assign a confidence score: 1.0 for explicitly stated, 0.7-0.9 for strongly implied
- Include the supporting evidence (text span) for each relationship
- Respect the source/target type constraints listed above
- A single abstract may contain multiple relationships between different entity pairs
"""


def extract_relations_from_abstract(
    abstract: dict,
    entities: list[dict],
    provider: str = "openai",
    model: str | None = None,
) -> list[dict]:
    """Extract biomedical relationships from a single abstract.

    Args:
        abstract: Dict with 'pmid', 'title', 'abstract' keys.
        entities: List of entity dicts found in this abstract.
        provider: LLM provider.
        model: Model name.

    Returns:
        List of relationship dicts with pmid attached.
    """
    # Format entities for the prompt
    entity_lines = []
    for e in entities:
        entity_lines.append(f"  - {e['name']} ({e['entity_type']})")
    entities_str = "\n".join(entity_lines)

    text = f"Title: {abstract['title']}\n\nAbstract: {abstract['abstract']}"

    prompt = f"""Extract all biomedical relationships from the following PubMed abstract.

{text}

Entities found in this abstract:
{entities_str}

Extract relationships between these entities, following the type constraints."""

    result: ExtractedRelationships = chat_completion_structured(
        prompt=prompt,
        output_schema=ExtractedRelationships,
        system=SYSTEM_PROMPT,
        provider=provider,
        model=model,
    )

    relationships = []
    valid_count = 0
    invalid_count = 0

    for rel in result.relationships:
        # Validate against ontology constraints
        if validate_relationship(rel):
            rel_dict = rel.model_dump()
            rel_dict["pmid"] = abstract["pmid"]
            relationships.append(rel_dict)
            valid_count += 1
        else:
            invalid_count += 1
            print(f"    [FILTERED] Invalid constraint: {rel.source_name} ({rel.source_type.value}) "
                  f"-[{rel.relationship_type.value}]-> {rel.target_name} ({rel.target_type.value})")

    if invalid_count > 0:
        print(f"    Filtered {invalid_count} relationships that violated ontology constraints")

    return relationships


def main():
    parser = argparse.ArgumentParser(description="Extract biomedical relationships from abstracts.")
    parser.add_argument("--provider", type=str, default="openai",
                        choices=["openai", "anthropic", "ollama"])
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    # Load abstracts
    fetched = RAW_DIR / "pubmed_abstracts.json"
    input_file = fetched if fetched.exists() else RAW_DIR / "sample_abstracts.json"
    print(f"Loading abstracts from {input_file}...")
    with open(input_file) as f:
        abstracts = json.load(f)

    # Load extracted entities
    entities_file = PROCESSED_DIR / "extracted_entities.json"
    if not entities_file.exists():
        print(f"ERROR: Entities file not found at {entities_file}")
        print("Run 02_extract_biomedical_entities.py first.")
        sys.exit(1)

    with open(entities_file) as f:
        all_entities = json.load(f)

    # Group entities by PMID
    entities_by_pmid = {}
    for entity in all_entities:
        pmid = entity["pmid"]
        entities_by_pmid.setdefault(pmid, []).append(entity)

    print(f"Extracting relationships from {len(abstracts)} abstracts...")

    all_relationships = []
    for i, abstract in enumerate(abstracts):
        pmid = abstract["pmid"]
        entities = entities_by_pmid.get(pmid, [])

        if not entities:
            print(f"\n[{i+1}/{len(abstracts)}] PMID {pmid}: No entities found, skipping.")
            continue

        print(f"\n[{i+1}/{len(abstracts)}] PMID {pmid}: {abstract['title'][:60]}...")
        print(f"  Using {len(entities)} entities")

        try:
            relationships = extract_relations_from_abstract(
                abstract, entities, provider=args.provider, model=args.model
            )
            all_relationships.extend(relationships)
            print(f"  Extracted {len(relationships)} valid relationships")

            for rel in relationships:
                print(f"    {rel['source_name']} -[{rel['relationship_type']}]-> {rel['target_name']} "
                      f"(conf: {rel['confidence']:.2f})")

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Save results
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_file = PROCESSED_DIR / "extracted_relationships.json"
    with open(output_file, "w") as f:
        json.dump(all_relationships, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Total relationships extracted: {len(all_relationships)}")
    print(f"Saved to {output_file}")

    # Summary by relationship type
    type_summary = {}
    for r in all_relationships:
        t = r["relationship_type"]
        type_summary[t] = type_summary.get(t, 0) + 1
    print("\nSummary by relationship type:")
    for t, c in sorted(type_summary.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")


if __name__ == "__main__":
    main()
