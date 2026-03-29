"""Extract biomedical entities from PubMed abstracts using LLM-based NER.

Uses the biomedical ontology and LangChain's structured output to extract
genes, diseases, drugs, proteins, pathways, cell types, and organisms.

Usage:
    python src/02_extract_biomedical_entities.py
    python src/02_extract_biomedical_entities.py --provider anthropic
    python src/02_extract_biomedical_entities.py --input data/raw/sample_abstracts.json

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
    ExtractedEntities,
    BiomedicalEntity,
)

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

SYSTEM_PROMPT = """You are a biomedical named entity recognition (NER) expert.
Given a PubMed abstract, extract all biomedical entities with their types.

Entity types you must identify:
- Gene: Human or model organism genes (e.g., TP53, BRCA1, KRAS, EGFR)
- Protein: Protein products, enzymes, receptors (e.g., p53, PARP1, HER2, VEGFR2)
- Disease: Diseases, conditions, syndromes (e.g., breast cancer, Parkinson's disease)
- Drug: Drugs, compounds, therapeutics (e.g., olaparib, metformin, trastuzumab)
- Pathway: Biological signaling pathways (e.g., MAPK signaling, PI3K-AKT-mTOR, apoptosis)
- CellType: Cell types mentioned (e.g., T-cell, hepatocyte, dopaminergic neuron)
- Organism: Organisms or species (e.g., Homo sapiens, Mus musculus)

Guidelines:
- Use canonical names where possible (gene symbols in uppercase, drug generic names)
- Include synonyms and abbreviations found in the text
- Do NOT include general biomedical terms that are not specific entities
- Include the source text span where each entity appears
- Be comprehensive -- extract ALL entities, not just the most prominent ones
"""


def extract_entities_from_abstract(
    abstract: dict,
    provider: str = "openai",
    model: str | None = None,
) -> list[dict]:
    """Extract biomedical entities from a single abstract.

    Args:
        abstract: Dict with 'pmid', 'title', 'abstract' keys.
        provider: LLM provider to use.
        model: Model name (defaults per provider).

    Returns:
        List of entity dicts with pmid attached.
    """
    text = f"Title: {abstract['title']}\n\nAbstract: {abstract['abstract']}"

    prompt = f"""Extract all biomedical entities from the following PubMed abstract.

{text}

Return every gene, protein, disease, drug, pathway, cell type, and organism mentioned."""

    result: ExtractedEntities = chat_completion_structured(
        prompt=prompt,
        output_schema=ExtractedEntities,
        system=SYSTEM_PROMPT,
        provider=provider,
        model=model,
    )

    entities = []
    for entity in result.entities:
        entity_dict = entity.model_dump()
        entity_dict["pmid"] = abstract["pmid"]
        entities.append(entity_dict)

    return entities


def main():
    parser = argparse.ArgumentParser(description="Extract biomedical entities from abstracts.")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to input abstracts JSON file")
    parser.add_argument("--provider", type=str, default="openai",
                        choices=["openai", "anthropic", "ollama"],
                        help="LLM provider")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (uses provider default if not set)")
    args = parser.parse_args()

    # Determine input file
    if args.input:
        input_file = Path(args.input)
    else:
        # Prefer fetched data, fall back to samples
        fetched = RAW_DIR / "pubmed_abstracts.json"
        input_file = fetched if fetched.exists() else RAW_DIR / "sample_abstracts.json"

    print(f"Loading abstracts from {input_file}...")
    with open(input_file) as f:
        abstracts = json.load(f)

    print(f"Extracting entities from {len(abstracts)} abstracts...")

    all_entities = []
    for i, abstract in enumerate(abstracts):
        print(f"\n[{i+1}/{len(abstracts)}] Processing PMID {abstract['pmid']}: {abstract['title'][:60]}...")
        try:
            entities = extract_entities_from_abstract(
                abstract, provider=args.provider, model=args.model
            )
            all_entities.extend(entities)
            print(f"  Extracted {len(entities)} entities")

            # Show entity types found
            type_counts = {}
            for e in entities:
                t = e["entity_type"]
                type_counts[t] = type_counts.get(t, 0) + 1
            for t, c in sorted(type_counts.items()):
                print(f"    {t}: {c}")

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Save results
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_file = PROCESSED_DIR / "extracted_entities.json"
    with open(output_file, "w") as f:
        json.dump(all_entities, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Total entities extracted: {len(all_entities)}")
    print(f"Saved to {output_file}")

    # Summary by type
    type_summary = {}
    for e in all_entities:
        t = e["entity_type"]
        type_summary[t] = type_summary.get(t, 0) + 1
    print("\nSummary by entity type:")
    for t, c in sorted(type_summary.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")


if __name__ == "__main__":
    main()
