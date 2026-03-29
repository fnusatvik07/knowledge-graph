"""Evaluate entity and relationship extraction quality against a gold standard.

Runs LLM-based extraction on each gold standard document, then compares
extracted entities/relationships against expected ground truth using:
- Precision, Recall, F1 (per-document and aggregate)
- Fuzzy matching for entity names (configurable threshold)

LangChain structured output: https://python.langchain.com/docs/how_to/structured_output/

Usage:
    python src/01_extract_and_evaluate.py
    python src/01_extract_and_evaluate.py --provider openai --threshold 0.8
"""

import sys
import json
import argparse
from pathlib import Path
from difflib import SequenceMatcher

# Shared imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pydantic import BaseModel, Field
from shared.llm_clients import chat_completion_structured

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
GOLD_STANDARD_PATH = PROJECT_DIR / "data" / "gold_standard.json"
OUTPUT_DIR = PROJECT_DIR / "output"


# ── Pydantic schemas for extraction ─────────────────────────────────────

class ExtractedEntity(BaseModel):
    name: str = Field(description="Canonical entity name")
    type: str = Field(description="Entity type: PERSON, ORGANIZATION, LOCATION, PRODUCT, EVENT, OTHER")


class ExtractedRelationship(BaseModel):
    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    type: str = Field(description="Relationship type")


class ExtractionOutput(BaseModel):
    entities: list[ExtractedEntity] = Field(description="Extracted entities")
    relationships: list[ExtractedRelationship] = Field(description="Extracted relationships")


# ── Fuzzy matching ───────────────────────────────────────────────────────

def fuzzy_match(s1: str, s2: str, threshold: float = 0.8) -> bool:
    """Check if two strings are similar enough to be considered a match."""
    s1_lower = s1.strip().lower()
    s2_lower = s2.strip().lower()

    # Exact match
    if s1_lower == s2_lower:
        return True

    # One contains the other
    if s1_lower in s2_lower or s2_lower in s1_lower:
        return True

    # Fuzzy similarity
    ratio = SequenceMatcher(None, s1_lower, s2_lower).ratio()
    return ratio >= threshold


def find_best_match(candidate: str, reference_list: list[str], threshold: float = 0.8) -> str | None:
    """Find the best fuzzy match for a candidate in a reference list."""
    best_match = None
    best_score = 0.0

    for ref in reference_list:
        score = SequenceMatcher(None, candidate.strip().lower(), ref.strip().lower()).ratio()
        if score > best_score and score >= threshold:
            best_score = score
            best_match = ref

    # Also check containment
    for ref in reference_list:
        if candidate.strip().lower() in ref.strip().lower() or ref.strip().lower() in candidate.strip().lower():
            return ref

    return best_match


# ── Extraction ───────────────────────────────────────────────────────────

def extract_from_document(text: str, provider: str = "openai", model: str = None) -> ExtractionOutput:
    """Run LLM-based entity and relationship extraction on a document."""
    prompt = f"""Extract all named entities and relationships from the following text.

For entities, classify them as: PERSON, ORGANIZATION, LOCATION, PRODUCT, EVENT, or OTHER.
For relationships, identify the source entity, target entity, and relationship type.

Text:
{text}

Extract every entity and relationship mentioned."""

    result = chat_completion_structured(
        prompt=prompt,
        output_schema=ExtractionOutput,
        system="You are an expert knowledge graph extraction system. Extract all entities and "
               "relationships precisely. Use canonical names for entities.",
        provider=provider,
        model=model,
    )
    return result


# ── Evaluation metrics ───────────────────────────────────────────────────

def evaluate_entities(
    extracted: list[ExtractedEntity],
    expected: list[dict],
    threshold: float = 0.8,
) -> dict:
    """Evaluate entity extraction quality using precision, recall, F1."""
    expected_names = [e["name"] for e in expected]
    extracted_names = [e.name for e in extracted]

    # True positives: extracted entities that match expected entities
    tp = 0
    matched_expected = set()
    matched_extracted = set()

    for i, ext_name in enumerate(extracted_names):
        match = find_best_match(ext_name, expected_names, threshold)
        if match and match not in matched_expected:
            tp += 1
            matched_expected.add(match)
            matched_extracted.add(i)

    fp = len(extracted_names) - tp  # Extracted but not expected
    fn = len(expected_names) - tp   # Expected but not extracted

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Find missed entities
    missed = [name for name in expected_names if name not in matched_expected]
    extra = [extracted_names[i] for i in range(len(extracted_names)) if i not in matched_extracted]

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "missed": missed,
        "extra": extra,
    }


def evaluate_relationships(
    extracted: list[ExtractedRelationship],
    expected: list[dict],
    threshold: float = 0.8,
) -> dict:
    """Evaluate relationship extraction quality using precision, recall, F1.

    Matches on source-target pairs (fuzzy) regardless of relationship type label.
    """
    tp = 0
    matched_expected = set()
    matched_extracted = set()

    for i, ext_rel in enumerate(extracted):
        for j, exp_rel in enumerate(expected):
            if j in matched_expected:
                continue
            # Match if source and target fuzzy-match
            source_match = fuzzy_match(ext_rel.source, exp_rel["source"], threshold)
            target_match = fuzzy_match(ext_rel.target, exp_rel["target"], threshold)
            if source_match and target_match:
                tp += 1
                matched_expected.add(j)
                matched_extracted.add(i)
                break

    fp = len(extracted) - tp
    fn = len(expected) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    missed = [
        f"{expected[j]['source']} -> {expected[j]['target']}"
        for j in range(len(expected)) if j not in matched_expected
    ]
    extra = [
        f"{extracted[i].source} -> {extracted[i].target}"
        for i in range(len(extracted)) if i not in matched_extracted
    ]

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "missed": missed,
        "extra": extra,
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate entity/relationship extraction")
    parser.add_argument("--provider", type=str, default="openai", help="LLM provider")
    parser.add_argument("--model", type=str, default=None, help="LLM model name")
    parser.add_argument("--threshold", type=float, default=0.8, help="Fuzzy match threshold (0-1)")
    args = parser.parse_args()

    print("=" * 70)
    print("  KG Evaluation: Entity & Relationship Extraction")
    print("=" * 70)
    print(f"\n  Provider: {args.provider}")
    print(f"  Fuzzy match threshold: {args.threshold}\n")

    # Load gold standard
    with open(GOLD_STANDARD_PATH) as f:
        gold = json.load(f)

    documents = gold["documents"]
    all_entity_results = []
    all_relationship_results = []

    for doc in documents:
        print(f"\n{'─'*60}")
        print(f"  Document: {doc['title']} ({doc['doc_id']})")
        print(f"{'─'*60}")

        # Run extraction
        print("  Extracting entities and relationships...")
        extraction = extract_from_document(doc["text"], provider=args.provider, model=args.model)
        print(f"  Extracted: {len(extraction.entities)} entities, {len(extraction.relationships)} relationships")

        # Evaluate entities
        entity_eval = evaluate_entities(extraction.entities, doc["expected_entities"], args.threshold)
        all_entity_results.append(entity_eval)

        print(f"\n  Entity Evaluation:")
        print(f"    Precision: {entity_eval['precision']:.2%}")
        print(f"    Recall:    {entity_eval['recall']:.2%}")
        print(f"    F1:        {entity_eval['f1']:.2%}")
        if entity_eval["missed"]:
            print(f"    Missed:    {', '.join(entity_eval['missed'])}")
        if entity_eval["extra"]:
            print(f"    Extra:     {', '.join(entity_eval['extra'])}")

        # Evaluate relationships
        rel_eval = evaluate_relationships(extraction.relationships, doc["expected_relationships"], args.threshold)
        all_relationship_results.append(rel_eval)

        print(f"\n  Relationship Evaluation:")
        print(f"    Precision: {rel_eval['precision']:.2%}")
        print(f"    Recall:    {rel_eval['recall']:.2%}")
        print(f"    F1:        {rel_eval['f1']:.2%}")
        if rel_eval["missed"]:
            print(f"    Missed:    {', '.join(rel_eval['missed'][:5])}")
        if rel_eval["extra"]:
            print(f"    Extra:     {', '.join(rel_eval['extra'][:5])}")

    # Aggregate results
    print(f"\n\n{'='*70}")
    print("  AGGREGATE RESULTS")
    print(f"{'='*70}")

    # Entity aggregate
    total_tp_e = sum(r["true_positives"] for r in all_entity_results)
    total_fp_e = sum(r["false_positives"] for r in all_entity_results)
    total_fn_e = sum(r["false_negatives"] for r in all_entity_results)
    agg_prec_e = total_tp_e / (total_tp_e + total_fp_e) if (total_tp_e + total_fp_e) > 0 else 0
    agg_rec_e = total_tp_e / (total_tp_e + total_fn_e) if (total_tp_e + total_fn_e) > 0 else 0
    agg_f1_e = 2 * agg_prec_e * agg_rec_e / (agg_prec_e + agg_rec_e) if (agg_prec_e + agg_rec_e) > 0 else 0

    print(f"\n  Entity Extraction (Micro-averaged):")
    print(f"    Precision: {agg_prec_e:.2%}")
    print(f"    Recall:    {agg_rec_e:.2%}")
    print(f"    F1:        {agg_f1_e:.2%}")

    # Relationship aggregate
    total_tp_r = sum(r["true_positives"] for r in all_relationship_results)
    total_fp_r = sum(r["false_positives"] for r in all_relationship_results)
    total_fn_r = sum(r["false_negatives"] for r in all_relationship_results)
    agg_prec_r = total_tp_r / (total_tp_r + total_fp_r) if (total_tp_r + total_fp_r) > 0 else 0
    agg_rec_r = total_tp_r / (total_tp_r + total_fn_r) if (total_tp_r + total_fn_r) > 0 else 0
    agg_f1_r = 2 * agg_prec_r * agg_rec_r / (agg_prec_r + agg_rec_r) if (agg_prec_r + agg_rec_r) > 0 else 0

    print(f"\n  Relationship Extraction (Micro-averaged):")
    print(f"    Precision: {agg_prec_r:.2%}")
    print(f"    Recall:    {agg_rec_r:.2%}")
    print(f"    F1:        {agg_f1_r:.2%}")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "entity_aggregate": {"precision": agg_prec_e, "recall": agg_rec_e, "f1": agg_f1_e},
        "relationship_aggregate": {"precision": agg_prec_r, "recall": agg_rec_r, "f1": agg_f1_r},
        "per_document": [
            {
                "doc_id": doc["doc_id"],
                "entity_eval": all_entity_results[i],
                "relationship_eval": all_relationship_results[i],
            }
            for i, doc in enumerate(documents)
        ],
    }
    output_path = OUTPUT_DIR / "extraction_evaluation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
