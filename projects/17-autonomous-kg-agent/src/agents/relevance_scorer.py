"""Relevance Scorer -- evaluates candidate documents against the current
KG state to decide whether they should be ingested.

Considers: entity overlap, coverage of gap areas, and content quality.
"""

import sys
from pathlib import Path

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
from shared.llm_clients import chat_completion_structured


# ── Schemas ──────────────────────────────────────────────────────────

class RelevanceAssessment(BaseModel):
    relevance_score: float = Field(description="Relevance score from 0.0 (irrelevant) to 1.0 (highly relevant)")
    reason: str = Field(description="Explanation for the relevance score")
    entities_found: list[str] = Field(default_factory=list, description="Entities in the document that relate to the KG domain")
    covers_gaps: bool = Field(description="Whether the document covers known gap areas in the KG")
    quality_assessment: str = Field(description="Assessment of document quality: high, medium, or low")


def score_relevance(source: dict, kg_stats: dict) -> dict:
    """Evaluate a candidate document for relevance to the current KG.

    Args:
        source: dict with keys: id, title, text
        kg_stats: Current KG statistics (entity types, counts, etc.)

    Returns:
        dict with: relevance_score (0-1), reason, entities_found, covers_gaps, quality_assessment
    """
    # Build context about current KG
    entity_types = kg_stats.get("entity_types", {})
    total_entities = kg_stats.get("total_entities", 0)
    coverage = kg_stats.get("coverage_score", 0.0)

    kg_context = f"""Current Knowledge Graph state:
- Total entities: {total_entities}
- Entity types: {entity_types}
- Coverage score: {coverage:.2f}
- Domain: Artificial Intelligence and related technologies"""

    assessment = chat_completion_structured(
        prompt=f"""Evaluate this candidate document for relevance to our knowledge graph.

{kg_context}

Candidate Document:
Title: {source.get('title', 'Unknown')}
Text: {source.get('text', '')}

Consider:
1. Does this document contain entities already in our KG? (entity overlap)
2. Does it cover topics or subtopics that are missing from our KG? (gap coverage)
3. Is the content factual and high quality?
4. Is it relevant to the domain of AI and technology?

Score from 0.0 (completely irrelevant, e.g., cooking recipes) to 1.0 (highly relevant and filling gaps).""",
        output_schema=RelevanceAssessment,
        system="You are evaluating documents for relevance to an AI knowledge graph. Be strict about relevance.",
    )

    return assessment.model_dump()


if __name__ == "__main__":
    # Test with a relevant document
    result = score_relevance(
        source={
            "id": "test",
            "title": "Deep Learning Advances",
            "text": "Deep learning has revolutionized AI with architectures like Transformers and CNNs.",
        },
        kg_stats={"total_entities": 10, "entity_types": {"Technology": 5, "Person": 3}},
    )
    print(f"Score: {result['relevance_score']}")
    print(f"Reason: {result['reason']}")
