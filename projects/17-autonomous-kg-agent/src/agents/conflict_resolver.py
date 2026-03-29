"""Conflict Resolver -- handles contradictory facts in the knowledge graph.

When a new fact contradicts an existing one, assesses reliability using
source authority, recency, and corroboration. Decides whether to keep
the existing fact, replace it, or flag it for human review.
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone

from pydantic import BaseModel, Field
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
from shared.llm_clients import chat_completion_structured
from shared.config import get_neo4j_config


# ── Schemas ──────────────────────────────────────────────────────────

class ConflictAssessment(BaseModel):
    has_conflict: bool = Field(description="Whether a genuine conflict exists")
    existing_fact: str = Field(description="The existing fact in the KG")
    new_fact: str = Field(description="The new conflicting fact")
    resolution: str = Field(description="One of: keep_existing, replace, flag_for_review")
    reason: str = Field(description="Explanation for the resolution")
    confidence: float = Field(description="Confidence in the resolution 0-1")


class ConflictBatch(BaseModel):
    conflicts: list[ConflictAssessment] = Field(default_factory=list)


def _get_driver():
    config = get_neo4j_config()
    return GraphDatabase.driver(config["uri"], auth=(config["user"], config["password"]))


def resolve_conflicts(extraction) -> list[dict]:
    """Check extracted facts against existing KG and resolve conflicts.

    Args:
        extraction: ExtractionResult with entities and relationships

    Returns:
        List of conflict resolution dicts (empty if no conflicts found)
    """
    driver = _get_driver()
    conflicts_found = []
    now = datetime.now(timezone.utc).isoformat()

    with driver.session() as session:
        # Check each extracted entity against existing ones
        for entity in extraction.entities:
            result = session.run(
                """
                MATCH (e:Entity {name: $name})
                RETURN e.entity_type AS existing_type,
                       e.description AS existing_description,
                       e.source AS existing_source,
                       e.updated_at AS existing_updated_at
                """,
                name=entity.name,
            )
            record = result.single()
            if record is None:
                continue  # New entity, no conflict

            existing_type = record["existing_type"]
            existing_desc = record["existing_description"] or ""

            # Check for type conflict
            if existing_type and entity.entity_type and existing_type != entity.entity_type:
                conflicts_found.append({
                    "entity": entity.name,
                    "field": "entity_type",
                    "existing_value": existing_type,
                    "new_value": entity.entity_type,
                    "existing_source": record["existing_source"],
                    "existing_updated": record["existing_updated_at"],
                })

            # Check for description conflict (significant difference)
            if existing_desc and entity.description:
                if len(existing_desc) > 20 and len(entity.description) > 20:
                    # Only flag if both descriptions are substantial
                    conflicts_found.append({
                        "entity": entity.name,
                        "field": "description",
                        "existing_value": existing_desc[:200],
                        "new_value": entity.description[:200],
                        "existing_source": record["existing_source"],
                        "existing_updated": record["existing_updated_at"],
                    })

    driver.close()

    if not conflicts_found:
        return []

    # Use LLM to assess and resolve conflicts
    conflicts_text = json.dumps(conflicts_found, indent=2, default=str)

    assessment = chat_completion_structured(
        prompt=f"""Analyze these potential conflicts between existing facts in a knowledge graph
and newly extracted facts. For each conflict, decide how to resolve it.

Conflicts:
{conflicts_text}

Consider:
- Source authority (is the new source more authoritative?)
- Recency (is the new information more up-to-date?)
- Specificity (is one description more specific/accurate?)
- For entity type conflicts: which type is more appropriate?

Resolution options:
- keep_existing: the existing fact is correct or more reliable
- replace: the new fact should replace the existing one
- flag_for_review: the conflict is ambiguous and needs human review""",
        output_schema=ConflictBatch,
        system="You are a knowledge graph conflict resolution specialist. Be conservative -- prefer keeping existing data unless the new data is clearly better.",
    )

    # Apply resolutions
    resolved = []
    driver = _get_driver()

    with driver.session() as session:
        for conflict in assessment.conflicts:
            if conflict.resolution == "replace":
                # Find the original conflict data to get the entity name
                matching = [c for c in conflicts_found if c["existing_value"] in conflict.existing_fact]
                if matching:
                    entity_name = matching[0]["entity"]
                    field = matching[0]["field"]
                    new_value = matching[0]["new_value"]

                    if field == "entity_type":
                        session.run(
                            """
                            MATCH (e:Entity {name: $name})
                            SET e.entity_type = $new_value,
                                e.conflict_resolved_at = $now,
                                e.conflict_resolution = 'replaced'
                            """,
                            name=entity_name,
                            new_value=new_value,
                            now=now,
                        )
                    elif field == "description":
                        session.run(
                            """
                            MATCH (e:Entity {name: $name})
                            SET e.description = $new_value,
                                e.conflict_resolved_at = $now,
                                e.conflict_resolution = 'replaced'
                            """,
                            name=entity_name,
                            new_value=new_value,
                            now=now,
                        )

            elif conflict.resolution == "flag_for_review":
                matching = [c for c in conflicts_found if c["existing_value"] in conflict.existing_fact]
                if matching:
                    entity_name = matching[0]["entity"]
                    session.run(
                        """
                        MATCH (e:Entity {name: $name})
                        SET e.needs_review = true,
                            e.review_reason = $reason,
                            e.flagged_at = $now
                        """,
                        name=entity_name,
                        reason=conflict.reason,
                        now=now,
                    )

            resolved.append(conflict.model_dump())

    driver.close()
    return resolved


if __name__ == "__main__":
    from pydantic import BaseModel as BM

    class TestEntity(BM):
        name: str
        entity_type: str
        description: str

    class TestExtraction(BM):
        entities: list[TestEntity]
        relationships: list = []

    extraction = TestExtraction(
        entities=[
            TestEntity(name="GPT-4", entity_type="Model", description="A large language model by OpenAI"),
        ]
    )
    conflicts = resolve_conflicts(extraction)
    print(f"Conflicts resolved: {len(conflicts)}")
    for c in conflicts:
        print(f"  {c['resolution']}: {c['reason']}")
