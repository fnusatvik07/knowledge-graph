"""Biomedical ontology definitions for the knowledge graph.

Defines entity types, relationship types, and constraints for the biomedical domain.
Uses Pydantic models for validation and structured LLM extraction.

References:
    - LangChain structured output: https://python.langchain.com/docs/how_to/structured_output/
    - Pydantic models: https://docs.pydantic.dev/latest/
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Entity Types ─────────────────────────────────────────────────────────


class EntityType(str, Enum):
    """Biomedical entity types supported by the knowledge graph."""
    GENE = "Gene"
    PROTEIN = "Protein"
    DISEASE = "Disease"
    DRUG = "Drug"
    PATHWAY = "Pathway"
    CELL_TYPE = "CellType"
    ORGANISM = "Organism"


class BiomedicalEntity(BaseModel):
    """A single biomedical entity extracted from text."""
    name: str = Field(description="Canonical name of the entity (e.g., 'TP53', 'breast cancer', 'metformin')")
    entity_type: EntityType = Field(description="The type of biomedical entity")
    synonyms: list[str] = Field(default_factory=list, description="Alternative names or abbreviations")
    description: Optional[str] = Field(default=None, description="Brief description of the entity")
    source_text: Optional[str] = Field(default=None, description="The text span where this entity was found")


# ── Relationship Types ───────────────────────────────────────────────────


class RelationshipType(str, Enum):
    """Biomedical relationship types with semantic meaning."""
    TREATS = "TREATS"
    CAUSES = "CAUSES"
    INHIBITS = "INHIBITS"
    ACTIVATES = "ACTIVATES"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    EXPRESSED_IN = "EXPRESSED_IN"
    TARGETS = "TARGETS"
    INTERACTS_WITH = "INTERACTS_WITH"


# Source/target constraints for each relationship type
RELATIONSHIP_CONSTRAINTS: dict[RelationshipType, dict[str, list[EntityType]]] = {
    RelationshipType.TREATS: {
        "source": [EntityType.DRUG],
        "target": [EntityType.DISEASE],
    },
    RelationshipType.CAUSES: {
        "source": [EntityType.GENE, EntityType.DRUG],
        "target": [EntityType.DISEASE],
    },
    RelationshipType.INHIBITS: {
        "source": [EntityType.DRUG],
        "target": [EntityType.GENE, EntityType.PROTEIN],
    },
    RelationshipType.ACTIVATES: {
        "source": [EntityType.DRUG, EntityType.GENE],
        "target": [EntityType.GENE, EntityType.PROTEIN, EntityType.PATHWAY],
    },
    RelationshipType.ASSOCIATED_WITH: {
        "source": [EntityType.GENE, EntityType.PROTEIN],
        "target": [EntityType.DISEASE],
    },
    RelationshipType.EXPRESSED_IN: {
        "source": [EntityType.GENE, EntityType.PROTEIN],
        "target": [EntityType.CELL_TYPE],
    },
    RelationshipType.TARGETS: {
        "source": [EntityType.DRUG],
        "target": [EntityType.PROTEIN],
    },
    RelationshipType.INTERACTS_WITH: {
        "source": [EntityType.PROTEIN],
        "target": [EntityType.PROTEIN],
    },
}


class BiomedicalRelationship(BaseModel):
    """A relationship between two biomedical entities."""
    source_name: str = Field(description="Name of the source entity")
    source_type: EntityType = Field(description="Type of the source entity")
    relationship_type: RelationshipType = Field(description="The type of relationship")
    target_name: str = Field(description="Name of the target entity")
    target_type: EntityType = Field(description="Type of the target entity")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    evidence: Optional[str] = Field(default=None, description="Supporting text from the source")


# ── Extraction Schemas (for LLM structured output) ──────────────────────


class ExtractedEntities(BaseModel):
    """Collection of entities extracted from a single abstract."""
    entities: list[BiomedicalEntity] = Field(
        description="All biomedical entities found in the text"
    )


class ExtractedRelationships(BaseModel):
    """Collection of relationships extracted from a single abstract."""
    relationships: list[BiomedicalRelationship] = Field(
        description="All biomedical relationships found in the text"
    )


# ── Validation ───────────────────────────────────────────────────────────


def validate_relationship(rel: BiomedicalRelationship) -> bool:
    """Check if a relationship satisfies the ontology constraints.

    Args:
        rel: A BiomedicalRelationship to validate.

    Returns:
        True if the relationship satisfies source/target type constraints.
    """
    constraints = RELATIONSHIP_CONSTRAINTS.get(rel.relationship_type)
    if constraints is None:
        return False

    source_valid = rel.source_type in constraints["source"]
    target_valid = rel.target_type in constraints["target"]
    return source_valid and target_valid


def get_valid_relationships_for_entity(entity_type: EntityType) -> list[dict]:
    """Get all valid relationship types where this entity type can be source or target.

    Args:
        entity_type: The entity type to check.

    Returns:
        List of dicts with 'relationship', 'role' (source/target), and valid partner types.
    """
    results = []
    for rel_type, constraints in RELATIONSHIP_CONSTRAINTS.items():
        if entity_type in constraints["source"]:
            results.append({
                "relationship": rel_type,
                "role": "source",
                "partner_types": constraints["target"],
            })
        if entity_type in constraints["target"]:
            results.append({
                "relationship": rel_type,
                "role": "target",
                "partner_types": constraints["source"],
            })
    return results
