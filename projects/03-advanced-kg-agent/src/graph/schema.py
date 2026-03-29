"""Ontology definition for the knowledge graph.

Defines entity types, relationship types with source/target constraints,
and property schemas using Pydantic models.
"""

from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))


# ---------------------------------------------------------------------------
# Entity types
# ---------------------------------------------------------------------------

class EntityType(str, Enum):
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    TECHNOLOGY = "TECHNOLOGY"
    CONCEPT = "CONCEPT"
    PAPER = "PAPER"
    EVENT = "EVENT"


# ---------------------------------------------------------------------------
# Relationship types
# ---------------------------------------------------------------------------

class RelationshipType(str, Enum):
    WORKS_AT = "WORKS_AT"
    FOUNDED = "FOUNDED"
    DEVELOPED = "DEVELOPED"
    PUBLISHED = "PUBLISHED"
    USES = "USES"
    RELATED_TO = "RELATED_TO"
    PRESENTED_AT = "PRESENTED_AT"
    FUNDED_BY = "FUNDED_BY"
    COLLABORATED_WITH = "COLLABORATED_WITH"
    AUTHORED = "AUTHORED"


# ---------------------------------------------------------------------------
# Relationship constraint: valid (source_type, target_type) pairs
# ---------------------------------------------------------------------------

RELATIONSHIP_CONSTRAINTS: dict[RelationshipType, list[tuple[EntityType, EntityType]]] = {
    RelationshipType.WORKS_AT: [
        (EntityType.PERSON, EntityType.ORGANIZATION),
    ],
    RelationshipType.FOUNDED: [
        (EntityType.PERSON, EntityType.ORGANIZATION),
    ],
    RelationshipType.DEVELOPED: [
        (EntityType.PERSON, EntityType.TECHNOLOGY),
        (EntityType.ORGANIZATION, EntityType.TECHNOLOGY),
    ],
    RelationshipType.PUBLISHED: [
        (EntityType.PERSON, EntityType.PAPER),
        (EntityType.ORGANIZATION, EntityType.PAPER),
    ],
    RelationshipType.USES: [
        (EntityType.ORGANIZATION, EntityType.TECHNOLOGY),
        (EntityType.TECHNOLOGY, EntityType.TECHNOLOGY),
        (EntityType.PERSON, EntityType.TECHNOLOGY),
    ],
    RelationshipType.RELATED_TO: [
        (EntityType.CONCEPT, EntityType.CONCEPT),
        (EntityType.TECHNOLOGY, EntityType.TECHNOLOGY),
        (EntityType.TECHNOLOGY, EntityType.CONCEPT),
        (EntityType.CONCEPT, EntityType.TECHNOLOGY),
        (EntityType.ORGANIZATION, EntityType.ORGANIZATION),
    ],
    RelationshipType.PRESENTED_AT: [
        (EntityType.PAPER, EntityType.EVENT),
        (EntityType.PERSON, EntityType.EVENT),
    ],
    RelationshipType.FUNDED_BY: [
        (EntityType.ORGANIZATION, EntityType.ORGANIZATION),
        (EntityType.PERSON, EntityType.ORGANIZATION),
    ],
    RelationshipType.COLLABORATED_WITH: [
        (EntityType.PERSON, EntityType.PERSON),
        (EntityType.ORGANIZATION, EntityType.ORGANIZATION),
    ],
    RelationshipType.AUTHORED: [
        (EntityType.PERSON, EntityType.PAPER),
    ],
}


# ---------------------------------------------------------------------------
# Property schemas
# ---------------------------------------------------------------------------

class EntityProperties(BaseModel):
    """Properties that can be attached to any entity node."""
    name: str = Field(..., description="Canonical name of the entity")
    entity_type: EntityType = Field(..., description="Type of the entity")
    description: Optional[str] = Field(None, description="Short description")
    aliases: list[str] = Field(default_factory=list, description="Alternative names")
    source: Optional[str] = Field(None, description="Document source of extraction")
    embedding: Optional[list[float]] = Field(None, description="Vector embedding", exclude=True)


class RelationshipProperties(BaseModel):
    """Properties that can be attached to any relationship edge."""
    relationship_type: RelationshipType = Field(..., description="Type of relationship")
    description: Optional[str] = Field(None, description="Description of the relationship")
    weight: float = Field(1.0, ge=0.0, le=1.0, description="Strength / confidence")
    source: Optional[str] = Field(None, description="Document source of extraction")


class TemporalMetadata(BaseModel):
    """Temporal validity metadata for nodes and edges."""
    valid_from: Optional[str] = Field(None, description="ISO date when this fact became valid")
    valid_to: Optional[str] = Field(None, description="ISO date when this fact was invalidated")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Confidence score")
    source: Optional[str] = Field(None, description="Provenance / source document")
    last_updated: Optional[str] = Field(None, description="ISO datetime of last update")


# ---------------------------------------------------------------------------
# Extraction output models (used by LLM structured output)
# ---------------------------------------------------------------------------

class ExtractedEntity(BaseModel):
    """An entity extracted from text."""
    name: str
    entity_type: EntityType
    description: str = ""
    aliases: list[str] = Field(default_factory=list)


class ExtractedRelationship(BaseModel):
    """A relationship extracted from text."""
    source_entity: str = Field(..., description="Name of the source entity")
    target_entity: str = Field(..., description="Name of the target entity")
    relationship_type: RelationshipType
    description: str = ""


class ExtractionResult(BaseModel):
    """Complete extraction output from a text chunk."""
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)


def validate_relationship(
    rel_type: RelationshipType,
    source_type: EntityType,
    target_type: EntityType,
) -> bool:
    """Check whether a relationship is valid given the ontology constraints."""
    allowed = RELATIONSHIP_CONSTRAINTS.get(rel_type, [])
    return (source_type, target_type) in allowed


def get_entity_types_list() -> list[str]:
    """Return entity type names for use in prompts."""
    return [e.value for e in EntityType]


def get_relationship_types_list() -> list[str]:
    """Return relationship type names for use in prompts."""
    return [r.value for r in RelationshipType]
