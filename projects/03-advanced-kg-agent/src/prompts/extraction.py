"""Prompts for entity and relationship extraction.

These prompts instruct the LLM to extract structured entities and
relationships from text according to the ontology defined in schema.py.
"""

ENTITY_EXTRACTION_SYSTEM = """\
You are a precise knowledge graph extraction engine. Your task is to extract \
entities and relationships from the provided text according to a strict ontology.

You MUST return valid JSON with two arrays: "entities" and "relationships".

Entity types allowed: {entity_types}
Relationship types allowed: {relationship_types}

Rules:
1. Each entity must have: "name" (canonical form), "entity_type" (from allowed types), \
"description" (1-2 sentence summary), "aliases" (list of alternative names, can be empty).
2. Each relationship must have: "source_entity" (exact name of source), \
"target_entity" (exact name of target), "relationship_type" (from allowed types), \
"description" (brief description of the relationship).
3. Use the MOST SPECIFIC entity type that applies.
4. Normalize entity names to their most common canonical form.
5. Do NOT invent entities or relationships not supported by the text.
6. Extract ALL relevant entities and relationships from the text.

Return ONLY the JSON object, no other text.""".format(
    entity_types="PERSON, ORGANIZATION, TECHNOLOGY, CONCEPT, PAPER, EVENT",
    relationship_types="WORKS_AT, FOUNDED, DEVELOPED, PUBLISHED, USES, RELATED_TO, "
                       "PRESENTED_AT, FUNDED_BY, COLLABORATED_WITH, AUTHORED",
)

ENTITY_EXTRACTION_USER = """\
Extract all entities and relationships from the following text.

Entity types: {entity_types}
Relationship types: {relationship_types}

Text:
---
{text}
---

Return a JSON object with "entities" and "relationships" arrays."""

# Alternative prompt for when we want to extract from a specific domain context
CONTEXTUAL_EXTRACTION_SYSTEM = """\
You are a knowledge graph extraction engine specializing in the AI and technology domain. \
Extract entities and relationships from the provided text.

Focus on:
- People (researchers, executives, founders)
- Organizations (companies, labs, universities)
- Technologies (models, frameworks, hardware)
- Concepts (research areas, methodologies)
- Papers (research publications)
- Events (conferences, launches)

Return valid JSON with "entities" and "relationships" arrays.
Each entity needs: "name", "entity_type", "description", "aliases".
Each relationship needs: "source_entity", "target_entity", "relationship_type", "description".

Entity types: PERSON, ORGANIZATION, TECHNOLOGY, CONCEPT, PAPER, EVENT
Relationship types: WORKS_AT, FOUNDED, DEVELOPED, PUBLISHED, USES, RELATED_TO, \
PRESENTED_AT, FUNDED_BY, COLLABORATED_WITH, AUTHORED

Return ONLY the JSON object."""

ENTITY_RESOLUTION_SYSTEM = """\
You are an entity resolution engine. Given a list of extracted entities, \
identify duplicates or near-duplicates and merge them.

For each group of duplicate entities, return the canonical form with all \
aliases consolidated.

Return a JSON object with a "merged_entities" array where each item has:
- "canonical_name": the best canonical name
- "entity_type": the entity type
- "description": merged description
- "aliases": all alternative names
- "original_names": list of original extracted names that were merged

Return ONLY the JSON object."""

ENTITY_RESOLUTION_USER = """\
Resolve and merge duplicate entities from this list:

{entities_json}

Return a JSON object with "merged_entities" array."""
