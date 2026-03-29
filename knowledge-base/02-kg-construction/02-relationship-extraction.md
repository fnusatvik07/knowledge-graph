# Relationship Extraction

After extracting entities, the next step is identifying how they relate to each other. Relationships are the *edges* that give a knowledge graph its power — without them, you just have a list of entities.

## What is a Relationship?

A relationship is a directed connection between two entities:

```
(Source Entity) --[RELATIONSHIP_TYPE]--> (Target Entity)
```

Examples:
```
(Einstein) --[DEVELOPED]--> (Theory of Relativity)
(Einstein) --[WORKED_AT]--> (Princeton University)
(Theory of Relativity) --[FIELD_OF]--> (Physics)
(Princeton University) --[LOCATED_IN]--> (New Jersey)
```

## LLM-Based Relationship Extraction

### Combined Extraction (Recommended)
In practice, entities and relationships are often extracted together in a single LLM call to reduce cost and improve coherence:

```python
from pydantic import BaseModel

class Entity(BaseModel):
    name: str
    type: str
    description: str

class Relationship(BaseModel):
    source: str          # Source entity name
    target: str          # Target entity name
    relation_type: str   # e.g., DEVELOPED, WORKS_AT
    description: str     # Natural language description
    strength: float      # 0.0 to 1.0 confidence

class ExtractionResult(BaseModel):
    entities: list[Entity]
    relationships: list[Relationship]
```

### Extraction Prompt Pattern

```
Given the following text, extract:
1. All entities (people, organizations, concepts, technologies, etc.)
2. All relationships between those entities

For each relationship, provide:
- source: The name of the source entity
- target: The name of the target entity
- relation_type: A concise, uppercase label (e.g., DEVELOPED, WORKS_AT, PART_OF)
- description: A sentence describing the relationship in context
- strength: Confidence from 0.0 to 1.0

Important:
- Only extract relationships explicitly supported by the text
- Use the exact entity names from your entity list
- Each relationship should be directional (source acts on target)
```

## Open vs Closed Information Extraction

### Open Information Extraction (Open IE)
Relation types are not predefined — the LLM generates whatever labels it finds appropriate.

**Pros**: Captures all relationships, no schema needed
**Cons**: Inconsistent labels ("works_at" vs "employed_by" vs "AFFILIATED_WITH"), harder to query

### Closed Information Extraction (Closed IE)
Relation types are constrained to a predefined set.

**Pros**: Consistent, queryable, schema-aligned
**Cons**: May miss unexpected relationship types

```python
ALLOWED_RELATIONS = [
    "DEVELOPED", "WORKS_AT", "LOCATED_IN", "PART_OF",
    "FOUNDED", "COLLABORATED_WITH", "PUBLISHED", "RELATED_TO"
]

prompt = f"""Only use these relationship types: {ALLOWED_RELATIONS}
If a relationship doesn't fit any type, use RELATED_TO as a fallback."""
```

### Recommendation
Start with **open IE** for exploratory analysis, then move to **closed IE** with a defined ontology for production systems.

## Handling Multi-Chunk Documents

When processing a document chunk by chunk, relationships may span chunk boundaries:

```
Chunk 1: "Einstein was born in Ulm, Germany in 1879."
Chunk 2: "He later moved to Switzerland where he worked at the patent office."
```

The pronoun "He" in Chunk 2 refers to Einstein in Chunk 1.

### Strategies:
1. **Overlapping chunks**: Include 200+ characters of overlap so cross-boundary relationships appear in at least one chunk
2. **Entity carry-forward**: Pass the list of extracted entities from previous chunks as context
3. **Two-pass extraction**: First extract all entities across the document, then extract relationships with full entity context

## Relationship Quality

### What Makes a Good Relationship?
- **Explicit**: Directly supported by the text (not inferred speculation)
- **Specific**: "DEVELOPED" is better than "RELATED_TO"
- **Directional**: Clear which entity is the subject and which is the object
- **Described**: The description field adds context lost in the label

### Common Pitfalls
- **Over-extraction**: Creating relationships between every pair of co-occurring entities
- **Vague labels**: Using "RELATED_TO" for everything
- **Missing directionality**: "Einstein and Relativity" — who developed what?
- **Hallucinated relationships**: The LLM inventing connections not in the text

## What's Next

With entities and relationships extracted, you need an **ontology** to organize them — covered next.
