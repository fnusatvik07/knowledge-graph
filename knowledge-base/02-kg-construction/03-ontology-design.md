# Ontology Design for Knowledge Graphs

An **ontology** defines the schema of your knowledge graph: what types of entities exist, what types of relationships connect them, and what constraints govern the graph structure.

## Why Ontology Matters

Without an ontology, your knowledge graph is a free-for-all:
- Entity types are inconsistent ("person" vs "Person" vs "PERSON" vs "human")
- Relationship types proliferate ("works_at" vs "employed_by" vs "AFFILIATED_WITH")
- The graph becomes hard to query and reason over

Research shows that **ontology-grounded approaches achieve the highest RAG performance with minimal hallucination** compared to schema-free approaches (NeurIPS 2025).

## Ontology Components

### 1. Entity Types
The categories of nodes in your graph:

```python
ENTITY_TYPES = [
    "PERSON",           # Individual people
    "ORGANIZATION",     # Companies, universities, agencies
    "LOCATION",         # Cities, countries, regions
    "CONCEPT",          # Abstract ideas, theories, methods
    "TECHNOLOGY",       # Tools, frameworks, languages
    "EVENT",            # Conferences, incidents, milestones
    "PUBLICATION",      # Papers, books, reports
]
```

### 2. Relationship Types
The categories of edges, with source/target type constraints:

```python
RELATIONSHIP_TYPES = {
    "WORKS_AT":        {"source": "PERSON", "target": "ORGANIZATION"},
    "LOCATED_IN":      {"source": ["ORGANIZATION", "EVENT"], "target": "LOCATION"},
    "DEVELOPED":       {"source": "PERSON", "target": ["CONCEPT", "TECHNOLOGY"]},
    "PUBLISHED":       {"source": "PERSON", "target": "PUBLICATION"},
    "PART_OF":         {"source": "CONCEPT", "target": "CONCEPT"},
    "USES":            {"source": "TECHNOLOGY", "target": "TECHNOLOGY"},
    "FOUNDED":         {"source": "PERSON", "target": "ORGANIZATION"},
    "COLLABORATED_WITH": {"source": "PERSON", "target": "PERSON"},
}
```

### 3. Property Schemas
What attributes each entity/relationship type should have:

```python
ENTITY_PROPERTIES = {
    "PERSON": ["full_name", "role", "affiliation"],
    "ORGANIZATION": ["name", "type", "founded_year"],
    "CONCEPT": ["name", "field", "description"],
}
```

## Design Approaches

### Schema-Free (Open)
No predefined types — let the LLM decide.

**Use when**: Exploring a new domain, building a prototype, you don't know what's in the data.

**Trade-off**: Maximum flexibility, minimum consistency. Good for discovery, bad for production.

### Schema-First (Closed)
Define all entity and relationship types before extraction.

**Use when**: You understand the domain, building for production, need consistent queries.

**Trade-off**: May miss unexpected entity types. Best quality when the schema matches the domain.

### Schema-Guided (Hybrid)
Provide a base schema but allow the LLM to suggest new types.

```python
prompt = """Use these entity types when applicable: {ENTITY_TYPES}
If you encounter an entity that doesn't fit any type, assign it the most appropriate type
and note it as a suggested addition to the schema."""
```

**Use when**: You have a rough understanding of the domain but want to discover new patterns.

## Domain-Specific Ontology Examples

### Biomedical
```
Entity Types: Gene, Protein, Disease, Drug, Pathway, Cell_Type
Relationships: CAUSES, TREATS, INHIBITS, ACTIVATES, ASSOCIATED_WITH, EXPRESSED_IN
```

### Software Engineering
```
Entity Types: Function, Class, Module, API, Library, Pattern, Bug
Relationships: CALLS, INHERITS, IMPORTS, DEPENDS_ON, IMPLEMENTS, FIXES
```

### Scientific Literature
```
Entity Types: Author, Paper, Method, Dataset, Result, Institution
Relationships: AUTHORED, CITES, USES_METHOD, TRAINED_ON, ACHIEVES, AFFILIATED_WITH
```

## Best Practices

1. **Start small**: Begin with 5-7 entity types and 8-10 relationship types. Expand as needed.
2. **Use uppercase for types**: `PERSON`, `WORKS_AT` — consistent with graph database conventions.
3. **Include a catch-all**: `RELATED_TO` relationship and `OTHER` entity type for edge cases.
4. **Document your ontology**: Keep it in a single file that's referenced by extraction prompts.
5. **Iterate**: Review extracted graphs, find inconsistencies, refine the ontology.
6. **Validate**: After extraction, check that all entities and relationships conform to the ontology.

## Ontology in the Extraction Pipeline

```python
ONTOLOGY = {
    "entity_types": ["PERSON", "ORGANIZATION", "CONCEPT", "TECHNOLOGY"],
    "relationship_types": ["DEVELOPED", "WORKS_AT", "PART_OF", "USES"],
}

system_prompt = f"""You are a knowledge graph extraction system.
Use ONLY these entity types: {ONTOLOGY['entity_types']}
Use ONLY these relationship types: {ONTOLOGY['relationship_types']}
If an entity or relationship doesn't fit, use the closest match."""
```

This ontology-constrained extraction produces cleaner, more queryable graphs — which is critical for the retrieval phase of Graph RAG.
