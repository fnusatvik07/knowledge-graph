# Entity Extraction

Entity extraction is the first step in building a knowledge graph from unstructured text. The goal is to identify and categorize the key entities (people, places, concepts, events, etc.) mentioned in your documents.

## Approaches to Entity Extraction

### 1. Traditional NER (Named Entity Recognition)
Pre-trained models like spaCy's NER recognize standard entity types (PERSON, ORG, GPE, DATE). Fast but limited to predefined categories and struggles with domain-specific entities.

```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Einstein worked at Princeton University.")
# [(Einstein, PERSON), (Princeton University, ORG)]
```

### 2. LLM-Based Extraction (Recommended)
Use an LLM with structured output to extract entities with rich descriptions. This is the approach used by GraphRAG and throughout this repository.

```python
from openai import OpenAI

prompt = """Extract all entities from the following text.
For each entity, provide:
- name: The entity's canonical name
- type: One of [PERSON, ORGANIZATION, LOCATION, CONCEPT, EVENT, TECHNOLOGY]
- description: A brief description of the entity based on the text

Text: {text}

Return as JSON array."""
```

### 3. Hybrid Approach
Use traditional NER for a first pass, then LLM for refinement, description generation, and entity resolution.

## LLM-Based Extraction with Structured Output

The most reliable approach uses OpenAI's structured output (JSON mode or function calling) to ensure consistent entity formatting:

```python
from pydantic import BaseModel

class Entity(BaseModel):
    name: str
    type: str
    description: str

class EntityList(BaseModel):
    entities: list[Entity]

response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Extract entities from the text."},
        {"role": "user", "content": text}
    ],
    response_format=EntityList
)
```

## Key Challenges

### Entity Resolution (Coreference)
The same entity may be referred to differently across chunks:
- "Albert Einstein", "Einstein", "the physicist", "he"
- "MIT", "Massachusetts Institute of Technology", "the institute"

**Solution**: Include a `canonical_name` field and use the LLM to normalize references. In multi-chunk pipelines, maintain a running entity list and ask the LLM to merge duplicates.

### Entity Granularity
How specific should entities be? "Machine Learning" vs "Supervised Learning" vs "Random Forests"

**Solution**: Define your ontology (entity types and expected granularity) before extraction. More granular = richer graph but noisier. Less granular = cleaner graph but misses details.

### Chunk Boundary Issues
An entity description may span multiple chunks.

**Solution**: Use overlapping chunks (e.g., 200-character overlap) so entities near boundaries appear in multiple chunks. The graph construction phase handles deduplication.

## Extraction Prompts: Best Practices

1. **Be specific about entity types**: List the exact types you want (don't just say "entities")
2. **Require descriptions**: Descriptions provide context that's valuable during retrieval
3. **Set a domain context**: "You are analyzing scientific papers about climate change" helps the LLM calibrate granularity
4. **Provide examples**: Few-shot prompting dramatically improves extraction quality
5. **Use structured output**: JSON mode or Pydantic models prevent formatting errors

## Cost Optimization

Entity extraction is the most expensive step in KG construction — every text chunk requires an LLM call.

| Strategy | Savings | Trade-off |
|----------|---------|-----------|
| Use `gpt-4o-mini` instead of `gpt-4o` | ~10x cheaper | Slightly lower quality |
| Larger chunks (2000+ chars) | Fewer LLM calls | May miss entities in dense text |
| Batch extraction (entities + relations in one call) | 2x fewer calls | Longer prompts |
| Pre-filter chunks (skip boilerplate) | Variable | May miss context |

## What's Next

Once entities are extracted, the next step is to identify the **relationships** between them — covered in the next section.
