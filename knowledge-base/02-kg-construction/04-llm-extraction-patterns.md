# LLM Extraction Patterns with LangChain

Large language models are the most flexible and powerful tool for extracting structured knowledge from unstructured text. This section covers production-ready patterns using LangChain's unified LLM interface, focusing on reliable structured output, few-shot prompting, batch processing, and error handling.

> **LangChain Structured Output Docs**: https://python.langchain.com/docs/how_to/structured_output/

## Why LangChain for Extraction?

Our shared LLM layer uses LangChain (`langchain-openai`, `langchain-anthropic`, `langchain-ollama`), which provides:

- **Unified API** across providers (OpenAI, Anthropic, Ollama)
- **`.with_structured_output()`** for guaranteed schema-conformant responses
- **Built-in retry logic** and error handling
- **Batch processing** with rate limiting

## Structured Output with Pydantic Schemas

The foundation of reliable extraction is defining your output schema with Pydantic models, then using LangChain's `.with_structured_output()` to enforce it.

### Defining Extraction Schemas

```python
from pydantic import BaseModel, Field
from typing import Optional

class Entity(BaseModel):
    """A real-world entity extracted from text."""
    name: str = Field(description="Canonical name of the entity")
    entity_type: str = Field(description="One of: PERSON, ORGANIZATION, LOCATION, CONCEPT, EVENT, TECHNOLOGY")
    description: str = Field(description="Brief description based on the source text")
    confidence: float = Field(description="Confidence score between 0 and 1", ge=0, le=1)

class Relationship(BaseModel):
    """A relationship between two entities."""
    source: str = Field(description="Name of the source entity")
    target: str = Field(description="Name of the target entity")
    relation_type: str = Field(description="Type of relationship, e.g. WORKS_AT, LOCATED_IN")
    description: str = Field(description="Description of the relationship in context")

class ExtractionResult(BaseModel):
    """Complete extraction output from a text chunk."""
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
```

> **Pydantic with LangChain**: https://python.langchain.com/docs/concepts/structured_outputs/

### Using `.with_structured_output()`

This is the recommended approach for all LLM-based extraction. It works across OpenAI, Anthropic, and other providers.

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# OpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm = llm.with_structured_output(ExtractionResult)

# Anthropic
llm_claude = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
structured_llm_claude = llm_claude.with_structured_output(ExtractionResult)

# Ollama (local)
from langchain_ollama import ChatOllama
llm_local = ChatOllama(model="llama3.1")
structured_llm_local = llm_local.with_structured_output(ExtractionResult)
```

> **LangChain ChatOpenAI**: https://python.langchain.com/docs/integrations/chat/openai/
> **LangChain ChatAnthropic**: https://python.langchain.com/docs/integrations/chat/anthropic/

### Basic Extraction Call

```python
from langchain_core.messages import SystemMessage, HumanMessage

system_prompt = SystemMessage(content="""You are a knowledge graph extraction engine.
Extract all entities and relationships from the provided text.
Be thorough but precise. Only extract information explicitly stated or strongly implied.""")

text = "Marie Curie conducted groundbreaking research on radioactivity at the University of Paris. She was awarded the Nobel Prize in Physics in 1903."

result = structured_llm.invoke([
    system_prompt,
    HumanMessage(content=f"Extract entities and relationships from:\n\n{text}")
])

# result is a validated ExtractionResult instance
for entity in result.entities:
    print(f"  {entity.name} ({entity.entity_type}): {entity.description}")
for rel in result.relationships:
    print(f"  {rel.source} --[{rel.relation_type}]--> {rel.target}")
```

## Few-Shot Prompting for Extraction

Few-shot examples dramatically improve extraction consistency, especially for domain-specific entity types and relationship patterns.

```python
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

# Define examples as input/output pairs
examples = [
    {
        "input": "Dr. Sarah Chen leads the AI research division at Acme Corp in San Francisco.",
        "output": """{
  "entities": [
    {"name": "Sarah Chen", "entity_type": "PERSON", "description": "AI researcher, leads the AI research division", "confidence": 0.95},
    {"name": "Acme Corp", "entity_type": "ORGANIZATION", "description": "Company with an AI research division", "confidence": 0.95},
    {"name": "San Francisco", "entity_type": "LOCATION", "description": "City where Acme Corp is located", "confidence": 0.90}
  ],
  "relationships": [
    {"source": "Sarah Chen", "target": "Acme Corp", "relation_type": "LEADS_DIVISION_AT", "description": "leads the AI research division"},
    {"source": "Acme Corp", "target": "San Francisco", "relation_type": "LOCATED_IN", "description": "located in San Francisco"}
  ]
}"""
    },
    {
        "input": "TensorFlow, developed by Google Brain, was released in 2015 as an open-source library.",
        "output": """{
  "entities": [
    {"name": "TensorFlow", "entity_type": "TECHNOLOGY", "description": "Open-source machine learning library", "confidence": 0.99},
    {"name": "Google Brain", "entity_type": "ORGANIZATION", "description": "AI research team that developed TensorFlow", "confidence": 0.95}
  ],
  "relationships": [
    {"source": "Google Brain", "target": "TensorFlow", "relation_type": "DEVELOPED", "description": "developed TensorFlow, released in 2015"}
  ]
}"""
    }
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}")
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract entities and relationships from text into a knowledge graph."),
    few_shot_prompt,
    ("human", "{input}")
])

chain = final_prompt | structured_llm
result = chain.invoke({"input": "New text to extract from..."})
```

> **Few-shot prompting in LangChain**: https://python.langchain.com/docs/how_to/few_shot_examples_chat/

## Batch Processing Patterns

Real knowledge graph construction involves processing hundreds or thousands of text chunks. LangChain provides built-in batch processing with rate limiting.

### Simple Batch Processing

```python
texts = ["chunk 1...", "chunk 2...", "chunk 3..."]

messages_batch = [
    [system_prompt, HumanMessage(content=f"Extract entities and relationships:\n\n{t}")]
    for t in texts
]

# Process in batch with concurrency control
results = structured_llm.batch(
    messages_batch,
    config={"max_concurrency": 5}  # limit parallel requests
)
```

### Batch with Progress Tracking

```python
from langchain_core.runnables import RunnableConfig
from tqdm import tqdm

def extract_with_progress(texts: list[str], batch_size: int = 10) -> list[ExtractionResult]:
    all_results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting"):
        batch = texts[i:i + batch_size]
        messages = [
            [system_prompt, HumanMessage(content=f"Extract:\n\n{t}")]
            for t in batch
        ]
        results = structured_llm.batch(messages, config={"max_concurrency": 5})
        all_results.extend(results)
    return all_results
```

### Async Batch Processing

```python
import asyncio

async def extract_async(texts: list[str]) -> list[ExtractionResult]:
    messages_batch = [
        [system_prompt, HumanMessage(content=f"Extract:\n\n{t}")]
        for t in texts
    ]
    results = await structured_llm.abatch(
        messages_batch,
        config={"max_concurrency": 10}
    )
    return results

# Run
results = asyncio.run(extract_async(my_texts))
```

> **LangChain batch operations**: https://python.langchain.com/docs/how_to/batch/

## Error Handling and Retry Logic

Production extraction pipelines must handle API failures, malformed outputs, and rate limits gracefully.

### Built-in Retry with `.with_retry()`

```python
from langchain_core.runnables import RunnableConfig

# Add automatic retries to the structured LLM
robust_llm = structured_llm.with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True,  # exponential backoff with jitter
)

result = robust_llm.invoke([system_prompt, HumanMessage(content=text)])
```

### Custom Fallback Chain

```python
from langchain_core.runnables import RunnableLambda

# Primary: GPT-4o, Fallback: Claude
primary = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(ExtractionResult)
fallback = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0).with_structured_output(ExtractionResult)

robust_extractor = primary.with_fallbacks([fallback])
```

### Handling Extraction Failures per Chunk

```python
from pydantic import ValidationError
import logging

logger = logging.getLogger(__name__)

def safe_extract(text: str, llm=structured_llm) -> ExtractionResult | None:
    """Extract with error handling. Returns None on failure."""
    try:
        result = llm.with_retry(stop_after_attempt=3).invoke([
            system_prompt,
            HumanMessage(content=f"Extract entities and relationships:\n\n{text}")
        ])
        return result
    except ValidationError as e:
        logger.warning(f"Schema validation failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return None

# Process all chunks, collecting successes and logging failures
results = []
failed_chunks = []
for i, chunk in enumerate(text_chunks):
    result = safe_extract(chunk)
    if result:
        results.append(result)
    else:
        failed_chunks.append(i)

print(f"Extracted {len(results)}/{len(text_chunks)} chunks. Failed: {failed_chunks}")
```

## Merging Extraction Results

After extracting from multiple chunks, merge and deduplicate entities.

```python
from collections import defaultdict

def merge_extractions(results: list[ExtractionResult]) -> ExtractionResult:
    """Merge multiple extraction results, deduplicating entities."""
    entity_map: dict[str, Entity] = {}
    all_relationships: list[Relationship] = []

    for result in results:
        for entity in result.entities:
            key = entity.name.lower().strip()
            if key in entity_map:
                # Keep the one with higher confidence
                if entity.confidence > entity_map[key].confidence:
                    entity_map[key] = entity
            else:
                entity_map[key] = entity

        all_relationships.extend(result.relationships)

    # Deduplicate relationships
    seen_rels = set()
    unique_rels = []
    for rel in all_relationships:
        key = (rel.source.lower(), rel.target.lower(), rel.relation_type)
        if key not in seen_rels:
            seen_rels.add(key)
            unique_rels.append(rel)

    return ExtractionResult(
        entities=list(entity_map.values()),
        relationships=unique_rels
    )
```

## Best Practices

1. **Always use structured output** -- never parse free-text LLM responses for entity extraction
2. **Set temperature to 0** for deterministic extraction
3. **Include few-shot examples** from your actual domain
4. **Process in batches** with concurrency limits to avoid rate limiting
5. **Implement fallbacks** across providers for reliability
6. **Validate and deduplicate** results after extraction
7. **Log failures** with the source text for later reprocessing

## Next Steps

- [05 - Graph Building Pipeline](./05-graph-building-pipeline.md) -- how to take extraction results and build the graph
- [Ontology Design](./03-ontology-design.md) -- designing the schema your extractions conform to
