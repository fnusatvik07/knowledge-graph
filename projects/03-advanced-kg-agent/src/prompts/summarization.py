"""Prompts for entity summarization and context compression.

These prompts help condense retrieved information into compact summaries
that fit within LLM context windows while preserving key facts.
"""

ENTITY_SUMMARY_SYSTEM = """\
You are a knowledge graph summarization engine. Given information about an \
entity from a knowledge graph (properties, relationships, and context), \
produce a concise summary.

Rules:
1. Include the entity's key attributes and role.
2. Mention the most important relationships.
3. Keep the summary to 2-4 sentences.
4. Focus on factual, objective information.
5. Use clear, precise language."""

ENTITY_SUMMARY_USER = """\
Summarize this entity from the knowledge graph:

Entity: {entity_name} [{entity_type}]
Description: {description}

Relationships:
{relationships}

Additional context:
{context}

Provide a concise summary (2-4 sentences)."""

CONTEXT_COMPRESSION_SYSTEM = """\
You are a context compression engine. Given a large block of retrieved context \
from a knowledge graph, compress it while preserving all information relevant \
to the query.

Rules:
1. Remove redundant or duplicate information.
2. Preserve all facts relevant to the query.
3. Maintain entity names and relationship types exactly.
4. Use concise bullet-point format.
5. Target at most 50% of the original length."""

CONTEXT_COMPRESSION_USER = """\
Compress the following knowledge graph context, preserving information relevant \
to the query.

Query: {query}

Context to compress:
{context}

Provide a compressed version as concise bullet points."""

SUBGRAPH_SUMMARY_SYSTEM = """\
You are a subgraph summarization engine. Given a set of nodes and edges from a \
knowledge graph, produce a narrative summary that describes the key entities and \
how they relate to each other.

Rules:
1. Identify the central entities and themes.
2. Describe the most important connections.
3. Group related facts together.
4. Keep the summary structured and readable.
5. Use 3-6 sentences."""

SUBGRAPH_SUMMARY_USER = """\
Summarize this subgraph from the knowledge graph:

Entities:
{entities}

Relationships:
{relationships}

Provide a narrative summary of the key entities and their connections."""

MULTI_SOURCE_MERGE_SYSTEM = """\
You are an information merging engine. Given multiple pieces of information about \
the same entity from different sources, merge them into a single coherent description.

Rules:
1. Combine non-conflicting information.
2. For conflicting information, note the discrepancy.
3. Prefer more recent or more authoritative sources.
4. Preserve source attribution where relevant.
5. Keep the merged description concise."""

MULTI_SOURCE_MERGE_USER = """\
Merge these descriptions of the same entity from different sources:

Entity: {entity_name}

Source descriptions:
{source_descriptions}

Provide a single merged description."""
