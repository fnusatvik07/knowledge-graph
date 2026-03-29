"""Extractor Agent -- parses documents, extracts entities and relationships,
and writes them to the shared Neo4j knowledge graph with provenance metadata.

Uses LangGraph StateGraph with structured LLM output for extraction.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import TypedDict

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
from shared.llm_clients import chat_completion_structured, get_chat_model
from shared.config import get_neo4j_config


# ── Pydantic schemas for structured extraction ───────────────────────

class ExtractedEntity(BaseModel):
    name: str = Field(description="The canonical name of the entity")
    entity_type: str = Field(description="Type such as Person, Organization, Technology, Concept, Product")
    description: str = Field(description="Brief description of the entity")


class ExtractedRelationship(BaseModel):
    source: str = Field(description="Name of the source entity")
    target: str = Field(description="Name of the target entity")
    relationship_type: str = Field(description="Type such as DEVELOPED_BY, USES, PART_OF, RELATED_TO")
    description: str = Field(description="Brief description of the relationship")


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)


# ── LangGraph state ──────────────────────────────────────────────────

class ExtractorState(TypedDict):
    document_id: str
    document_text: str
    document_title: str
    extracted: ExtractionResult | None
    entities_written: int
    relationships_written: int
    summary: str


# ── Graph nodes ──────────────────────────────────────────────────────

def extract_entities_and_relationships(state: ExtractorState) -> dict:
    """Use LLM structured output to extract entities and relationships."""
    prompt = f"""Analyze the following document and extract all entities and relationships.

Title: {state["document_title"]}
Text: {state["document_text"]}

Extract:
- All important entities (people, organizations, technologies, concepts, products)
- All relationships between extracted entities
- Use consistent entity names (e.g., "GPT-4" not "GPT4")
- Use UPPER_SNAKE_CASE for relationship types
"""
    result = chat_completion_structured(
        prompt=prompt,
        output_schema=ExtractionResult,
        system="You are an expert knowledge graph engineer. Extract entities and relationships from text with precision.",
    )
    return {"extracted": result}


def write_to_neo4j(state: ExtractorState) -> dict:
    """Write extracted entities and relationships to Neo4j with provenance."""
    config = get_neo4j_config()
    driver = GraphDatabase.driver(config["uri"], auth=(config["user"], config["password"]))

    extracted = state["extracted"]
    now = datetime.now(timezone.utc).isoformat()
    doc_id = state["document_id"]
    entities_written = 0
    rels_written = 0

    with driver.session() as session:
        # Write entities
        for entity in extracted.entities:
            session.run(
                """
                MERGE (e:Entity {name: $name})
                SET e.entity_type = $entity_type,
                    e.description = $description,
                    e.source_doc = $source_doc,
                    e.extracted_by = 'extractor_agent',
                    e.updated_at = $updated_at
                """,
                name=entity.name,
                entity_type=entity.entity_type,
                description=entity.description,
                source_doc=doc_id,
                updated_at=now,
            )
            entities_written += 1

        # Write relationships
        for rel in extracted.relationships:
            session.run(
                f"""
                MATCH (s:Entity {{name: $source}})
                MATCH (t:Entity {{name: $target}})
                MERGE (s)-[r:{rel.relationship_type}]->(t)
                SET r.description = $description,
                    r.source_doc = $source_doc,
                    r.extracted_by = 'extractor_agent',
                    r.created_at = $created_at
                """,
                source=rel.source,
                target=rel.target,
                description=rel.description,
                source_doc=doc_id,
                created_at=now,
            )
            rels_written += 1

    driver.close()
    return {
        "entities_written": entities_written,
        "relationships_written": rels_written,
    }


def generate_summary(state: ExtractorState) -> dict:
    """Generate a summary of the extraction."""
    extracted = state["extracted"]
    entity_types = {}
    for e in extracted.entities:
        entity_types[e.entity_type] = entity_types.get(e.entity_type, 0) + 1

    type_breakdown = ", ".join(f"{t}: {c}" for t, c in sorted(entity_types.items()))
    summary = (
        f"Extracted from '{state['document_title']}' (ID: {state['document_id']}): "
        f"{state['entities_written']} entities ({type_breakdown}), "
        f"{state['relationships_written']} relationships."
    )
    return {"summary": summary}


# ── Build the graph ──────────────────────────────────────────────────

def build_extractor_graph() -> StateGraph:
    """Build and compile the Extractor Agent graph."""
    graph = StateGraph(ExtractorState)

    graph.add_node("extract", extract_entities_and_relationships)
    graph.add_node("write", write_to_neo4j)
    graph.add_node("summarize", generate_summary)

    graph.set_entry_point("extract")
    graph.add_edge("extract", "write")
    graph.add_edge("write", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile()


def run_extractor(document_id: str, title: str, text: str) -> str:
    """Convenience function to run the extractor on a single document."""
    app = build_extractor_graph()
    result = app.invoke({
        "document_id": document_id,
        "document_text": text,
        "document_title": title,
        "extracted": None,
        "entities_written": 0,
        "relationships_written": 0,
        "summary": "",
    })
    return result["summary"]


if __name__ == "__main__":
    summary = run_extractor(
        document_id="test_001",
        title="Test Document",
        text="OpenAI developed GPT-4, a large language model based on the Transformer architecture.",
    )
    print(summary)
