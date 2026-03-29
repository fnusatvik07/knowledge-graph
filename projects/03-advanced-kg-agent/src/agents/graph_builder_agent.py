"""Graph Builder Agent: extracts entities and relationships from documents.

Uses a LangGraph agent with LLM-based structured extraction to populate
the Neo4j knowledge graph with temporal metadata.
"""
# LangChain docs: https://python.langchain.com/docs/integrations/chat/

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from langgraph.graph import StateGraph, END

from shared.llm_clients import chat_completion_structured, get_embedding
from shared.document_loader import chunk_text
from src.graph.neo4j_store import Neo4jStore
from src.graph.temporal_layer import TemporalLayer
from src.graph.schema import (
    ExtractionResult,
    EntityType,
    get_entity_types_list,
    get_relationship_types_list,
)
from src.prompts.extraction import (
    ENTITY_EXTRACTION_SYSTEM,
    ENTITY_EXTRACTION_USER,
)


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class GraphBuilderState(TypedDict):
    """State for the graph builder agent."""
    documents: list[dict[str, Any]]       # Input documents
    chunks: list[dict[str, Any]]          # Chunked text with metadata
    extractions: list[ExtractionResult]   # Extracted entities and relationships
    nodes_created: int
    relationships_created: int
    errors: list[str]
    status: str


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def chunk_documents(state: GraphBuilderState) -> dict[str, Any]:
    """Split documents into chunks for extraction."""
    chunks: list[dict[str, Any]] = []
    for doc in state["documents"]:
        text_chunks = chunk_text(doc["content"], chunk_size=1500, overlap=200)
        for i, chunk in enumerate(text_chunks):
            chunks.append({
                "text": chunk,
                "source": doc.get("source", "unknown"),
                "title": doc.get("title", "unknown"),
                "chunk_index": i,
            })
    return {"chunks": chunks, "status": "chunked"}


def extract_entities_and_relationships(state: GraphBuilderState) -> dict[str, Any]:
    """Extract entities and relationships from each chunk using the LLM.

    Uses chat_completion_structured() with ExtractionResult as the Pydantic
    output schema for type-safe extraction.
    LangChain structured output: https://python.langchain.com/docs/how_to/structured_output/
    """
    extractions: list[ExtractionResult] = []
    errors: list[str] = []

    entity_types = ", ".join(get_entity_types_list())
    relationship_types = ", ".join(get_relationship_types_list())

    for chunk in state["chunks"]:
        try:
            prompt = ENTITY_EXTRACTION_USER.format(
                text=chunk["text"],
                entity_types=entity_types,
                relationship_types=relationship_types,
            )
            extraction = chat_completion_structured(
                prompt=prompt,
                output_schema=ExtractionResult,
                system=ENTITY_EXTRACTION_SYSTEM,
                temperature=0.0,
            )
            # Attach source metadata to each entity
            for entity in extraction.entities:
                entity.description = entity.description or ""
            extractions.append(extraction)
        except Exception as e:
            errors.append(f"Extraction error on chunk: {e}")

    return {
        "extractions": extractions,
        "errors": state.get("errors", []) + errors,
        "status": "extracted",
    }


def write_to_graph(state: GraphBuilderState) -> dict[str, Any]:
    """Write extracted entities and relationships to Neo4j."""
    # Neo4j store and temporal layer are injected via closure
    store: Neo4jStore = state["_store"]  # type: ignore[typeddict-item]
    temporal = TemporalLayer(store)

    nodes_created = 0
    rels_created = 0
    errors: list[str] = []
    seen_entities: dict[str, EntityType] = {}

    for extraction in state["extractions"]:
        # Create entity nodes
        for entity in extraction.entities:
            try:
                # Determine the label (use entity_type as label)
                label = entity.entity_type.value

                # Generate embedding for the entity
                embed_text = f"{entity.name}: {entity.description}"
                embedding = get_embedding(embed_text)

                properties = {
                    "name": entity.name,
                    "entity_type": entity.entity_type.value,
                    "description": entity.description,
                    "aliases": json.dumps(entity.aliases),
                    "embedding": embedding,
                }

                store.create_node(label, properties, merge=True)
                # Also create with generic Entity label for vector index
                store.create_node("Entity", properties, merge=True)

                temporal.add_node_temporal_metadata(
                    label="Entity",
                    name=entity.name,
                    source=state["chunks"][0].get("source", "unknown") if state["chunks"] else "unknown",
                    confidence=0.9,
                )

                seen_entities[entity.name] = entity.entity_type
                nodes_created += 1
            except Exception as e:
                errors.append(f"Node creation error for {entity.name}: {e}")

        # Create relationships
        for rel in extraction.relationships:
            try:
                source_type = seen_entities.get(rel.source_entity)
                target_type = seen_entities.get(rel.target_entity)
                if not source_type or not target_type:
                    continue

                props = {"description": rel.description}
                store.create_relationship(
                    source_name=rel.source_entity,
                    source_label=source_type.value,
                    target_name=rel.target_entity,
                    target_label=target_type.value,
                    rel_type=rel.relationship_type.value,
                    properties=props,
                )

                temporal.add_relationship_temporal_metadata(
                    source_name=rel.source_entity,
                    target_name=rel.target_entity,
                    rel_type=rel.relationship_type.value,
                    confidence=0.85,
                )

                rels_created += 1
            except Exception as e:
                errors.append(f"Relationship error {rel.source_entity}->{rel.target_entity}: {e}")

    return {
        "nodes_created": nodes_created,
        "relationships_created": rels_created,
        "errors": state.get("errors", []) + errors,
        "status": "completed",
    }


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph_builder_workflow(store: Neo4jStore) -> StateGraph:
    """Build the LangGraph workflow for graph construction."""

    def _write_to_graph(state: GraphBuilderState) -> dict[str, Any]:
        state["_store"] = store  # type: ignore[typeddict-item]
        return write_to_graph(state)

    workflow = StateGraph(GraphBuilderState)
    workflow.add_node("chunk", chunk_documents)
    workflow.add_node("extract", extract_entities_and_relationships)
    workflow.add_node("write", _write_to_graph)

    workflow.set_entry_point("chunk")
    workflow.add_edge("chunk", "extract")
    workflow.add_edge("extract", "write")
    workflow.add_edge("write", END)

    return workflow


class GraphBuilderAgent:
    """High-level interface for document-to-graph ingestion."""

    def __init__(self, store: Neo4jStore) -> None:
        self.store = store
        self.workflow = build_graph_builder_workflow(store)
        self.app = self.workflow.compile()

    def ingest(self, documents: list[dict[str, Any]]) -> dict[str, Any]:
        """Ingest documents into the knowledge graph.

        Args:
            documents: List of dicts with 'title', 'content', 'source'.

        Returns:
            Final state with stats on created nodes/relationships.
        """
        initial_state: GraphBuilderState = {
            "documents": documents,
            "chunks": [],
            "extractions": [],
            "nodes_created": 0,
            "relationships_created": 0,
            "errors": [],
            "status": "started",
        }
        result = self.app.invoke(initial_state)
        return {
            "nodes_created": result.get("nodes_created", 0),
            "relationships_created": result.get("relationships_created", 0),
            "errors": result.get("errors", []),
            "status": result.get("status", "unknown"),
        }
