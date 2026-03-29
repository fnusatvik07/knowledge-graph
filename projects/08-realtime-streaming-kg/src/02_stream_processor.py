"""Core streaming processor for the Real-Time Streaming Knowledge Graph.

Watches a directory for new files using `watchfiles`. When a file arrives:
1. Load and chunk the document
2. Extract entities and relationships via LLM (structured output)
3. Upsert to Neo4j with temporal metadata (valid_from timestamps)
4. Perform entity resolution (merge duplicates)
5. Handle fact invalidation (when new info contradicts old)

watchfiles docs: https://watchfiles.helpmanual.io/
LangChain structured output: https://python.langchain.com/docs/how_to/structured_output/
Neo4j Python driver: https://neo4j.com/docs/python-manual/current/

Usage:
    python src/02_stream_processor.py --watch-dir ./data/watch
    python src/02_stream_processor.py --watch-dir ./data/watch --provider openai
"""

import sys
import json
import argparse
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

# Shared imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pydantic import BaseModel, Field
from neo4j import GraphDatabase

from shared.llm_clients import chat_completion_structured, chat_completion, get_embedding

# ── Neo4j connection settings ────────────────────────────────────────────
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

# ── Change log for WebSocket broadcasting ────────────────────────────────
# This file is written by the processor and read by the WebSocket server.
CHANGE_LOG_PATH = Path(__file__).resolve().parent.parent / "output" / "change_log.jsonl"


# ── Pydantic schemas for structured LLM extraction ──────────────────────

class ExtractedEntity(BaseModel):
    """A single entity extracted from text."""
    name: str = Field(description="The canonical name of the entity")
    entity_type: str = Field(description="Type: PERSON, ORGANIZATION, LOCATION, PRODUCT, EVENT, or OTHER")
    description: str = Field(description="Brief description of this entity from the text")


class ExtractedRelationship(BaseModel):
    """A single relationship between two entities."""
    source: str = Field(description="Name of the source entity")
    target: str = Field(description="Name of the target entity")
    relationship: str = Field(description="Type of relationship (e.g., WORKS_AT, ACQUIRED, PARTNERS_WITH)")
    description: str = Field(description="Brief description of this relationship from the text")
    confidence: float = Field(default=0.9, description="Confidence score 0-1")


class ExtractionResult(BaseModel):
    """Complete extraction result from a document."""
    entities: list[ExtractedEntity] = Field(description="List of extracted entities")
    relationships: list[ExtractedRelationship] = Field(description="List of extracted relationships")


class ContradictionCheck(BaseModel):
    """Result of checking for contradictions with existing facts."""
    has_contradiction: bool = Field(description="Whether a contradiction was found")
    old_fact: str = Field(default="", description="The old fact that is contradicted")
    new_fact: str = Field(default="", description="The new fact that replaces it")
    explanation: str = Field(default="", description="Why this is a contradiction")


# ── Helper functions ─────────────────────────────────────────────────────

def normalize_entity_name(name: str) -> str:
    """Normalize an entity name for deduplication."""
    return name.strip().lower().replace("  ", " ")


def make_entity_id(name: str, entity_type: str) -> str:
    """Create a unique entity ID from name and type."""
    normalized = normalize_entity_name(name)
    return hashlib.md5(f"{normalized}::{entity_type}".encode()).hexdigest()


def get_driver():
    """Create a Neo4j driver instance."""
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def log_change(change_type: str, data: dict):
    """Append a change event to the change log for WebSocket broadcasting."""
    CHANGE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": change_type,
        "data": data,
    }
    with open(CHANGE_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Core extraction ─────────────────────────────────────────────────────

def extract_entities_and_relationships(
    text: str, provider: str = "openai", model: str = None
) -> ExtractionResult:
    """Extract entities and relationships from text using LLM structured output.

    Uses LangChain's .with_structured_output() for type-safe extraction.
    Docs: https://python.langchain.com/docs/how_to/structured_output/
    """
    prompt = f"""Analyze the following text and extract all named entities and their relationships.

For entities, identify:
- People (PERSON)
- Companies/Organizations (ORGANIZATION)
- Locations (LOCATION)
- Products/Technologies (PRODUCT)
- Events/Deals (EVENT)

For relationships, identify how entities are connected (e.g., WORKS_AT, CEO_OF, ACQUIRED,
PARTNERS_WITH, LOCATED_IN, ADVISES, CLIENT_OF, FOUNDED, REPLACED_BY).

Text:
{text}

Extract ALL entities and relationships mentioned in the text."""

    result = chat_completion_structured(
        prompt=prompt,
        output_schema=ExtractionResult,
        system="You are an expert entity and relationship extractor for knowledge graphs. "
               "Extract precise, factual information only. Use canonical entity names.",
        provider=provider,
        model=model,
    )
    return result


def check_contradictions(
    new_text: str, existing_facts: list[str], provider: str = "openai", model: str = None
) -> list[ContradictionCheck]:
    """Check if new text contradicts any existing facts in the graph.

    Uses LLM to compare new information against known facts.
    """
    if not existing_facts:
        return []

    facts_str = "\n".join(f"- {f}" for f in existing_facts)
    prompt = f"""Compare the new text against these existing facts and identify ANY contradictions.

Existing facts:
{facts_str}

New text:
{new_text}

For each contradiction found, explain what changed. A contradiction is when the new text
provides information that directly conflicts with or updates an existing fact (e.g., a person
changed roles, a deal value changed, a relationship ended)."""

    class ContradictionResults(BaseModel):
        contradictions: list[ContradictionCheck] = Field(
            description="List of contradictions found (empty if none)"
        )

    result = chat_completion_structured(
        prompt=prompt,
        output_schema=ContradictionResults,
        system="You are a fact-checking expert. Identify contradictions precisely.",
        provider=provider,
        model=model,
    )
    return [c for c in result.contradictions if c.has_contradiction]


# ── Neo4j upsert operations ─────────────────────────────────────────────

def upsert_entity(session, entity: ExtractedEntity, source_id: str, timestamp: str,
                   embedding: list[float] = None):
    """Upsert an entity node into Neo4j with temporal metadata."""
    entity_id = make_entity_id(entity.name, entity.entity_type)

    query = """
    MERGE (e:Entity {entity_id: $entity_id})
    ON CREATE SET
        e.name = $name,
        e.entity_type = $entity_type,
        e.description = $description,
        e.valid_from = $timestamp,
        e.created_from = $source_id,
        e.mention_count = 1
    ON MATCH SET
        e.description = CASE
            WHEN size(e.description) < size($description) THEN $description
            ELSE e.description
        END,
        e.last_updated = $timestamp,
        e.mention_count = e.mention_count + 1
    """
    params = {
        "entity_id": entity_id,
        "name": entity.name,
        "entity_type": entity.entity_type,
        "description": entity.description,
        "timestamp": timestamp,
        "source_id": source_id,
    }

    session.run(query, params)

    # Set embedding if available
    if embedding:
        session.run(
            "MATCH (e:Entity {entity_id: $entity_id}) SET e.embedding = $embedding",
            {"entity_id": entity_id, "embedding": embedding},
        )

    # Add entity type as an additional label
    label = entity.entity_type.replace(" ", "_").upper()
    try:
        session.run(f"MATCH (e:Entity {{entity_id: $entity_id}}) SET e:{label}", {"entity_id": entity_id})
    except Exception:
        pass  # Skip if label is invalid

    log_change("entity_added", {
        "entity_id": entity_id,
        "name": entity.name,
        "type": entity.entity_type,
        "source": source_id,
    })

    return entity_id


def upsert_relationship(session, rel: ExtractedRelationship, source_id: str, timestamp: str):
    """Upsert a relationship between two entities."""
    source_id_hash = make_entity_id(rel.source, "")
    target_id_hash = make_entity_id(rel.target, "")

    # Find source and target by normalized name (fuzzy match)
    query = """
    MATCH (source:Entity)
    WHERE toLower(source.name) = toLower($source_name)
    WITH source LIMIT 1
    MATCH (target:Entity)
    WHERE toLower(target.name) = toLower($target_name)
    WITH source, target LIMIT 1
    MERGE (source)-[r:RELATES_TO {rel_type: $rel_type}]->(target)
    ON CREATE SET
        r.description = $description,
        r.confidence = $confidence,
        r.valid_from = $timestamp,
        r.created_from = $source_id,
        r.invalidated = false
    ON MATCH SET
        r.description = $description,
        r.confidence = $confidence,
        r.last_updated = $timestamp
    RETURN source.name AS src, target.name AS tgt
    """
    result = session.run(query, {
        "source_name": rel.source,
        "target_name": rel.target,
        "rel_type": rel.relationship,
        "description": rel.description,
        "confidence": rel.confidence,
        "timestamp": timestamp,
        "source_id": source_id,
    })

    records = list(result)
    if records:
        log_change("relationship_added", {
            "source": rel.source,
            "target": rel.target,
            "type": rel.relationship,
            "source_doc": source_id,
        })
        return True
    return False


def invalidate_fact(session, old_fact: str, new_fact: str, reason: str, timestamp: str):
    """Mark existing relationships as invalidated when contradicted."""
    # Search for relationships matching the old fact description
    query = """
    MATCH ()-[r:RELATES_TO]->()
    WHERE r.description CONTAINS $search_term AND r.invalidated = false
    SET r.invalidated = true,
        r.valid_until = $timestamp,
        r.invalidation_reason = $reason
    RETURN count(r) AS invalidated_count
    """
    # Extract a search term from the old fact
    search_terms = old_fact.split()[:5]
    search_term = " ".join(search_terms) if search_terms else old_fact

    result = session.run(query, {
        "search_term": search_term,
        "timestamp": timestamp,
        "reason": reason,
    })
    count = result.single()["invalidated_count"]

    if count > 0:
        log_change("fact_invalidated", {
            "old_fact": old_fact,
            "new_fact": new_fact,
            "reason": reason,
            "invalidated_count": count,
        })

    return count


def record_source(session, source_id: str, filename: str, timestamp: str):
    """Record the source document in the graph."""
    session.run("""
        MERGE (s:Source {source_id: $source_id})
        SET s.filename = $filename,
            s.processed_at = $timestamp
    """, {
        "source_id": source_id,
        "filename": filename,
        "timestamp": timestamp,
    })


# ── Main processing pipeline ────────────────────────────────────────────

def process_document(file_path: Path, driver, provider: str = "openai", model: str = None):
    """Process a single document: extract, resolve, upsert, check contradictions."""
    print(f"\n{'='*60}")
    print(f"Processing: {file_path.name}")
    print(f"{'='*60}")

    # Read the document
    text = file_path.read_text(encoding="utf-8")
    timestamp = datetime.now(timezone.utc).isoformat()
    source_id = hashlib.md5(file_path.name.encode()).hexdigest()

    # Step 1: Extract entities and relationships
    print("\n[1/4] Extracting entities and relationships...")
    extraction = extract_entities_and_relationships(text, provider=provider, model=model)
    print(f"  Found {len(extraction.entities)} entities, {len(extraction.relationships)} relationships")

    for e in extraction.entities:
        print(f"    Entity: {e.name} ({e.entity_type})")
    for r in extraction.relationships:
        print(f"    Rel: {r.source} --[{r.relationship}]--> {r.target}")

    # Step 2: Get existing facts for contradiction checking
    print("\n[2/4] Checking for contradictions with existing facts...")
    with driver.session() as session:
        result = session.run("""
            MATCH ()-[r:RELATES_TO]->()
            WHERE r.invalidated = false
            RETURN r.description AS fact
            LIMIT 100
        """)
        existing_facts = [record["fact"] for record in result if record["fact"]]

    contradictions = check_contradictions(text, existing_facts, provider=provider, model=model)
    if contradictions:
        print(f"  Found {len(contradictions)} contradictions!")
        for c in contradictions:
            print(f"    OLD: {c.old_fact}")
            print(f"    NEW: {c.new_fact}")
            print(f"    WHY: {c.explanation}")
    else:
        print("  No contradictions found.")

    # Step 3: Upsert to Neo4j
    print("\n[3/4] Upserting to Neo4j...")
    with driver.session() as session:
        # Record the source document
        record_source(session, source_id, file_path.name, timestamp)

        # Upsert entities with embeddings
        for entity in extraction.entities:
            try:
                embedding = get_embedding(
                    f"{entity.name}: {entity.description}",
                    provider=provider,
                )
            except Exception:
                embedding = None
            upsert_entity(session, entity, source_id, timestamp, embedding)
            print(f"    Upserted entity: {entity.name}")

        # Upsert relationships
        for rel in extraction.relationships:
            success = upsert_relationship(session, rel, source_id, timestamp)
            status = "linked" if success else "skipped (entities not found)"
            print(f"    Relationship: {rel.source} -> {rel.target} [{status}]")

        # Step 4: Handle contradictions / invalidation
        if contradictions:
            print("\n[4/4] Invalidating contradicted facts...")
            for c in contradictions:
                count = invalidate_fact(session, c.old_fact, c.new_fact, c.explanation, timestamp)
                print(f"    Invalidated {count} facts: {c.explanation[:80]}")
        else:
            print("\n[4/4] No facts to invalidate.")

    # Print summary
    print(f"\nDone processing {file_path.name}")
    print_graph_stats(driver)


def print_graph_stats(driver):
    """Print current graph statistics."""
    with driver.session() as session:
        nodes = session.run("MATCH (e:Entity) RETURN count(e) AS c").single()["c"]
        edges = session.run("MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS c").single()["c"]
        sources = session.run("MATCH (s:Source) RETURN count(s) AS c").single()["c"]
        invalidated = session.run(
            "MATCH ()-[r:RELATES_TO]->() WHERE r.invalidated = true RETURN count(r) AS c"
        ).single()["c"]

        print(f"\n  Graph Stats:")
        print(f"    Entities:     {nodes}")
        print(f"    Relationships: {edges} ({invalidated} invalidated)")
        print(f"    Sources:      {sources}")


# ── File watcher ─────────────────────────────────────────────────────────

def watch_directory(watch_dir: Path, driver, provider: str = "openai", model: str = None):
    """Watch a directory for new files and process them as they arrive.

    Uses `watchfiles` for efficient filesystem monitoring.
    Docs: https://watchfiles.helpmanual.io/
    """
    from watchfiles import watch, Change

    watch_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nWatching directory: {watch_dir}")
    print("Drop .txt files into this directory to process them.\n")
    print("Press Ctrl+C to stop.\n")

    # Process any existing files first
    existing = sorted(watch_dir.glob("*.txt"))
    if existing:
        print(f"Found {len(existing)} existing file(s) to process...")
        for f in existing:
            process_document(f, driver, provider=provider, model=model)

    # Watch for new files
    try:
        for changes in watch(watch_dir):
            for change_type, path_str in changes:
                path = Path(path_str)
                if change_type == Change.added and path.suffix == ".txt":
                    process_document(path, driver, provider=provider, model=model)
    except KeyboardInterrupt:
        print("\nStopped watching.")


def main():
    parser = argparse.ArgumentParser(description="Stream Processor for Real-Time KG")
    parser.add_argument(
        "--watch-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data" / "watch"),
        help="Directory to watch for new files",
    )
    parser.add_argument("--provider", type=str, default="openai", help="LLM provider")
    parser.add_argument("--model", type=str, default=None, help="LLM model name")
    args = parser.parse_args()

    driver = get_driver()
    try:
        driver.verify_connectivity()
        print("Connected to Neo4j.")
        watch_directory(Path(args.watch_dir), driver, provider=args.provider, model=args.model)
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Neo4j is running: docker compose up -d")
        sys.exit(1)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
