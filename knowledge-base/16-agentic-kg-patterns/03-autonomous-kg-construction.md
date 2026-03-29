# Autonomous KG Construction

## Overview

An autonomous KG construction agent does not wait for humans to feed it documents.
It actively identifies gaps in the knowledge graph, selects relevant sources,
extracts and validates facts, and continuously improves the graph's coverage
and quality.

This is the difference between a pipeline (human triggers extraction) and an agent
(system identifies what needs to be done and does it).

---

## Gap Detection

### What Gaps Look Like in a KG

A knowledge graph has gaps when:

1. **Missing entity types.** The graph has many PERSON and ORGANIZATION entities but
   almost no TECHNOLOGY entities. The schema expects all four types.

2. **Sparse neighborhoods.** An entity has very few relationships compared to similar
   entities. "Geoffrey Hinton" has 15 edges; "Yann LeCun" has only 2. LeCun's
   neighborhood is suspiciously sparse.

3. **Disconnected components.** A group of entities is not connected to the main graph.
   This often indicates missing "bridge" relationships.

4. **Missing inverse relationships.** "Alice WORKS_AT Acme" exists but "Acme EMPLOYS
   Alice" does not. Depending on the schema, this may be a gap.

5. **Temporal gaps.** No entities have been updated in the last 30 days in a domain
   that changes weekly.

### Detection Queries

```cypher
// Find entity types with fewer than expected instances
MATCH (e:Entity)
WITH e.type AS type, count(*) AS cnt
ORDER BY cnt ASC
RETURN type, cnt

// Find entities with sparse neighborhoods (fewer edges than average)
MATCH (e:Entity)
WITH e, size((e)--()) AS degree
WITH avg(degree) AS avg_degree
MATCH (e:Entity)
WHERE size((e)--()) < avg_degree * 0.3
RETURN e.name, e.type, size((e)--()) AS degree
ORDER BY degree ASC

// Find disconnected components
CALL gds.wcc.stream('kg_graph')
YIELD nodeId, componentId
WITH componentId, count(*) AS size
WHERE size < 5  // Small disconnected clusters
RETURN componentId, size
ORDER BY size ASC

// Find stale entities (not updated in N days)
MATCH (e:Entity)
WHERE e.updated_at < datetime() - duration({days: 30})
RETURN e.name, e.type, e.updated_at
ORDER BY e.updated_at ASC
```

### Gap Detection Agent

```python
class GapDetector:
    """Analyzes KG to identify areas needing improvement."""

    def __init__(self, kg_client, llm):
        self.kg = kg_client
        self.llm = llm

    def detect_all_gaps(self) -> list:
        """Run all gap detection strategies and return prioritized gaps."""
        gaps = []
        gaps.extend(self.detect_type_imbalance())
        gaps.extend(self.detect_sparse_entities())
        gaps.extend(self.detect_disconnected_components())
        gaps.extend(self.detect_stale_entities())

        # Prioritize: disconnected components > sparse entities > type gaps > stale
        gaps.sort(key=lambda g: g["priority"])
        return gaps

    def detect_type_imbalance(self) -> list:
        type_counts = self.kg.query("MATCH (e:Entity) RETURN e.type, count(*)")
        expected_types = {"PERSON", "ORGANIZATION", "TECHNOLOGY", "CONCEPT"}
        found_types = {r["e.type"] for r in type_counts}
        missing = expected_types - found_types

        gaps = []
        for t in missing:
            gaps.append({
                "type": "missing_entity_type",
                "entity_type": t,
                "priority": 1,
                "action": f"Find and ingest documents about {t} entities",
            })
        return gaps
```

---

## Relevance Scoring

### Should This Document Be Ingested?

Not every document is relevant to the KG's domain. An autonomous agent must
decide whether a candidate document is worth the cost of extraction.

### Scoring Dimensions

1. **Domain relevance.** Does this document cover topics already in the KG or
   topics the KG should cover? Compare document embedding to KG entity embeddings.

2. **Novelty.** Does this document contain information not already in the KG?
   If 90% of the entities are already known, the document adds little value.

3. **Quality.** Is this document from a reliable source? Does it contain structured
   factual claims or just opinion?

4. **Gap coverage.** Does this document fill an identified gap? If the KG is missing
   TECHNOLOGY entities and this document describes a new framework, it scores high.

### Implementation

```python
class RelevanceScorer:
    """Score candidate documents for KG ingestion relevance."""

    def __init__(self, kg_client, llm):
        self.kg = kg_client
        self.llm = llm
        self.gaps = []  # Populated by GapDetector

    def score(self, document: str) -> dict:
        """Score a document on multiple dimensions. Returns 0-1 scores."""

        # Domain relevance: cosine similarity to KG entity descriptions
        doc_embedding = self.llm.embed(document[:2000])
        kg_embeddings = self.kg.get_entity_embeddings()
        domain_score = max_cosine_similarity(doc_embedding, kg_embeddings)

        # Novelty: what fraction of extractable entities are new?
        candidate_entities = self.llm.quick_extract(document[:2000])
        existing = sum(1 for e in candidate_entities
                       if self.kg.entity_exists(e["name"]))
        novelty_score = 1.0 - (existing / max(len(candidate_entities), 1))

        # Gap coverage: does this fill known gaps?
        gap_score = self.score_gap_coverage(candidate_entities)

        # Combined score (weighted)
        combined = (
            0.3 * domain_score +
            0.4 * novelty_score +
            0.3 * gap_score
        )

        return {
            "domain_relevance": domain_score,
            "novelty": novelty_score,
            "gap_coverage": gap_score,
            "combined": combined,
            "recommendation": "ingest" if combined > 0.5 else "skip",
        }

    def score_gap_coverage(self, candidate_entities: list) -> float:
        """Score how well candidate entities fill known gaps."""
        if not self.gaps:
            return 0.5  # Neutral if no gaps identified

        filled = 0
        for gap in self.gaps:
            if gap["type"] == "missing_entity_type":
                if any(e.get("type") == gap["entity_type"]
                       for e in candidate_entities):
                    filled += 1
        return filled / max(len(self.gaps), 1)
```

---

## Self-Healing

### Types of KG Errors

1. **Extraction errors.** The LLM misidentified an entity type ("Apple" classified
   as FRUIT instead of ORGANIZATION in a tech document).

2. **Duplicate entities.** "Geoffrey Hinton", "G. Hinton", "Prof. Hinton" are all
   the same person but appear as separate nodes.

3. **Contradictions.** Two sources disagree: one says "Alice works at Acme", another
   says "Alice works at Beta Corp." Both cannot be true simultaneously (unless
   she changed jobs).

4. **Orphan nodes.** Entities with no relationships — often a sign of incomplete
   extraction.

5. **Schema violations.** A PERSON entity has no NAME property, or a relationship
   is missing its confidence score.

### Self-Healing Agent

```python
class SelfHealingAgent:
    """Detect and fix KG quality issues autonomously."""

    def __init__(self, kg_client, llm):
        self.kg = kg_client
        self.llm = llm

    def heal(self) -> dict:
        """Run all healing strategies. Returns summary of fixes."""
        report = {
            "duplicates_merged": self.merge_duplicates(),
            "contradictions_resolved": self.resolve_contradictions(),
            "orphans_connected": self.connect_orphans(),
            "types_corrected": self.correct_types(),
        }
        return report

    def merge_duplicates(self) -> int:
        """Find and merge duplicate entities."""
        # Strategy: entities with similar names AND similar neighborhoods
        candidates = self.kg.query("""
            MATCH (a:Entity), (b:Entity)
            WHERE a.id < b.id
            AND apoc.text.levenshteinSimilarity(a.name, b.name) > 0.8
            RETURN a, b
        """)

        merged_count = 0
        for a, b in candidates:
            # Confirm with LLM
            is_duplicate = self.llm.classify(
                f"Are these the same entity?\n"
                f"A: {a['name']} ({a['type']}): {a.get('description', '')}\n"
                f"B: {b['name']} ({b['type']}): {b.get('description', '')}\n",
                options=["same", "different"]
            )
            if is_duplicate == "same":
                self.kg.merge_entities(a["id"], b["id"])
                merged_count += 1

        return merged_count

    def resolve_contradictions(self) -> int:
        """Find and resolve contradictory facts."""
        # Find entities with conflicting relationship targets
        conflicts = self.kg.query("""
            MATCH (e:Entity)-[r1:WORKS_AT]->(o1:Entity),
                  (e)-[r2:WORKS_AT]->(o2:Entity)
            WHERE o1 <> o2
            AND r1.valid_to IS NULL AND r2.valid_to IS NULL
            RETURN e, r1, o1, r2, o2
        """)

        resolved_count = 0
        for e, r1, o1, r2, o2 in conflicts:
            # Use recency: more recent relationship wins
            if r1["created_at"] > r2["created_at"]:
                self.kg.expire_relationship(r2["id"])
            else:
                self.kg.expire_relationship(r1["id"])
            resolved_count += 1

        return resolved_count

    def connect_orphans(self) -> int:
        """Find orphan nodes and attempt to connect them."""
        orphans = self.kg.query("""
            MATCH (e:Entity)
            WHERE NOT (e)--()
            RETURN e
        """)

        connected = 0
        for orphan in orphans:
            # Ask LLM to suggest relationships
            similar = self.kg.vector_search(orphan["description"], top_k=5)
            if similar:
                suggestions = self.llm.suggest_relationships(orphan, similar)
                for suggestion in suggestions:
                    if suggestion["confidence"] > 0.7:
                        self.kg.create_relationship(
                            orphan["id"],
                            suggestion["target_id"],
                            suggestion["type"],
                            confidence=suggestion["confidence"],
                        )
                        connected += 1

        return connected

    def correct_types(self) -> int:
        """Verify and correct entity type classifications."""
        # Sample entities and verify with LLM
        sample = self.kg.query("""
            MATCH (e:Entity)
            WHERE e.confidence < 0.7
            RETURN e
            LIMIT 50
        """)

        corrected = 0
        for entity in sample:
            # Get neighborhood context for better type inference
            neighbors = self.kg.get_neighbors(entity["id"], max_hops=1)
            context = format_neighborhood(entity, neighbors)

            correct_type = self.llm.classify(
                f"Based on this entity and its relationships, what is its type?\n"
                f"{context}",
                options=["PERSON", "ORGANIZATION", "TECHNOLOGY", "CONCEPT"]
            )

            if correct_type != entity["type"]:
                self.kg.update_entity_type(entity["id"], correct_type)
                corrected += 1

        return corrected
```

---

## Staleness Detection

### Why Staleness Matters

Knowledge graphs represent facts about the world, and the world changes. An entity
that was accurate six months ago may be outdated today. People change jobs, companies
merge, technologies evolve.

### Staleness Scoring

```python
def compute_staleness_score(entity: dict, domain_config: dict) -> float:
    """Score entity staleness from 0 (fresh) to 1 (stale).

    Args:
        entity: Entity with updated_at timestamp
        domain_config: Per-type staleness thresholds
            e.g., {"PERSON": 90, "TECHNOLOGY": 30, "CONCEPT": 365}
            (days before considered stale)
    """
    days_since_update = (datetime.now() - entity["updated_at"]).days
    threshold = domain_config.get(entity["type"], 180)

    if days_since_update <= threshold * 0.5:
        return 0.0  # Fresh
    elif days_since_update <= threshold:
        return (days_since_update - threshold * 0.5) / (threshold * 0.5)  # Linear decay
    else:
        return 1.0  # Stale
```

### Re-ingestion Triggers

```python
def check_and_trigger_reingestion(kg_client, threshold: float = 0.8):
    """Flag stale entities for re-ingestion."""
    stale_entities = kg_client.query("""
        MATCH (e:Entity)
        WHERE e.updated_at < datetime() - duration({days: 30})
        RETURN e.id, e.name, e.type, e.updated_at, e.source
        ORDER BY e.updated_at ASC
    """)

    for entity in stale_entities:
        score = compute_staleness_score(entity, DOMAIN_CONFIG)
        if score >= threshold:
            # Trigger re-ingestion from original source
            reingestion_queue.put({
                "entity_id": entity["id"],
                "entity_name": entity["name"],
                "source": entity["source"],
                "staleness_score": score,
                "action": "re_extract",
            })
```

---

## Exploration Strategies

### Breadth-First vs. Depth-First

An autonomous agent must decide: should it explore many topics shallowly or
go deep on one topic?

**Breadth-first exploration:**
- Cover many entity types and domains.
- Good for initial KG population.
- Finds many connections between diverse topics.
- Risk: shallow coverage, many gaps within each topic.

**Depth-first exploration:**
- Go deep on one topic before moving to the next.
- Good for expert-level KGs.
- Produces dense, well-connected subgraphs.
- Risk: narrow coverage, missing cross-domain connections.

### Hybrid Strategy

```python
class ExplorationStrategy:
    """Decide what to explore next based on KG state."""

    def __init__(self, kg_client, mode: str = "hybrid"):
        self.kg = kg_client
        self.mode = mode

    def next_topic(self) -> str:
        if self.mode == "breadth":
            return self.least_covered_type()
        elif self.mode == "depth":
            return self.sparsest_neighborhood()
        else:  # hybrid
            # Alternate: breadth when coverage is uneven, depth otherwise
            type_counts = self.kg.get_type_distribution()
            imbalance = max(type_counts.values()) / max(min(type_counts.values()), 1)
            if imbalance > 3:
                return self.least_covered_type()
            else:
                return self.sparsest_neighborhood()

    def least_covered_type(self) -> str:
        """Find the entity type with the fewest instances."""
        counts = self.kg.query("""
            MATCH (e:Entity)
            RETURN e.type AS type, count(*) AS cnt
            ORDER BY cnt ASC
            LIMIT 1
        """)
        return counts[0]["type"] if counts else "CONCEPT"

    def sparsest_neighborhood(self) -> str:
        """Find the entity with the sparsest neighborhood (needs more context)."""
        sparse = self.kg.query("""
            MATCH (e:Entity)
            WITH e, size((e)--()) AS degree
            WHERE degree > 0  // Not orphan (those are a different problem)
            ORDER BY degree ASC
            LIMIT 1
            RETURN e.name
        """)
        return sparse[0]["e.name"] if sparse else None
```

---

## Quality Gates

### Extraction Pipeline with Gates

Every fact goes through a pipeline before entering the canonical KG:

```
Document --> [Extract] --> [Validate] --> [Human Review?] --> [Commit to KG]
                |               |                |
                v               v                v
            Raw facts      Validated facts   Approved facts
           (staging)        (pending)         (canonical)
```

### Implementation with LangGraph

```python
from langgraph.graph import StateGraph, END

class QualityState(TypedDict):
    document: str
    extracted_facts: list
    validated_facts: list
    rejected_facts: list
    requires_human_review: bool

def extract(state):
    facts = extractor.extract(state["document"])
    return {"extracted_facts": facts}

def validate(state):
    validated = []
    rejected = []
    for fact in state["extracted_facts"]:
        # Check against existing KG for consistency
        is_consistent = validator.check_consistency(fact, kg_client)
        # Check confidence threshold
        if fact["confidence"] > 0.8 and is_consistent:
            validated.append(fact)
        elif fact["confidence"] > 0.5:
            # Medium confidence: needs human review
            validated.append({**fact, "needs_review": True})
        else:
            rejected.append(fact)

    needs_human = any(f.get("needs_review") for f in validated)
    return {
        "validated_facts": validated,
        "rejected_facts": rejected,
        "requires_human_review": needs_human,
    }

def route_after_validation(state) -> str:
    if state["requires_human_review"]:
        return "human_review"
    return "commit"

def commit_to_kg(state):
    """Write validated facts to the canonical KG."""
    for fact in state["validated_facts"]:
        if not fact.get("needs_review") or fact.get("human_approved"):
            kg_client.upsert(fact)
    return state

workflow = StateGraph(QualityState)
workflow.add_node("extract", extract)
workflow.add_node("validate", validate)
workflow.add_node("human_review", human_review_node)
workflow.add_node("commit", commit_to_kg)

workflow.set_entry_point("extract")
workflow.add_edge("extract", "validate")
workflow.add_conditional_edges("validate", route_after_validation, {
    "human_review": "human_review",
    "commit": "commit",
})
workflow.add_edge("human_review", "commit")
workflow.add_edge("commit", END)
```

---

## Dynamic Tool Selection

### Agent Decides How to Fill Gaps

An autonomous agent has multiple tools at its disposal. For each identified gap,
it must decide the most efficient way to fill it:

| Tool | Best For | Cost | Latency |
|------|----------|------|---------|
| Local KG search | Finding existing but unlinked facts | Free | <100ms |
| Global KG search | Finding related communities/summaries | Free | <500ms |
| Document re-extraction | Getting more facts from existing docs | $ | 2-5s |
| Web search | Finding new information | $$ | 3-10s |
| LLM reasoning | Inferring missing relationships | $$$ | 1-5s |

### LangGraph Implementation with Conditional Edges

```python
def decide_tool(state) -> str:
    """Agent decides which tool to use based on the gap type."""
    gap = state["current_gap"]

    if gap["type"] == "missing_relationship":
        # First try: local KG search (maybe the relationship exists but unlinked)
        existing = kg_client.search_relationships(
            gap["source_entity"], gap["target_entity"]
        )
        if existing:
            return "link_existing"

    if gap["type"] == "sparse_entity":
        # Check if we have unprocessed documents about this entity
        unprocessed = document_store.find_unprocessed(gap["entity_name"])
        if unprocessed:
            return "re_extract"

    if gap["type"] == "stale_entity":
        return "web_search"

    # Default: ask LLM to infer
    return "llm_reason"

workflow.add_conditional_edges("analyze_gap", decide_tool, {
    "link_existing": "link_existing_node",
    "re_extract": "re_extract_node",
    "web_search": "web_search_node",
    "llm_reason": "llm_reason_node",
})
```

---

## Key Takeaways

1. **Gap detection is the starting point.** An autonomous agent must know what it
   does not know. Regularly query the KG for missing types, sparse neighborhoods,
   disconnected components, and stale entities.

2. **Relevance scoring saves cost.** Not every document is worth extracting. Score
   candidates on domain relevance, novelty, and gap coverage before committing
   to extraction.

3. **Self-healing is continuous.** Duplicates, contradictions, and type errors
   accumulate over time. Run self-healing checks on a schedule.

4. **Quality gates prevent garbage.** Every extracted fact should pass through
   validation before entering the canonical KG. Use confidence thresholds
   and human review for borderline cases.

5. **Dynamic tool selection is what makes it an agent.** A pipeline always does
   the same steps. An agent chooses the most efficient tool for each situation
   based on the current KG state.
