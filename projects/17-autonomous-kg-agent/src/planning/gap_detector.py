"""Gap Detector -- analyzes the knowledge graph for coverage gaps.

Examines entity type distribution, relationship density, connected
components, and topic coverage to identify areas that need improvement.
Returns a prioritized list of gaps.
"""

import sys
from pathlib import Path

from pydantic import BaseModel, Field
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
from shared.llm_clients import chat_completion_structured
from shared.config import get_neo4j_config


# ── Schemas ──────────────────────────────────────────────────────────

class Gap(BaseModel):
    gap_type: str = Field(description="One of: underrepresented_type, isolated_entity, disconnected_component, missing_subtopic, sparse_relationships")
    description: str = Field(description="Description of the gap")
    priority: float = Field(description="Priority score 0-1 (1 = highest priority)")
    suggested_action: str = Field(description="What action would fill this gap")
    related_entities: list[str] = Field(default_factory=list, description="Entities related to this gap")


class GapAnalysisResult(BaseModel):
    gaps: list[Gap] = Field(default_factory=list)


def _get_driver():
    config = get_neo4j_config()
    return GraphDatabase.driver(config["uri"], auth=(config["user"], config["password"]))


def detect_coverage_gaps() -> list[dict]:
    """Analyze the KG and return a prioritized list of coverage gaps.

    Checks:
    - Entity type distribution: are some types underrepresented?
    - Relationship density: are some entities isolated?
    - Connected components: are there disconnected subgraphs?
    - Topic coverage: what subtopics are missing?

    Returns:
        List of gap dicts with priority scores
    """
    driver = _get_driver()
    analysis_data = {}

    with driver.session() as session:
        # 1. Entity type distribution
        result = session.run(
            "MATCH (e:Entity) RETURN e.entity_type AS type, count(*) AS count ORDER BY count DESC"
        )
        type_dist = {r["type"]: r["count"] for r in result}
        analysis_data["type_distribution"] = type_dist

        # 2. Orphan entities (no relationships)
        result = session.run(
            """
            MATCH (e:Entity) WHERE NOT (e)--()
            RETURN e.name AS name, e.entity_type AS type
            LIMIT 20
            """
        )
        orphans = [{"name": r["name"], "type": r["type"]} for r in result]
        analysis_data["orphan_entities"] = orphans

        # 3. Sparse entities (only 1 connection)
        result = session.run(
            """
            MATCH (e:Entity)
            WITH e, size([(e)--() | 1]) AS degree
            WHERE degree = 1
            RETURN e.name AS name, e.entity_type AS type, degree
            ORDER BY e.name
            LIMIT 20
            """
        )
        sparse = [{"name": r["name"], "type": r["type"], "degree": r["degree"]} for r in result]
        analysis_data["sparse_entities"] = sparse

        # 4. Relationship type distribution
        result = session.run(
            "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY count DESC"
        )
        rel_dist = {r["type"]: r["count"] for r in result}
        analysis_data["relationship_distribution"] = rel_dist

        # 5. Check for disconnected components (approximate via weakly connected)
        result = session.run(
            """
            MATCH (e:Entity)
            WITH e, size([(e)--() | 1]) AS degree
            RETURN count(e) AS total,
                   sum(CASE WHEN degree = 0 THEN 1 ELSE 0 END) AS isolated,
                   avg(degree) AS avg_degree
            """
        )
        record = result.single()
        if record:
            analysis_data["connectivity"] = {
                "total_entities": record["total"],
                "isolated_entities": record["isolated"],
                "avg_degree": round(record["avg_degree"] or 0, 2),
            }

        # 6. Get all entity names for topic analysis
        result = session.run("MATCH (e:Entity) RETURN e.name AS name LIMIT 100")
        all_entities = [r["name"] for r in result]
        analysis_data["all_entities"] = all_entities

    driver.close()

    # If the KG is empty, return a single high-priority gap
    if not type_dist:
        return [{
            "gap_type": "empty_graph",
            "description": "The knowledge graph is empty. Initial ingestion needed.",
            "priority": 1.0,
            "suggested_action": "Ingest the seed document and candidate sources.",
            "related_entities": [],
        }]

    # Use LLM to analyze gaps and identify missing subtopics
    analysis_text = f"""Knowledge Graph Analysis:

Entity Type Distribution: {type_dist}
Orphan Entities (no connections): {[o['name'] for o in orphans]}
Sparse Entities (1 connection): {[s['name'] for s in sparse]}
Relationship Types: {rel_dist}
Connectivity: {analysis_data.get('connectivity', {})}
All Entities: {all_entities[:50]}

Domain: Artificial Intelligence (history, key figures, technologies, organizations, concepts)"""

    gap_result = chat_completion_structured(
        prompt=f"""Analyze this knowledge graph and identify coverage gaps.

{analysis_text}

For the domain of Artificial Intelligence, identify:
1. Underrepresented entity types (e.g., too few Person nodes, no Event nodes)
2. Isolated or sparse entities that need more connections
3. Missing subtopics of AI that should be covered
4. Areas where relationship density is too low

Prioritize gaps by importance (1.0 = critical, 0.1 = minor).
Suggest specific actions to fill each gap.""",
        output_schema=GapAnalysisResult,
        system="You are a knowledge graph coverage analyst. Identify the most impactful gaps.",
    )

    # Combine LLM analysis with structural findings
    gaps = [g.model_dump() for g in gap_result.gaps]

    # Add structural orphan gaps if not already covered
    for orphan in orphans:
        already_covered = any(
            orphan["name"] in g.get("description", "")
            for g in gaps
        )
        if not already_covered:
            gaps.append({
                "gap_type": "isolated_entity",
                "description": f"Entity '{orphan['name']}' ({orphan['type']}) has no relationships",
                "priority": 0.6,
                "suggested_action": f"Search for relationships involving {orphan['name']}",
                "related_entities": [orphan["name"]],
            })

    # Sort by priority
    gaps.sort(key=lambda g: g["priority"], reverse=True)
    return gaps


if __name__ == "__main__":
    gaps = detect_coverage_gaps()
    print(f"Found {len(gaps)} gaps:")
    for gap in gaps:
        print(f"  [{gap['priority']:.1f}] {gap['gap_type']}: {gap['description'][:80]}")
