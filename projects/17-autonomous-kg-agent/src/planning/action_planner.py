"""Action Planner -- decides the next action for the autonomous agent
based on detected gaps, available sources, and action history.

Uses LLM reasoning to prioritize and select the most impactful action.
"""

import sys
import json
from pathlib import Path

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
from shared.llm_clients import chat_completion_structured


# ── Schemas ──────────────────────────────────────────────────────────

class PlannedAction(BaseModel):
    action_type: str = Field(
        description="One of: ingest_source, enrich_entity, validate_and_dedup, web_search"
    )
    reasoning: str = Field(description="Why this action was chosen")
    expected_impact: str = Field(description="Expected improvement to the KG")
    priority: float = Field(description="Priority 0-1")
    # Fields for specific action types
    source_id: str = Field(default="", description="Source ID for ingest_source action")
    entity_name: str = Field(default="", description="Entity name for enrich_entity action")
    query: str = Field(default="", description="Search query for web_search action")


class ActionPlan(BaseModel):
    selected_action: PlannedAction
    alternative_actions: list[PlannedAction] = Field(default_factory=list)


def plan_next_action(
    gaps: list[dict],
    kg_stats: dict,
    pending_sources: list[dict],
    action_history: list[dict],
) -> dict:
    """Plan the next action for the autonomous agent.

    Decision logic:
    - If entity coverage is low -> ingest more documents
    - If relationships are sparse -> enrich existing entities
    - If quality is low -> validate and dedup
    - If staleness detected -> web search for fresh info

    Args:
        gaps: Detected coverage gaps with priorities
        kg_stats: Current KG statistics
        pending_sources: Available sources not yet processed
        action_history: Previous actions taken

    Returns:
        dict describing the action to take
    """
    # Build context for LLM
    total_entities = kg_stats.get("total_entities", 0)
    quality = kg_stats.get("quality_score", 0.0)
    coverage = kg_stats.get("coverage_score", 0.0)

    top_gaps = gaps[:5] if gaps else []
    gap_descriptions = "\n".join(
        f"  [{g['priority']:.1f}] {g['gap_type']}: {g['description']}"
        for g in top_gaps
    ) or "  No gaps detected."

    recent_actions = action_history[-5:] if action_history else []
    history_text = "\n".join(
        f"  [{a.get('iteration', '?')}] {a['action_type']}: {a.get('result', '')[:100]}"
        for a in recent_actions
    ) or "  No previous actions."

    pending_text = "\n".join(
        f"  [{s.get('id', '?')}] {s.get('title', 'Unknown')} (status: {s.get('status', 'pending')})"
        for s in pending_sources[:5]
    ) or "  No pending sources."

    plan = chat_completion_structured(
        prompt=f"""You are an autonomous agent managing a knowledge graph about Artificial Intelligence.
Decide the single most impactful action to take next.

Current KG State:
- Total entities: {total_entities}
- Quality score: {quality:.2f}
- Coverage score: {coverage:.2f}

Top Gaps:
{gap_descriptions}

Available Sources (not yet ingested):
{pending_text}

Recent Actions:
{history_text}

Choose ONE action:
1. ingest_source: Pick a pending source to ingest (set source_id)
2. enrich_entity: Search the web to enrich a sparse entity (set entity_name)
3. validate_and_dedup: Clean up duplicates and inconsistencies
4. web_search: Search for information on a missing topic (set query)

Rules:
- Don't repeat the same action type 3 times in a row
- Prioritize ingestion when there are unprocessed sources
- Prioritize enrichment when entities have few connections
- Prioritize validation after several ingestions
- Prioritize web search when specific subtopics are missing""",
        output_schema=ActionPlan,
        system="You are an autonomous KG improvement planner. Choose the highest-impact action.",
    )

    # Build the action dict
    action = plan.selected_action.model_dump()

    # Attach the full source data if ingesting
    if action["action_type"] == "ingest_source" and action.get("source_id"):
        matching_source = next(
            (s for s in pending_sources if s.get("id") == action["source_id"]),
            None,
        )
        if matching_source:
            action["source"] = matching_source
        elif pending_sources:
            # Fallback to first pending source
            action["source"] = pending_sources[0]
            action["source_id"] = pending_sources[0].get("id", "")
    elif action["action_type"] == "ingest_source" and pending_sources:
        # LLM didn't specify an ID, pick the first one
        action["source"] = pending_sources[0]
        action["source_id"] = pending_sources[0].get("id", "")

    return action


if __name__ == "__main__":
    action = plan_next_action(
        gaps=[{
            "gap_type": "missing_subtopic",
            "description": "No coverage of deep learning",
            "priority": 0.9,
            "suggested_action": "Search for deep learning information",
            "related_entities": [],
        }],
        kg_stats={"total_entities": 5, "quality_score": 0.3, "coverage_score": 0.2},
        pending_sources=[{"id": "src_001", "title": "Neural Networks", "status": "pending"}],
        action_history=[],
    )
    print(f"Action: {action['action_type']}")
    print(f"Reasoning: {action['reasoning']}")
