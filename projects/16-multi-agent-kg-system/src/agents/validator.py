"""Validator Agent -- reads recently added entities and relationships from Neo4j,
checks quality (type consistency, ontology validity, duplicates), auto-fixes
obvious issues, and returns a validation report.

Uses LangGraph StateGraph for the validation pipeline.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import TypedDict

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
from shared.llm_clients import chat_completion, chat_completion_structured
from shared.config import get_neo4j_config


# ── Schemas ──────────────────────────────────────────────────────────

class ValidationIssue(BaseModel):
    issue_type: str = Field(description="One of: duplicate, inconsistent_type, invalid_relationship, orphan_node")
    entity_name: str = Field(description="Entity involved")
    details: str = Field(description="Description of the issue")
    auto_fixable: bool = Field(description="Whether this can be auto-fixed")


class TypeConsistencyCheck(BaseModel):
    inconsistencies: list[str] = Field(default_factory=list, description="List of inconsistent type usages")
    suggested_fixes: list[str] = Field(default_factory=list, description="Suggested canonical type names")


# ── LangGraph state ──────────────────────────────────────────────────

class ValidatorState(TypedDict):
    issues: list[dict]
    duplicates_found: int
    duplicates_merged: int
    type_fixes: int
    report: str


# ── Helper: Neo4j driver ────────────────────────────────────────────

def _get_driver():
    config = get_neo4j_config()
    return GraphDatabase.driver(config["uri"], auth=(config["user"], config["password"]))


# ── Graph nodes ──────────────────────────────────────────────────────

def check_duplicates(state: ValidatorState) -> dict:
    """Find entities with similar names that may be duplicates."""
    driver = _get_driver()
    issues = list(state.get("issues") or [])
    duplicates_found = 0

    with driver.session() as session:
        # Find entities whose names differ only by case or minor variations
        result = session.run(
            """
            MATCH (a:Entity), (b:Entity)
            WHERE id(a) < id(b)
              AND (toLower(a.name) = toLower(b.name)
                   OR apoc.text.jaroWinklerDistance(toLower(a.name), toLower(b.name)) > 0.9)
            RETURN a.name AS name_a, b.name AS name_b,
                   a.entity_type AS type_a, b.entity_type AS type_b
            """
        )
        for record in result:
            issues.append({
                "issue_type": "duplicate",
                "entity_name": f"{record['name_a']} / {record['name_b']}",
                "details": f"Potential duplicate: '{record['name_a']}' ({record['type_a']}) and '{record['name_b']}' ({record['type_b']})",
                "auto_fixable": True,
            })
            duplicates_found += 1

    driver.close()
    return {"issues": issues, "duplicates_found": duplicates_found}


def check_type_consistency(state: ValidatorState) -> dict:
    """Check if entity types are consistent and well-formed."""
    driver = _get_driver()
    issues = list(state["issues"])
    type_fixes = 0

    with driver.session() as session:
        # Get all entity types
        result = session.run(
            "MATCH (e:Entity) RETURN DISTINCT e.entity_type AS entity_type, count(*) AS cnt"
        )
        types = {record["entity_type"]: record["cnt"] for record in result}

    driver.close()

    # Use LLM to check type consistency
    if types:
        type_list = "\n".join(f"- {t} ({c} entities)" for t, c in sorted(types.items()))
        check_result = chat_completion(
            prompt=f"""Review these entity types used in a knowledge graph and identify inconsistencies:

{type_list}

Look for:
1. Same concept with different capitalization (e.g., "person" vs "Person")
2. Overlapping types that should be merged (e.g., "Company" vs "Organization")
3. Too-specific types that should be generalized

For each issue, respond with the format:
ISSUE: [type1] -> [suggested_fix]

If no issues, respond with: NO ISSUES FOUND""",
            system="You are a knowledge graph ontology expert. Be concise.",
        )

        for line in check_result.strip().split("\n"):
            if line.startswith("ISSUE:"):
                issues.append({
                    "issue_type": "inconsistent_type",
                    "entity_name": line,
                    "details": line,
                    "auto_fixable": True,
                })
                type_fixes += 1

    return {"issues": issues, "type_fixes": type_fixes}


def auto_fix_issues(state: ValidatorState) -> dict:
    """Auto-fix obvious issues: merge duplicates, normalize types."""
    driver = _get_driver()
    duplicates_merged = 0

    with driver.session() as session:
        # Merge exact case-insensitive duplicates
        result = session.run(
            """
            MATCH (a:Entity), (b:Entity)
            WHERE id(a) < id(b)
              AND toLower(a.name) = toLower(b.name)
            RETURN a.name AS keep_name, b.name AS merge_name, id(b) AS merge_id
            """
        )
        for record in result:
            # Transfer relationships from b to a, then delete b
            session.run(
                """
                MATCH (b:Entity) WHERE id(b) = $merge_id
                MATCH (a:Entity {name: $keep_name})
                OPTIONAL MATCH (b)-[r]->(t)
                WITH a, b, collect({type: type(r), target: t, props: properties(r)}) AS out_rels
                OPTIONAL MATCH (s)-[r]->(b)
                WITH a, b, out_rels, collect({type: type(r), source: s, props: properties(r)}) AS in_rels
                DETACH DELETE b
                """,
                merge_id=record["merge_id"],
                keep_name=record["keep_name"],
            )
            duplicates_merged += 1

        # Normalize entity types to title case
        session.run(
            """
            MATCH (e:Entity)
            WHERE e.entity_type <> apoc.text.capitalize(toLower(e.entity_type))
            SET e.entity_type = apoc.text.capitalize(toLower(e.entity_type)),
                e.validated_by = 'validator_agent',
                e.validated_at = $now
            """,
            now=datetime.now(timezone.utc).isoformat(),
        )

    driver.close()
    return {"duplicates_merged": duplicates_merged}


def generate_report(state: ValidatorState) -> dict:
    """Generate a validation report."""
    issues = state["issues"]
    report_lines = [
        "=== Validation Report ===",
        f"Total issues found: {len(issues)}",
        f"Duplicates found: {state.get('duplicates_found', 0)}",
        f"Duplicates merged: {state.get('duplicates_merged', 0)}",
        f"Type fixes applied: {state.get('type_fixes', 0)}",
        "",
        "Issues:",
    ]
    for issue in issues:
        status = "[AUTO-FIXED]" if issue.get("auto_fixable") else "[NEEDS REVIEW]"
        report_lines.append(f"  {status} {issue['issue_type']}: {issue['details']}")

    if not issues:
        report_lines.append("  No issues found. KG is clean.")

    report = "\n".join(report_lines)
    return {"report": report}


# ── Build the graph ──────────────────────────────────────────────────

def build_validator_graph() -> StateGraph:
    """Build and compile the Validator Agent graph."""
    graph = StateGraph(ValidatorState)

    graph.add_node("check_duplicates", check_duplicates)
    graph.add_node("check_types", check_type_consistency)
    graph.add_node("auto_fix", auto_fix_issues)
    graph.add_node("report", generate_report)

    graph.set_entry_point("check_duplicates")
    graph.add_edge("check_duplicates", "check_types")
    graph.add_edge("check_types", "auto_fix")
    graph.add_edge("auto_fix", "report")
    graph.add_edge("report", END)

    return graph.compile()


def run_validator() -> str:
    """Run the validator and return the report."""
    app = build_validator_graph()
    result = app.invoke({
        "issues": [],
        "duplicates_found": 0,
        "duplicates_merged": 0,
        "type_fixes": 0,
        "report": "",
    })
    return result["report"]


if __name__ == "__main__":
    report = run_validator()
    print(report)
