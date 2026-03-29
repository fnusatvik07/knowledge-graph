# KG Versioning and CI/CD

Knowledge graphs are living systems. They grow, evolve, and sometimes need to be rolled back. This section covers strategies for versioning, schema migration, automated pipelines, and quality testing for production knowledge graphs.

## Why KG Lifecycle Management Matters

Unlike static databases, knowledge graphs change in complex ways:

- **New entities** are discovered and added
- **Relationships** are updated or corrected
- **Schema evolves** as understanding of the domain deepens
- **Extraction pipelines** improve, requiring re-processing of source data
- **Contradictions** emerge between new and existing facts
- **Data quality** varies across batches and sources

Without lifecycle management, you end up with a knowledge graph you cannot trust, debug, or reproduce.

## Versioning Strategies

### Strategy 1: Snapshot Versioning

Take full snapshots of the graph at regular intervals or after major changes.

```
v1.0/ -- initial graph (1,000 entities, 3,000 relationships)
v1.1/ -- added medical entities (1,500 entities, 4,200 relationships)
v2.0/ -- schema change: split Person into Researcher and Clinician
v2.1/ -- re-extracted from updated source documents
```

#### Implementation with Neo4j

```python
from neo4j import GraphDatabase
import json
from datetime import datetime

class GraphSnapshotManager:
    def __init__(self, driver):
        self.driver = driver

    def create_snapshot(self, version: str, description: str):
        """Export the entire graph as a JSON snapshot."""
        with self.driver.session() as session:
            # Export all nodes
            nodes = session.run("MATCH (n) RETURN n, labels(n) AS labels, id(n) AS id").data()
            # Export all relationships
            rels = session.run(
                "MATCH (a)-[r]->(b) RETURN id(a) AS source, id(b) AS target, type(r) AS type, properties(r) AS props"
            ).data()

        snapshot = {
            "version": version,
            "description": description,
            "timestamp": datetime.utcnow().isoformat(),
            "nodes": nodes,
            "relationships": rels,
            "stats": {
                "node_count": len(nodes),
                "relationship_count": len(rels),
            }
        }

        path = f"snapshots/{version}.json"
        with open(path, "w") as f:
            json.dump(snapshot, f, indent=2, default=str)

        return path

    def restore_snapshot(self, version: str):
        """Restore the graph from a snapshot."""
        with open(f"snapshots/{version}.json") as f:
            snapshot = json.load(f)

        with self.driver.session() as session:
            # Clear current graph
            session.run("MATCH (n) DETACH DELETE n")

            # Restore nodes
            for node in snapshot["nodes"]:
                labels = ":".join(node["labels"])
                props = node["n"]
                session.run(f"CREATE (n:{labels} $props)", props=props)

            # Restore relationships (simplified -- production needs ID mapping)
            # ...
```

**Pros**: Simple, complete, easy to compare versions
**Cons**: Storage-heavy for large graphs, slow for frequent snapshots

### Strategy 2: Event-Sourced Versioning

Record every change as an event. Reconstruct any version by replaying events.

```python
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

class ChangeType(Enum):
    CREATE_NODE = "create_node"
    UPDATE_NODE = "update_node"
    DELETE_NODE = "delete_node"
    CREATE_RELATIONSHIP = "create_relationship"
    DELETE_RELATIONSHIP = "delete_relationship"
    UPDATE_SCHEMA = "update_schema"

@dataclass
class GraphEvent:
    event_id: str
    timestamp: datetime
    change_type: ChangeType
    data: dict
    source: str  # "pipeline_v2", "manual_correction", "re-extraction"
    user: str = "system"
    metadata: dict = field(default_factory=dict)

class EventStore:
    def __init__(self, log_path: str):
        self.log_path = log_path

    def append(self, event: GraphEvent):
        """Append an event to the log."""
        with open(self.log_path, "a") as f:
            f.write(json.dumps({
                "event_id": event.event_id,
                "timestamp": event.timestamp.isoformat(),
                "change_type": event.change_type.value,
                "data": event.data,
                "source": event.source,
                "user": event.user,
                "metadata": event.metadata,
            }, default=str) + "\n")

    def replay_to(self, target_timestamp: datetime) -> list[GraphEvent]:
        """Get all events up to a point in time."""
        events = []
        with open(self.log_path) as f:
            for line in f:
                event_data = json.loads(line)
                if datetime.fromisoformat(event_data["timestamp"]) <= target_timestamp:
                    events.append(event_data)
        return events

    def get_changes_between(self, start: datetime, end: datetime) -> list[dict]:
        """Get all changes between two timestamps."""
        changes = []
        with open(self.log_path) as f:
            for line in f:
                event = json.loads(line)
                ts = datetime.fromisoformat(event["timestamp"])
                if start <= ts <= end:
                    changes.append(event)
        return changes
```

**Pros**: Full history, any point-in-time reconstruction, audit trail
**Cons**: More complex, replay can be slow for long histories

### Strategy 3: Hybrid (Recommended)

Combine both: event-sourced for change tracking, periodic snapshots for fast restoration.

```
snapshots/
  v1.0.json        -- full snapshot
  v2.0.json        -- full snapshot (after schema change)
events/
  changes.jsonl     -- all individual events
  |-- events between v1.0 and v2.0
  |-- events after v2.0
```

## Schema Migration

As your understanding of the domain evolves, the graph schema changes. Manage this carefully.

### Migration Scripts

```python
class SchemaMigration:
    def __init__(self, version: str, description: str):
        self.version = version
        self.description = description

    def up(self, session):
        """Apply the migration."""
        raise NotImplementedError

    def down(self, session):
        """Reverse the migration."""
        raise NotImplementedError

class Migration_002_SplitPersonNode(SchemaMigration):
    """Split generic Person nodes into Researcher and Clinician."""

    def __init__(self):
        super().__init__("002", "Split Person into Researcher and Clinician")

    def up(self, session):
        # Add Researcher label to persons with publications
        session.run("""
            MATCH (p:Person)-[:AUTHORED]->(pub:Publication)
            SET p:Researcher
            REMOVE p:Person
        """)
        # Add Clinician label to persons with patient relationships
        session.run("""
            MATCH (p:Person)-[:TREATS]->(patient:Patient)
            SET p:Clinician
            REMOVE p:Person
        """)
        # Remaining persons keep their label
        session.run("""
            MATCH (p:Person)
            WHERE NOT p:Researcher AND NOT p:Clinician
            SET p:Researcher
            REMOVE p:Person
        """)

    def down(self, session):
        session.run("MATCH (n:Researcher) SET n:Person REMOVE n:Researcher")
        session.run("MATCH (n:Clinician) SET n:Person REMOVE n:Clinician")

class MigrationRunner:
    def __init__(self, driver):
        self.driver = driver
        self.migrations: list[SchemaMigration] = []

    def register(self, migration: SchemaMigration):
        self.migrations.append(migration)

    def run_pending(self):
        with self.driver.session() as session:
            # Track applied migrations in the graph itself
            applied = session.run(
                "MATCH (m:_Migration) RETURN m.version AS version"
            ).data()
            applied_versions = {m["version"] for m in applied}

            for migration in self.migrations:
                if migration.version not in applied_versions:
                    print(f"Applying migration {migration.version}: {migration.description}")
                    migration.up(session)
                    session.run(
                        "CREATE (m:_Migration {version: $v, applied_at: datetime()})",
                        v=migration.version
                    )
```

## CI/CD Pipeline for Knowledge Graphs

### Pipeline Architecture

```
Source Documents (new/updated)
        |
        v
  [1. Extract] -- LLM extraction with LangChain
        |          (see ../02-kg-construction/04-llm-extraction-patterns.md)
        v
  [2. Validate] -- Schema conformance, quality checks
        |
        v
  [3. Test] -- Compare against baseline, run regression tests
        |
        v
  [4. Stage] -- Load into staging graph database
        |
        v
  [5. Review] -- Human review of diffs (optional)
        |
        v
  [6. Deploy] -- Promote to production graph
        |
        v
  [7. Monitor] -- Track quality metrics over time
```

### GitHub Actions Example

```yaml
name: KG Pipeline
on:
  push:
    paths:
      - 'data/sources/**'
  schedule:
    - cron: '0 2 * * 1'  # Weekly Monday 2 AM

jobs:
  extract-and-validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install langchain-openai pydantic neo4j

      - name: Extract entities from new documents
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: python scripts/extract_entities.py --input data/sources/ --output data/extracted/

      - name: Validate extraction quality
        run: python scripts/validate_extraction.py --input data/extracted/ --schema config/ontology.json

      - name: Run quality checks
        run: python scripts/quality_checks.py --threshold 0.7

      - name: Load into staging Neo4j
        env:
          NEO4J_URI: ${{ secrets.STAGING_NEO4J_URI }}
          NEO4J_PASSWORD: ${{ secrets.STAGING_NEO4J_PASSWORD }}
        run: python scripts/load_graph.py --input data/extracted/ --target staging

      - name: Run regression tests
        run: python scripts/regression_tests.py
```

### Quality Gate Implementation

```python
from langchain_openai import ChatOpenAI

class KGQualityGate:
    """Quality gate for CI/CD pipeline. Blocks deployment if quality drops."""

    def __init__(self, thresholds: dict):
        self.thresholds = thresholds
        # Default thresholds
        self.thresholds.setdefault("min_entity_f1", 0.7)
        self.thresholds.setdefault("min_relationship_f1", 0.6)
        self.thresholds.setdefault("min_ontology_conformance", 0.9)
        self.thresholds.setdefault("max_isolated_node_ratio", 0.15)
        self.thresholds.setdefault("max_quality_regression", 0.05)

    def check(self, current_metrics: dict, baseline_metrics: dict = None) -> dict:
        """Run quality gate checks. Returns pass/fail with details."""
        results = {"passed": True, "checks": []}

        # Absolute threshold checks
        for metric, threshold in self.thresholds.items():
            if metric.startswith("min_"):
                actual_metric = metric[4:]
                if current_metrics.get(actual_metric, 0) < threshold:
                    results["passed"] = False
                    results["checks"].append({
                        "check": metric,
                        "status": "FAIL",
                        "actual": current_metrics.get(actual_metric),
                        "threshold": threshold,
                    })
                else:
                    results["checks"].append({"check": metric, "status": "PASS"})

            elif metric.startswith("max_"):
                actual_metric = metric[4:]
                if current_metrics.get(actual_metric, 0) > threshold:
                    results["passed"] = False
                    results["checks"].append({
                        "check": metric,
                        "status": "FAIL",
                        "actual": current_metrics.get(actual_metric),
                        "threshold": threshold,
                    })
                else:
                    results["checks"].append({"check": metric, "status": "PASS"})

        # Regression check (compare to baseline)
        if baseline_metrics:
            for key in ["entity_f1", "relationship_f1"]:
                if key in current_metrics and key in baseline_metrics:
                    regression = baseline_metrics[key] - current_metrics[key]
                    if regression > self.thresholds["max_quality_regression"]:
                        results["passed"] = False
                        results["checks"].append({
                            "check": f"regression_{key}",
                            "status": "FAIL",
                            "regression": regression,
                        })

        return results
```

> **LangChain ChatOpenAI**: https://python.langchain.com/docs/integrations/chat/openai/

## Change Tracking and Diffs

### Graph Diff

Compare two versions of a graph to see what changed:

```python
def graph_diff(old_snapshot: dict, new_snapshot: dict) -> dict:
    """Compare two graph snapshots and return the differences."""
    old_entities = {n["n"]["name"]: n for n in old_snapshot["nodes"]}
    new_entities = {n["n"]["name"]: n for n in new_snapshot["nodes"]}

    added_entities = set(new_entities.keys()) - set(old_entities.keys())
    removed_entities = set(old_entities.keys()) - set(new_entities.keys())
    common_entities = set(old_entities.keys()) & set(new_entities.keys())

    modified_entities = []
    for name in common_entities:
        if old_entities[name] != new_entities[name]:
            modified_entities.append({
                "name": name,
                "old": old_entities[name],
                "new": new_entities[name],
            })

    return {
        "added_entities": list(added_entities),
        "removed_entities": list(removed_entities),
        "modified_entities": modified_entities,
        "summary": {
            "added": len(added_entities),
            "removed": len(removed_entities),
            "modified": len(modified_entities),
        }
    }
```

## Rollback Strategies

### Snapshot-Based Rollback

```python
def rollback_to_version(snapshot_manager, driver, target_version: str):
    """Rollback the graph to a specific snapshot version."""
    print(f"Rolling back to version {target_version}...")

    # Create a safety snapshot of current state first
    snapshot_manager.create_snapshot(
        f"pre-rollback-{datetime.utcnow().isoformat()}",
        f"Safety snapshot before rollback to {target_version}"
    )

    # Restore the target version
    snapshot_manager.restore_snapshot(target_version)
    print(f"Rollback to {target_version} complete.")
```

### Event-Based Rollback

```python
def rollback_events(event_store, driver, from_timestamp: datetime):
    """Undo all events after a given timestamp by applying inverse operations."""
    events = event_store.get_changes_between(from_timestamp, datetime.utcnow())

    with driver.session() as session:
        # Process events in reverse order
        for event in reversed(events):
            if event["change_type"] == "create_node":
                # Undo: delete the created node
                session.run(
                    "MATCH (n {name: $name}) DETACH DELETE n",
                    name=event["data"]["name"]
                )
            elif event["change_type"] == "delete_node":
                # Undo: recreate the deleted node
                session.run(
                    "CREATE (n:$labels $props)",
                    labels=event["data"]["labels"],
                    props=event["data"]["properties"]
                )
            # ... handle other change types
```

## Testing KG Quality in Pipelines

```python
import pytest

class TestKnowledgeGraphQuality:
    """Tests to run in CI/CD pipeline before deploying graph updates."""

    def test_no_orphan_nodes(self, graph):
        """All nodes should have at least one relationship."""
        result = graph.query("""
            MATCH (n) WHERE NOT (n)--() RETURN count(n) AS orphans
        """)
        orphan_count = result[0]["orphans"]
        assert orphan_count < 10, f"Found {orphan_count} orphan nodes"

    def test_schema_conformance(self, graph, ontology):
        """All node labels and relationship types should be in the ontology."""
        labels = graph.query("CALL db.labels() YIELD label RETURN label")
        valid_labels = set(ontology["valid_node_types"])
        actual_labels = {r["label"] for r in labels}
        invalid = actual_labels - valid_labels
        assert not invalid, f"Invalid labels found: {invalid}"

    def test_no_self_loops(self, graph):
        """No node should have a relationship to itself."""
        result = graph.query("""
            MATCH (n)-[r]->(n) RETURN count(r) AS self_loops
        """)
        assert result[0]["self_loops"] == 0, "Self-loops detected"

    def test_entity_name_quality(self, graph):
        """Entity names should not be empty, too short, or too long."""
        result = graph.query("""
            MATCH (n) WHERE n.name IS NOT NULL
            AND (size(n.name) < 2 OR size(n.name) > 200)
            RETURN count(n) AS bad_names
        """)
        assert result[0]["bad_names"] == 0, "Found entities with invalid name lengths"

    def test_no_duplicate_entities(self, graph):
        """Check for likely duplicate entities (same name, same type)."""
        result = graph.query("""
            MATCH (n)
            WITH labels(n)[0] AS label, toLower(n.name) AS name, count(*) AS cnt
            WHERE cnt > 1
            RETURN label, name, cnt ORDER BY cnt DESC LIMIT 10
        """)
        assert len(result) == 0, f"Found duplicate entities: {result}"
```

## Best Practices

1. **Version everything** -- graph data, schema, extraction pipelines, and ontology
2. **Use event sourcing** for audit trails and fine-grained rollback
3. **Take periodic snapshots** for fast disaster recovery
4. **Automate quality gates** in your CI/CD pipeline
5. **Test schema migrations** on a staging graph before production
6. **Track metrics over time** to detect quality regressions early
7. **Keep extraction pipelines reproducible** -- pin model versions, save prompts

## Next Steps

- [01 - Beyond Text](./01-beyond-text.md) -- multimodal knowledge graphs
- [Graph Quality Metrics](../09-evaluation-metrics/01-graph-quality-metrics.md) -- the metrics used in quality gates
- [LLM Extraction Patterns](../02-kg-construction/04-llm-extraction-patterns.md) -- the extraction pipeline to version
