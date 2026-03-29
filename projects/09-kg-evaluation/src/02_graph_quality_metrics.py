"""Evaluate the structural quality of a knowledge graph stored in Neo4j.

Metrics computed:
- Node coverage: entities in documents vs entities in graph
- Edge density: ratio of edges to possible edges
- Connected component analysis
- Orphan node detection (nodes with no relationships)
- Ontology conformance: % of entities/relations matching expected schema
- Centrality distribution analysis (degree, betweenness)

Neo4j graph algorithms: https://neo4j.com/docs/graph-data-science/current/
LangChain Neo4j: https://python.langchain.com/docs/integrations/graphs/neo4j_cypher/

Usage:
    python src/02_graph_quality_metrics.py
    python src/02_graph_quality_metrics.py --uri bolt://localhost:7687
"""

import sys
import json
import argparse
from pathlib import Path
from collections import Counter

# Shared imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from neo4j import GraphDatabase

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
GOLD_STANDARD_PATH = PROJECT_DIR / "data" / "gold_standard.json"
OUTPUT_DIR = PROJECT_DIR / "output"

# ── Expected ontology (entity types and relationship types) ──────────────
EXPECTED_ENTITY_TYPES = {"PERSON", "ORGANIZATION", "LOCATION", "PRODUCT", "EVENT", "OTHER"}
EXPECTED_RELATIONSHIP_TYPES = {
    "CEO_OF", "CTO_OF", "FOUNDED", "ACQUIRED", "ADVISED", "LOCATED_IN",
    "WORKS_AT", "STUDIES_AT", "FUNDED", "PARTNERS_WITH", "MEMBER_OF",
    "REPRESENTS", "AUTHORED", "DEVELOPED", "LEADS", "TRIAL_SITE_FOR",
    "LAUNCHED_FROM", "CLIENT_OF", "CONTRACTED_FOR", "HELD_IN",
}


def get_driver(uri: str, user: str, password: str):
    return GraphDatabase.driver(uri, auth=(user, password))


# ── Metric functions ─────────────────────────────────────────────────────

def node_coverage(session, gold_standard: dict) -> dict:
    """Measure what fraction of expected entities appear in the graph."""
    # Get all expected entity names from gold standard
    expected_entities = set()
    for doc in gold_standard["documents"]:
        for entity in doc["expected_entities"]:
            expected_entities.add(entity["name"].lower())

    # Get all entity names from graph
    result = session.run("MATCH (e:Entity) RETURN toLower(e.name) AS name")
    graph_entities = {r["name"] for r in result}

    # Compute coverage
    found = expected_entities & graph_entities
    missing = expected_entities - graph_entities
    extra = graph_entities - expected_entities

    coverage = len(found) / len(expected_entities) if expected_entities else 0.0

    return {
        "expected_count": len(expected_entities),
        "graph_count": len(graph_entities),
        "found_count": len(found),
        "coverage": round(coverage, 4),
        "missing_entities": sorted(missing)[:20],
        "extra_entities": sorted(extra)[:20],
    }


def edge_density(session) -> dict:
    """Calculate the density of edges in the graph.

    Density = actual_edges / max_possible_edges
    For a directed graph: max_possible = n * (n - 1)
    """
    node_count = session.run("MATCH (e:Entity) RETURN count(e) AS c").single()["c"]
    edge_count = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]

    max_possible = node_count * (node_count - 1) if node_count > 1 else 1
    density = edge_count / max_possible

    avg_degree_result = session.run("""
        MATCH (e:Entity)
        OPTIONAL MATCH (e)-[r]-()
        RETURN avg(count(r)) AS avg_degree
    """)

    # Compute average degree manually
    degree_result = session.run("""
        MATCH (e:Entity)
        OPTIONAL MATCH (e)-[r]-()
        WITH e, count(r) AS degree
        RETURN avg(degree) AS avg_degree, max(degree) AS max_degree, min(degree) AS min_degree
    """).single()

    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "max_possible_edges": max_possible,
        "density": round(density, 6),
        "avg_degree": round(degree_result["avg_degree"] or 0, 2),
        "max_degree": degree_result["max_degree"] or 0,
        "min_degree": degree_result["min_degree"] or 0,
    }


def connected_components(session) -> dict:
    """Analyze connected components in the graph.

    Uses a simple BFS-based approach via Cypher (no GDS library required).
    """
    # Get all entities and their connections
    result = session.run("""
        MATCH (e:Entity)
        OPTIONAL MATCH (e)-[:RELATES_TO]-(other:Entity)
        RETURN e.entity_id AS node, collect(DISTINCT other.entity_id) AS neighbors
    """)

    # Build adjacency list
    adjacency = {}
    for record in result:
        node = record["node"]
        neighbors = [n for n in record["neighbors"] if n is not None]
        adjacency[node] = neighbors

    # Find connected components using BFS
    visited = set()
    components = []

    for node in adjacency:
        if node in visited:
            continue
        # BFS
        component = []
        queue = [node]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            for neighbor in adjacency.get(current, []):
                if neighbor not in visited:
                    queue.append(neighbor)
        components.append(component)

    component_sizes = sorted([len(c) for c in components], reverse=True)

    return {
        "num_components": len(components),
        "largest_component_size": component_sizes[0] if component_sizes else 0,
        "smallest_component_size": component_sizes[-1] if component_sizes else 0,
        "component_sizes": component_sizes[:10],
        "is_connected": len(components) <= 1,
    }


def orphan_nodes(session) -> dict:
    """Find nodes with no relationships (orphans)."""
    result = session.run("""
        MATCH (e:Entity)
        WHERE NOT (e)-[:RELATES_TO]-()
        RETURN e.name AS name, e.entity_type AS type
    """)

    orphans = [{"name": r["name"], "type": r["type"]} for r in result]
    total_nodes = session.run("MATCH (e:Entity) RETURN count(e) AS c").single()["c"]

    return {
        "orphan_count": len(orphans),
        "total_nodes": total_nodes,
        "orphan_percentage": round(len(orphans) / total_nodes * 100, 2) if total_nodes > 0 else 0,
        "orphans": orphans[:20],
    }


def ontology_conformance(session) -> dict:
    """Check what percentage of entities and relationships conform to the expected schema."""
    # Entity type conformance
    result = session.run("""
        MATCH (e:Entity)
        RETURN e.entity_type AS type, count(e) AS count
    """)
    entity_types = {r["type"]: r["count"] for r in result}

    conforming_entities = sum(c for t, c in entity_types.items() if t in EXPECTED_ENTITY_TYPES)
    total_entities = sum(entity_types.values())
    entity_conformance = conforming_entities / total_entities if total_entities > 0 else 0

    non_conforming_types = {t: c for t, c in entity_types.items() if t not in EXPECTED_ENTITY_TYPES}

    # Relationship type conformance
    result = session.run("""
        MATCH ()-[r:RELATES_TO]->()
        RETURN r.rel_type AS type, count(r) AS count
    """)
    rel_types = {r["type"]: r["count"] for r in result}

    conforming_rels = sum(c for t, c in rel_types.items() if t in EXPECTED_RELATIONSHIP_TYPES)
    total_rels = sum(rel_types.values())
    rel_conformance = conforming_rels / total_rels if total_rels > 0 else 0

    non_conforming_rels = {t: c for t, c in rel_types.items() if t not in EXPECTED_RELATIONSHIP_TYPES}

    return {
        "entity_conformance": round(entity_conformance, 4),
        "entity_type_distribution": entity_types,
        "non_conforming_entity_types": non_conforming_types,
        "relationship_conformance": round(rel_conformance, 4),
        "relationship_type_distribution": rel_types,
        "non_conforming_relationship_types": non_conforming_rels,
    }


def centrality_analysis(session) -> dict:
    """Analyze degree centrality distribution.

    Identifies the most and least connected entities.
    """
    result = session.run("""
        MATCH (e:Entity)
        OPTIONAL MATCH (e)-[r]-()
        WITH e.name AS name, e.entity_type AS type, count(r) AS degree
        RETURN name, type, degree
        ORDER BY degree DESC
    """)

    entities = [{"name": r["name"], "type": r["type"], "degree": r["degree"]} for r in result]

    if not entities:
        return {"top_entities": [], "degree_distribution": {}, "mean_degree": 0, "median_degree": 0}

    degrees = [e["degree"] for e in entities]
    degree_distribution = dict(Counter(degrees))

    # Calculate statistics
    mean_degree = sum(degrees) / len(degrees)
    sorted_degrees = sorted(degrees)
    median_degree = sorted_degrees[len(sorted_degrees) // 2]

    return {
        "top_entities": entities[:10],
        "bottom_entities": entities[-5:] if len(entities) > 5 else entities,
        "degree_distribution": degree_distribution,
        "mean_degree": round(mean_degree, 2),
        "median_degree": median_degree,
        "std_degree": round(
            (sum((d - mean_degree) ** 2 for d in degrees) / len(degrees)) ** 0.5, 2
        ),
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Graph Quality Metrics")
    parser.add_argument("--uri", type=str, default="bolt://localhost:7687")
    parser.add_argument("--user", type=str, default="neo4j")
    parser.add_argument("--password", type=str, default="password")
    args = parser.parse_args()

    print("=" * 70)
    print("  KG Evaluation: Graph Quality Metrics")
    print("=" * 70)

    driver = get_driver(args.uri, args.user, args.password)

    try:
        driver.verify_connectivity()
        print("  Connected to Neo4j.\n")
    except Exception as e:
        print(f"  Error: {e}")
        print("  Make sure Neo4j is running with data loaded.")
        sys.exit(1)

    # Load gold standard for coverage comparison
    gold_standard = None
    if GOLD_STANDARD_PATH.exists():
        with open(GOLD_STANDARD_PATH) as f:
            gold_standard = json.load(f)

    all_metrics = {}

    with driver.session() as session:
        # 1. Node Coverage
        if gold_standard:
            print(f"{'─'*60}")
            print("  1. NODE COVERAGE")
            print(f"{'─'*60}")
            coverage = node_coverage(session, gold_standard)
            all_metrics["node_coverage"] = coverage
            print(f"    Expected entities: {coverage['expected_count']}")
            print(f"    Graph entities:    {coverage['graph_count']}")
            print(f"    Coverage:          {coverage['coverage']:.2%}")
            if coverage["missing_entities"]:
                print(f"    Missing (sample):  {', '.join(coverage['missing_entities'][:5])}")
            print()

        # 2. Edge Density
        print(f"{'─'*60}")
        print("  2. EDGE DENSITY")
        print(f"{'─'*60}")
        density = edge_density(session)
        all_metrics["edge_density"] = density
        print(f"    Nodes:         {density['node_count']}")
        print(f"    Edges:         {density['edge_count']}")
        print(f"    Density:       {density['density']:.6f}")
        print(f"    Avg degree:    {density['avg_degree']}")
        print(f"    Max degree:    {density['max_degree']}")
        print()

        # 3. Connected Components
        print(f"{'─'*60}")
        print("  3. CONNECTED COMPONENTS")
        print(f"{'─'*60}")
        components = connected_components(session)
        all_metrics["connected_components"] = components
        print(f"    Components:     {components['num_components']}")
        print(f"    Largest:        {components['largest_component_size']} nodes")
        print(f"    Fully connected: {'Yes' if components['is_connected'] else 'No'}")
        if components["component_sizes"]:
            print(f"    Sizes:          {components['component_sizes']}")
        print()

        # 4. Orphan Nodes
        print(f"{'─'*60}")
        print("  4. ORPHAN NODES")
        print(f"{'─'*60}")
        orphans = orphan_nodes(session)
        all_metrics["orphan_nodes"] = orphans
        print(f"    Orphan count:   {orphans['orphan_count']} / {orphans['total_nodes']}")
        print(f"    Orphan %:       {orphans['orphan_percentage']}%")
        if orphans["orphans"]:
            for o in orphans["orphans"][:5]:
                print(f"      - {o['name']} ({o['type']})")
        print()

        # 5. Ontology Conformance
        print(f"{'─'*60}")
        print("  5. ONTOLOGY CONFORMANCE")
        print(f"{'─'*60}")
        conformance = ontology_conformance(session)
        all_metrics["ontology_conformance"] = conformance
        print(f"    Entity type conformance:  {conformance['entity_conformance']:.2%}")
        print(f"    Entity type distribution:")
        for t, c in sorted(conformance["entity_type_distribution"].items(), key=lambda x: -x[1]):
            marker = "" if t in EXPECTED_ENTITY_TYPES else " [NON-CONFORMING]"
            print(f"      {t:<20} {c}{marker}")
        print(f"\n    Relationship conformance: {conformance['relationship_conformance']:.2%}")
        if conformance["non_conforming_relationship_types"]:
            print(f"    Non-conforming rel types:")
            for t, c in conformance["non_conforming_relationship_types"].items():
                print(f"      {t:<25} {c}")
        print()

        # 6. Centrality Analysis
        print(f"{'─'*60}")
        print("  6. CENTRALITY ANALYSIS")
        print(f"{'─'*60}")
        centrality = centrality_analysis(session)
        all_metrics["centrality"] = centrality
        print(f"    Mean degree:    {centrality['mean_degree']}")
        print(f"    Median degree:  {centrality['median_degree']}")
        print(f"    Std deviation:  {centrality['std_degree']}")
        print(f"\n    Top 5 most connected entities:")
        for e in centrality["top_entities"][:5]:
            print(f"      {e['name']:<30} degree={e['degree']} ({e['type']})")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "graph_quality_metrics.json"
    with open(output_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")

    driver.close()


if __name__ == "__main__":
    main()
