"""Domain-specific queries for the biomedical knowledge graph.

Demonstrates Cypher-based traversals for common biomedical questions:
  - What drugs treat a given disease?
  - What genes are associated with a disease?
  - Find pathways connecting a gene to a disease (multi-hop)
  - Drug mechanism of action (drug -> protein -> pathway)

Usage:
    python src/05_query_biomedical_kg.py
    python src/05_query_biomedical_kg.py --disease "breast cancer"
    python src/05_query_biomedical_kg.py --gene "TP53"

References:
    - Neo4j Cypher manual: https://neo4j.com/docs/cypher-manual/current/
    - LangChain Neo4j integration: https://python.langchain.com/docs/integrations/graphs/neo4j_cypher/
"""

import argparse
import json
import sys
from pathlib import Path

# ── Shared imports ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from neo4j import GraphDatabase

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"

# ── Neo4j defaults ──────────────────────────────────────────────────────
DEFAULT_URI = "bolt://localhost:7687"
DEFAULT_USER = "neo4j"
DEFAULT_PASSWORD = "password"


class BiomedicalKGQuerier:
    """Query interface for the biomedical knowledge graph."""

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()

    def close(self):
        self.driver.close()

    def _run_query(self, query: str, params: dict = None) -> list[dict]:
        """Execute a Cypher query and return results as list of dicts."""
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]

    # ── Query 1: Drugs that treat a disease ──────────────────────────────

    def drugs_for_disease(self, disease_name: str) -> list[dict]:
        """Find all drugs that treat a given disease.

        Cypher pattern: (Drug)-[:TREATS]->(Disease)
        """
        query = """
            MATCH (d:Drug)-[r:TREATS]->(dis:Disease)
            WHERE toLower(dis.name) CONTAINS toLower($disease_name)
            RETURN d.name AS drug, dis.name AS disease,
                   r.confidence AS confidence, r.evidence AS evidence
            ORDER BY r.confidence DESC
        """
        return self._run_query(query, {"disease_name": disease_name})

    # ── Query 2: Genes associated with a disease ─────────────────────────

    def genes_for_disease(self, disease_name: str) -> list[dict]:
        """Find genes associated with a given disease.

        Cypher pattern: (Gene)-[:ASSOCIATED_WITH|CAUSES]->(Disease)
        """
        query = """
            MATCH (g:Gene)-[r:ASSOCIATED_WITH|CAUSES]->(dis:Disease)
            WHERE toLower(dis.name) CONTAINS toLower($disease_name)
            RETURN g.name AS gene, type(r) AS relationship,
                   dis.name AS disease, r.confidence AS confidence
            ORDER BY r.confidence DESC
        """
        return self._run_query(query, {"disease_name": disease_name})

    # ── Query 3: Drug mechanism of action ────────────────────────────────

    def drug_mechanism(self, drug_name: str) -> list[dict]:
        """Find the mechanism of action for a drug.

        Traverses: Drug -[:TARGETS|INHIBITS]-> Protein/Gene
                   and further to pathways if available.
        """
        query = """
            MATCH (d:Drug)-[r1:TARGETS|INHIBITS]->(p)
            WHERE toLower(d.name) CONTAINS toLower($drug_name)
            OPTIONAL MATCH (p)-[r2:ACTIVATES|ASSOCIATED_WITH]->(downstream)
            RETURN d.name AS drug, type(r1) AS action,
                   p.name AS target, labels(p) AS target_type,
                   type(r2) AS downstream_relation,
                   downstream.name AS downstream_target,
                   CASE WHEN downstream IS NOT NULL THEN labels(downstream) ELSE NULL END AS downstream_type
            ORDER BY d.name
        """
        return self._run_query(query, {"drug_name": drug_name})

    # ── Query 4: Pathways connecting gene to disease (multi-hop) ─────────

    def gene_disease_paths(self, gene_name: str, disease_name: str, max_hops: int = 4) -> list[dict]:
        """Find all paths connecting a gene to a disease.

        Uses variable-length path matching for multi-hop traversal.
        """
        query = """
            MATCH path = (g:Gene {name: $gene_name})-[*1..$max_hops]-(dis:Disease)
            WHERE toLower(dis.name) CONTAINS toLower($disease_name)
            RETURN [node in nodes(path) | node.name] AS path_nodes,
                   [rel in relationships(path) | type(rel)] AS path_relationships,
                   length(path) AS path_length
            ORDER BY path_length ASC
            LIMIT 10
        """
        return self._run_query(query, {
            "gene_name": gene_name,
            "disease_name": disease_name,
            "max_hops": max_hops,
        })

    # ── Query 5: All relationships for an entity ─────────────────────────

    def entity_neighborhood(self, entity_name: str, depth: int = 1) -> list[dict]:
        """Get all relationships for an entity up to a given depth.

        Useful for exploring the local neighborhood of any entity.
        """
        query = """
            MATCH path = (n:Entity {name: $entity_name})-[*1..$depth]-(m:Entity)
            RETURN n.name AS source, labels(n) AS source_labels,
                   [rel in relationships(path) | type(rel)] AS relationships,
                   m.name AS target, labels(m) AS target_labels
            LIMIT 50
        """
        return self._run_query(query, {"entity_name": entity_name, "depth": depth})

    # ── Query 6: Graph statistics ────────────────────────────────────────

    def graph_statistics(self) -> dict:
        """Get overall statistics about the biomedical KG."""
        stats = {}

        # Node counts by label
        result = self._run_query("""
            MATCH (n:Entity)
            WITH labels(n) AS lbls, count(n) AS cnt
            UNWIND lbls AS label
            WITH label, sum(cnt) AS total
            WHERE label <> 'Entity'
            RETURN label, total
            ORDER BY total DESC
        """)
        stats["nodes_by_type"] = {r["label"]: r["total"] for r in result}

        # Relationship counts by type
        result = self._run_query("""
            MATCH ()-[r]->()
            RETURN type(r) AS rel_type, count(r) AS total
            ORDER BY total DESC
        """)
        stats["relationships_by_type"] = {r["rel_type"]: r["total"] for r in result}

        # Total counts
        result = self._run_query("MATCH (n:Entity) RETURN count(n) AS total")
        stats["total_nodes"] = result[0]["total"] if result else 0
        result = self._run_query("MATCH ()-[r]->() RETURN count(r) AS total")
        stats["total_relationships"] = result[0]["total"] if result else 0

        return stats


def print_results(title: str, results: list[dict]):
    """Pretty-print query results."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    if not results:
        print("  No results found.")
        return
    for i, r in enumerate(results, 1):
        print(f"\n  [{i}]")
        for k, v in r.items():
            print(f"    {k}: {v}")


def main():
    parser = argparse.ArgumentParser(description="Query the biomedical knowledge graph.")
    parser.add_argument("--neo4j_uri", type=str, default=DEFAULT_URI)
    parser.add_argument("--neo4j_user", type=str, default=DEFAULT_USER)
    parser.add_argument("--neo4j_password", type=str, default=DEFAULT_PASSWORD)
    parser.add_argument("--disease", type=str, default=None, help="Query drugs/genes for this disease")
    parser.add_argument("--drug", type=str, default=None, help="Query mechanism of action for this drug")
    parser.add_argument("--gene", type=str, default=None, help="Query relationships for this gene")
    args = parser.parse_args()

    querier = BiomedicalKGQuerier(args.neo4j_uri, args.neo4j_user, args.neo4j_password)

    try:
        # Always show graph statistics
        stats = querier.graph_statistics()
        print("\nBiomedical KG Statistics:")
        print(f"  Total nodes: {stats['total_nodes']}")
        print(f"  Total relationships: {stats['total_relationships']}")
        print("\n  Nodes by type:")
        for t, c in stats.get("nodes_by_type", {}).items():
            print(f"    {t}: {c}")
        print("\n  Relationships by type:")
        for t, c in stats.get("relationships_by_type", {}).items():
            print(f"    {t}: {c}")

        # Run specific queries based on arguments
        if args.disease:
            results = querier.drugs_for_disease(args.disease)
            print_results(f"Drugs that treat '{args.disease}'", results)

            results = querier.genes_for_disease(args.disease)
            print_results(f"Genes associated with '{args.disease}'", results)

        if args.drug:
            results = querier.drug_mechanism(args.drug)
            print_results(f"Mechanism of action: {args.drug}", results)

        if args.gene:
            results = querier.entity_neighborhood(args.gene, depth=2)
            print_results(f"Neighborhood of gene '{args.gene}'", results)

        # If no specific query, run demo queries
        if not args.disease and not args.drug and not args.gene:
            print("\n\nRunning demo queries...\n")

            # Demo: Drugs for breast cancer
            results = querier.drugs_for_disease("breast cancer")
            print_results("Drugs that treat breast cancer", results)

            # Demo: Genes for colorectal cancer
            results = querier.genes_for_disease("colorectal cancer")
            print_results("Genes associated with colorectal cancer", results)

            # Demo: Olaparib mechanism
            results = querier.drug_mechanism("olaparib")
            print_results("Mechanism of action: olaparib", results)

            # Demo: TP53 neighborhood
            results = querier.entity_neighborhood("TP53", depth=2)
            print_results("Neighborhood of TP53 (depth=2)", results)

            # Demo: Multi-hop path from BRCA1 to breast cancer
            results = querier.gene_disease_paths("BRCA1", "breast cancer")
            print_results("Paths from BRCA1 to breast cancer", results)

        # Save all results
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        all_results = {
            "statistics": stats,
            "query_results": {},
        }
        if args.disease:
            all_results["query_results"]["drugs_for_disease"] = querier.drugs_for_disease(args.disease)
            all_results["query_results"]["genes_for_disease"] = querier.genes_for_disease(args.disease)
        output_file = OUTPUT_DIR / "query_results.json"
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to {output_file}")

    finally:
        querier.close()


if __name__ == "__main__":
    main()
