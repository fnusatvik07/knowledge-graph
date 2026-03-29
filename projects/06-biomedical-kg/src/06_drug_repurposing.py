"""Drug repurposing analysis using multi-hop reasoning on the biomedical KG.

Core idea: If Drug A treats Disease X, and Disease X shares genetic associations
with Disease Y, then Drug A might be a repurposing candidate for Disease Y.

This script implements several repurposing strategies:
  1. Shared gene targets: Drug -> Gene <- Disease (unexplored)
  2. Shared pathways: Drug -> Pathway <- Gene <- Disease
  3. Protein interaction bridges: Drug -> Protein1 -> Protein2 <- Disease

Usage:
    python src/06_drug_repurposing.py
    python src/06_drug_repurposing.py --drug "metformin"
    python src/06_drug_repurposing.py --disease "Parkinson's disease"

References:
    - Neo4j Cypher path queries: https://neo4j.com/docs/cypher-manual/current/patterns/
    - LangChain Neo4j: https://python.langchain.com/docs/integrations/graphs/neo4j_cypher/
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


class DrugRepurposingAnalyzer:
    """Analyze potential drug repurposing candidates using the biomedical KG."""

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()

    def close(self):
        self.driver.close()

    def _run_query(self, query: str, params: dict = None) -> list[dict]:
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]

    # ── Strategy 1: Shared Gene Targets ──────────────────────────────────

    def shared_gene_repurposing(self, drug_name: str = None, disease_name: str = None) -> list[dict]:
        """Find repurposing candidates via shared gene associations.

        Pattern: Drug -[:TARGETS|INHIBITS]-> Protein/Gene <-[:ASSOCIATED_WITH]- Disease
        where the Drug does NOT already TREAT that Disease.

        Args:
            drug_name: Find new diseases for this drug.
            disease_name: Find new drugs for this disease.

        Returns:
            List of candidate dicts with drug, disease, shared genes, and score.
        """
        if drug_name:
            query = """
                // Find genes/proteins that the drug targets
                MATCH (d:Drug)-[:TARGETS|INHIBITS]->(target)
                WHERE toLower(d.name) CONTAINS toLower($name)

                // Find diseases associated with those same genes/proteins
                MATCH (target)<-[:ASSOCIATED_WITH|CAUSES]-(gene:Gene)-[:ASSOCIATED_WITH]->(candidate_disease:Disease)

                // Exclude diseases the drug already treats
                WHERE NOT (d)-[:TREATS]->(candidate_disease)

                // Count shared targets as a score
                WITH d, candidate_disease,
                     collect(DISTINCT target.name) AS shared_targets,
                     collect(DISTINCT gene.name) AS shared_genes
                RETURN d.name AS drug,
                       candidate_disease.name AS candidate_disease,
                       shared_targets,
                       shared_genes,
                       size(shared_targets) + size(shared_genes) AS repurposing_score
                ORDER BY repurposing_score DESC
                LIMIT 20
            """
            return self._run_query(query, {"name": drug_name})
        elif disease_name:
            query = """
                // Find genes associated with the target disease
                MATCH (gene:Gene)-[:ASSOCIATED_WITH]->(target_disease:Disease)
                WHERE toLower(target_disease.name) CONTAINS toLower($name)

                // Find drugs that target those genes or their protein products
                MATCH (candidate_drug:Drug)-[:TARGETS|INHIBITS]->(protein)
                WHERE protein.name = gene.name OR (protein)-[:INTERACTS_WITH]->(:Protein {name: gene.name})

                // Exclude drugs that already treat this disease
                WHERE NOT (candidate_drug)-[:TREATS]->(target_disease)

                WITH candidate_drug, target_disease,
                     collect(DISTINCT gene.name) AS shared_genes
                RETURN candidate_drug.name AS drug,
                       target_disease.name AS disease,
                       shared_genes,
                       size(shared_genes) AS repurposing_score
                ORDER BY repurposing_score DESC
                LIMIT 20
            """
            return self._run_query(query, {"name": disease_name})
        return []

    # ── Strategy 2: Shared Pathway Repurposing ───────────────────────────

    def shared_pathway_repurposing(self) -> list[dict]:
        """Find repurposing candidates via shared pathway modulation.

        Pattern: Drug -[:ACTIVATES|INHIBITS]-> Pathway <-[:ACTIVATES]- Gene -[:ASSOCIATED_WITH]-> Disease
        where the Drug does NOT already TREAT that Disease.
        """
        query = """
            // Find drugs that modulate a pathway
            MATCH (d:Drug)-[r1:ACTIVATES|INHIBITS]->(pathway:Pathway)

            // Find genes that also modulate the same pathway and are associated with a disease
            MATCH (gene:Gene)-[:ACTIVATES]->(pathway)
            MATCH (gene)-[:ASSOCIATED_WITH]->(candidate_disease:Disease)

            // Exclude diseases the drug already treats
            WHERE NOT (d)-[:TREATS]->(candidate_disease)

            WITH d, candidate_disease, pathway,
                 collect(DISTINCT gene.name) AS pathway_genes,
                 type(r1) AS drug_action
            RETURN d.name AS drug,
                   candidate_disease.name AS candidate_disease,
                   pathway.name AS shared_pathway,
                   drug_action AS drug_pathway_action,
                   pathway_genes,
                   size(pathway_genes) AS repurposing_score
            ORDER BY repurposing_score DESC
            LIMIT 20
        """
        return self._run_query(query)

    # ── Strategy 3: Disease Similarity via Shared Genes ──────────────────

    def disease_similarity_repurposing(self) -> list[dict]:
        """Find repurposing candidates by disease genetic similarity.

        If Disease A and Disease B share genes, drugs for Disease A
        might work for Disease B.

        Pattern: Drug -[:TREATS]-> DiseaseA <-[:ASSOCIATED_WITH]- Gene -[:ASSOCIATED_WITH]-> DiseaseB
        """
        query = """
            // Find pairs of diseases that share gene associations
            MATCH (d1:Disease)<-[:ASSOCIATED_WITH]-(shared_gene:Gene)-[:ASSOCIATED_WITH]->(d2:Disease)
            WHERE d1 <> d2

            // Find drugs that treat the first disease but not the second
            MATCH (drug:Drug)-[:TREATS]->(d1)
            WHERE NOT (drug)-[:TREATS]->(d2)

            WITH drug, d1, d2,
                 collect(DISTINCT shared_gene.name) AS shared_genes
            RETURN drug.name AS drug,
                   d1.name AS treats_disease,
                   d2.name AS candidate_disease,
                   shared_genes,
                   size(shared_genes) AS genetic_similarity_score
            ORDER BY genetic_similarity_score DESC
            LIMIT 20
        """
        return self._run_query(query)

    # ── Strategy 4: Protein Interaction Bridge ───────────────────────────

    def protein_bridge_repurposing(self) -> list[dict]:
        """Find repurposing candidates via protein-protein interaction bridges.

        Pattern: Drug -[:TARGETS]-> Protein1 -[:INTERACTS_WITH]-> Protein2 <-[:ASSOCIATED_WITH]- Gene -> Disease
        """
        query = """
            MATCH (drug:Drug)-[:TARGETS]->(p1:Protein)-[:INTERACTS_WITH]-(p2:Protein)
            MATCH (gene:Gene)-[:ASSOCIATED_WITH]->(disease:Disease)
            WHERE p2.name = gene.name OR (gene)-[:EXPRESSED_IN]->()<-[:EXPRESSED_IN]-(p2)

            WHERE NOT (drug)-[:TREATS]->(disease)

            WITH drug, disease, p1, p2,
                 collect(DISTINCT gene.name) AS bridging_genes
            RETURN drug.name AS drug,
                   disease.name AS candidate_disease,
                   p1.name AS drug_target,
                   p2.name AS interacting_protein,
                   bridging_genes,
                   size(bridging_genes) AS bridge_score
            ORDER BY bridge_score DESC
            LIMIT 20
        """
        return self._run_query(query)

    # ── Comprehensive repurposing report ─────────────────────────────────

    def generate_repurposing_report(self, drug_name: str = None, disease_name: str = None) -> dict:
        """Generate a comprehensive drug repurposing report.

        Combines all strategies and ranks candidates.
        """
        report = {
            "query": {"drug": drug_name, "disease": disease_name},
            "strategies": {},
        }

        print("Running Strategy 1: Shared Gene Targets...")
        if drug_name or disease_name:
            results = self.shared_gene_repurposing(drug_name=drug_name, disease_name=disease_name)
            report["strategies"]["shared_gene_targets"] = results
            print(f"  Found {len(results)} candidates")

        print("Running Strategy 2: Shared Pathway Repurposing...")
        results = self.shared_pathway_repurposing()
        report["strategies"]["shared_pathway"] = results
        print(f"  Found {len(results)} candidates")

        print("Running Strategy 3: Disease Similarity...")
        results = self.disease_similarity_repurposing()
        report["strategies"]["disease_similarity"] = results
        print(f"  Found {len(results)} candidates")

        print("Running Strategy 4: Protein Interaction Bridge...")
        results = self.protein_bridge_repurposing()
        report["strategies"]["protein_bridge"] = results
        print(f"  Found {len(results)} candidates")

        # Aggregate all candidates and rank
        all_candidates = {}
        for strategy_name, candidates in report["strategies"].items():
            for c in candidates:
                drug = c.get("drug", "")
                disease = c.get("candidate_disease", c.get("disease", ""))
                key = f"{drug}|{disease}"
                if key not in all_candidates:
                    all_candidates[key] = {
                        "drug": drug,
                        "candidate_disease": disease,
                        "strategies": [],
                        "total_score": 0,
                    }
                score = c.get("repurposing_score", c.get("genetic_similarity_score",
                              c.get("bridge_score", 1)))
                all_candidates[key]["strategies"].append(strategy_name)
                all_candidates[key]["total_score"] += score

        ranked = sorted(all_candidates.values(), key=lambda x: -x["total_score"])
        report["ranked_candidates"] = ranked[:20]

        return report


def print_repurposing_results(title: str, results: list[dict]):
    """Pretty-print repurposing candidates."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    if not results:
        print("  No candidates found.")
        return
    for i, r in enumerate(results[:10], 1):
        print(f"\n  [{i}]")
        for k, v in r.items():
            if isinstance(v, list) and len(v) > 5:
                print(f"    {k}: {v[:5]}... ({len(v)} total)")
            else:
                print(f"    {k}: {v}")


def main():
    parser = argparse.ArgumentParser(description="Drug repurposing analysis via biomedical KG.")
    parser.add_argument("--neo4j_uri", type=str, default=DEFAULT_URI)
    parser.add_argument("--neo4j_user", type=str, default=DEFAULT_USER)
    parser.add_argument("--neo4j_password", type=str, default=DEFAULT_PASSWORD)
    parser.add_argument("--drug", type=str, default=None,
                        help="Find repurposing candidates for this drug")
    parser.add_argument("--disease", type=str, default=None,
                        help="Find repurposing candidates for this disease")
    args = parser.parse_args()

    analyzer = DrugRepurposingAnalyzer(args.neo4j_uri, args.neo4j_user, args.neo4j_password)

    try:
        print("Drug Repurposing Analysis")
        print("=" * 60)

        report = analyzer.generate_repurposing_report(
            drug_name=args.drug,
            disease_name=args.disease,
        )

        # Print results for each strategy
        for strategy_name, candidates in report["strategies"].items():
            print_repurposing_results(
                f"Strategy: {strategy_name.replace('_', ' ').title()}",
                candidates,
            )

        # Print ranked candidates
        if report["ranked_candidates"]:
            print(f"\n{'='*60}")
            print("  TOP REPURPOSING CANDIDATES (all strategies combined)")
            print(f"{'='*60}")
            for i, candidate in enumerate(report["ranked_candidates"][:10], 1):
                print(f"\n  [{i}] {candidate['drug']} -> {candidate['candidate_disease']}")
                print(f"      Score: {candidate['total_score']}")
                print(f"      Supported by: {', '.join(candidate['strategies'])}")

        # Save report
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_file = OUTPUT_DIR / "drug_repurposing_report.json"
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nFull report saved to {output_file}")

    finally:
        analyzer.close()


if __name__ == "__main__":
    main()
