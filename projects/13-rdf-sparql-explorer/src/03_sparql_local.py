"""Run SPARQL queries on a local RDF graph.

Demonstrates:
- SELECT queries
- FILTER (matching criteria)
- OPTIONAL (left join)
- Aggregation (COUNT, GROUP BY)
- CONSTRUCT (extract subgraphs)
- Property paths (transitive connections)
"""

import sys
from pathlib import Path

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, OWL, FOAF, SKOS, XSD

# ── Path setup ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Namespaces
EX = Namespace("http://example.org/kg/")
TECH = Namespace("http://example.org/tech/")
ORG = Namespace("http://example.org/org/")


def build_graph() -> Graph:
    """Build the same RDF graph as script 01, for querying."""
    g = Graph()
    g.bind("ex", EX)
    g.bind("tech", TECH)
    g.bind("org", ORG)
    g.bind("foaf", FOAF)
    g.bind("rdfs", RDFS)
    g.bind("skos", SKOS)

    # People
    people = [
        (EX.GeoffreyHinton, "Geoffrey Hinton", 1947, [ORG.UniversityOfToronto], [TECH.DeepLearning, TECH.NeuralNetworks]),
        (EX.YannLeCun, "Yann LeCun", 1960, [ORG.Meta, ORG.NYU], [TECH.ComputerVision, TECH.DeepLearning]),
        (EX.YoshuaBengio, "Yoshua Bengio", 1964, [ORG.Mila], [TECH.DeepLearning, TECH.NLP]),
        (EX.FeiFeiLi, "Fei-Fei Li", 1976, [ORG.Stanford], [TECH.ComputerVision]),
        (EX.TimBernersLee, "Tim Berners-Lee", 1955, [ORG.MIT], [TECH.SemanticWeb]),
        (EX.AndrewNg, "Andrew Ng", 1976, [ORG.Stanford], [TECH.DeepLearning, TECH.MachineLearning]),
        (EX.DemisHassabis, "Demis Hassabis", 1976, [ORG.DeepMind], [TECH.ReinforcementLearning, TECH.ArtificialIntelligence]),
    ]

    for uri, name, birth_year, orgs, areas in people:
        g.add((uri, RDF.type, FOAF.Person))
        g.add((uri, FOAF.name, Literal(name)))
        g.add((uri, EX.birthYear, Literal(birth_year, datatype=XSD.integer)))
        for org in orgs:
            g.add((uri, EX.worksAt, org))
        for area in areas:
            g.add((uri, EX.researchArea, area))

    # Awards
    g.add((EX.GeoffreyHinton, EX.wonAward, EX.TuringAward2018))
    g.add((EX.YannLeCun, EX.wonAward, EX.TuringAward2018))
    g.add((EX.YoshuaBengio, EX.wonAward, EX.TuringAward2018))
    g.add((EX.GeoffreyHinton, EX.wonAward, EX.NobelPrize2024))
    g.add((EX.DemisHassabis, EX.wonAward, EX.NobelPrize2024))

    g.add((EX.TuringAward2018, RDF.type, EX.Award))
    g.add((EX.TuringAward2018, RDFS.label, Literal("Turing Award 2018")))
    g.add((EX.TuringAward2018, EX.year, Literal(2018, datatype=XSD.integer)))

    g.add((EX.NobelPrize2024, RDF.type, EX.Award))
    g.add((EX.NobelPrize2024, RDFS.label, Literal("Nobel Prize in Physics 2024")))
    g.add((EX.NobelPrize2024, EX.year, Literal(2024, datatype=XSD.integer)))

    # Organizations
    orgs_data = [
        (ORG.UniversityOfToronto, "University of Toronto", "University"),
        (ORG.Meta, "Meta", "Company"),
        (ORG.NYU, "New York University", "University"),
        (ORG.Mila, "Mila - Quebec AI Institute", "ResearchLab"),
        (ORG.Stanford, "Stanford University", "University"),
        (ORG.MIT, "MIT", "University"),
        (ORG.Google, "Google", "Company"),
        (ORG.DeepMind, "DeepMind", "ResearchLab"),
    ]
    for uri, name, org_type in orgs_data:
        g.add((uri, RDF.type, FOAF.Organization))
        g.add((uri, FOAF.name, Literal(name)))
        g.add((uri, EX.orgType, Literal(org_type)))

    # Technologies with hierarchy
    techs = [
        (TECH.ArtificialIntelligence, "Artificial Intelligence", None),
        (TECH.MachineLearning, "Machine Learning", TECH.ArtificialIntelligence),
        (TECH.DeepLearning, "Deep Learning", TECH.MachineLearning),
        (TECH.NeuralNetworks, "Neural Networks", TECH.DeepLearning),
        (TECH.ComputerVision, "Computer Vision", TECH.ArtificialIntelligence),
        (TECH.NLP, "Natural Language Processing", TECH.ArtificialIntelligence),
        (TECH.SemanticWeb, "Semantic Web", None),
        (TECH.ReinforcementLearning, "Reinforcement Learning", TECH.MachineLearning),
    ]
    for uri, label, broader in techs:
        g.add((uri, RDF.type, SKOS.Concept))
        g.add((uri, SKOS.prefLabel, Literal(label)))
        if broader:
            g.add((uri, SKOS.broader, broader))
            g.add((broader, SKOS.narrower, uri))

    # Mentoring relationships
    g.add((EX.GeoffreyHinton, EX.mentored, EX.YannLeCun))
    g.add((EX.GeoffreyHinton, FOAF.knows, EX.YoshuaBengio))
    g.add((EX.YannLeCun, FOAF.knows, EX.YoshuaBengio))
    g.add((EX.AndrewNg, FOAF.knows, EX.FeiFeiLi))

    return g


def run_query(g: Graph, title: str, query: str) -> None:
    """Execute a SPARQL query and print results."""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")
    print(f"  Query:\n{query.strip()}\n")

    results = g.query(query)

    if hasattr(results, "bindings") and results.bindings:
        # SELECT query
        vars_ = [str(v) for v in results.vars]
        # Print header
        header = "  " + " | ".join(f"{v:25s}" for v in vars_)
        print(header)
        print("  " + "-" * len(header))
        for row in results:
            values = []
            for v in results.vars:
                val = row[v]
                if val is None:
                    values.append("(none)")
                elif isinstance(val, Literal):
                    values.append(str(val))
                else:
                    values.append(str(val).split("/")[-1])
            print("  " + " | ".join(f"{v:25s}" for v in values))
        print(f"\n  ({len(results)} results)")
    elif hasattr(results, "graph"):
        # CONSTRUCT query
        result_graph = results.graph
        print(f"  Constructed graph: {len(result_graph)} triples")
        for s, p, o in sorted(result_graph, key=lambda x: str(x[0])):
            s_label = str(s).split("/")[-1]
            p_label = str(p).split("/")[-1].split("#")[-1]
            o_label = str(o).split("/")[-1] if isinstance(o, URIRef) else str(o)
            print(f"    {s_label:20s} {p_label:15s} {o_label}")
    else:
        print("  (no results)")


def main():
    print("Building RDF graph for SPARQL queries...\n")
    g = build_graph()
    print(f"Graph has {len(g)} triples\n")

    # ── 1. Basic SELECT ───────────────────────────────────────────────

    run_query(g, "1. SELECT: All people and their organizations", """
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        PREFIX ex: <http://example.org/kg/>

        SELECT ?name ?org_name
        WHERE {
            ?person a foaf:Person ;
                    foaf:name ?name ;
                    ex:worksAt ?org .
            ?org foaf:name ?org_name .
        }
        ORDER BY ?name
    """)

    # ── 2. FILTER ─────────────────────────────────────────────────────

    run_query(g, "2. FILTER: People born before 1960", """
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        PREFIX ex: <http://example.org/kg/>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        SELECT ?name ?birthYear
        WHERE {
            ?person a foaf:Person ;
                    foaf:name ?name ;
                    ex:birthYear ?birthYear .
            FILTER (?birthYear < 1960)
        }
        ORDER BY ?birthYear
    """)

    # ── 3. OPTIONAL (left join) ───────────────────────────────────────

    run_query(g, "3. OPTIONAL: All people with awards (if any)", """
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        PREFIX ex: <http://example.org/kg/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?name ?award_label
        WHERE {
            ?person a foaf:Person ;
                    foaf:name ?name .
            OPTIONAL {
                ?person ex:wonAward ?award .
                ?award rdfs:label ?award_label .
            }
        }
        ORDER BY ?name
    """)

    # ── 4. Aggregation ────────────────────────────────────────────────

    run_query(g, "4. AGGREGATION: Count people per organization", """
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        PREFIX ex: <http://example.org/kg/>

        SELECT ?org_name (COUNT(?person) AS ?num_people)
        WHERE {
            ?person a foaf:Person ;
                    ex:worksAt ?org .
            ?org foaf:name ?org_name .
        }
        GROUP BY ?org_name
        ORDER BY DESC(?num_people)
    """)

    run_query(g, "4b. AGGREGATION: Count research areas per person", """
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        PREFIX ex: <http://example.org/kg/>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

        SELECT ?name (COUNT(?area) AS ?num_areas) (GROUP_CONCAT(?area_label; separator=", ") AS ?areas)
        WHERE {
            ?person a foaf:Person ;
                    foaf:name ?name ;
                    ex:researchArea ?area .
            ?area skos:prefLabel ?area_label .
        }
        GROUP BY ?name
        ORDER BY DESC(?num_areas)
    """)

    # ── 5. CONSTRUCT (extract subgraph) ───────────────────────────────

    run_query(g, "5. CONSTRUCT: Extract the award-winner subgraph", """
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        PREFIX ex: <http://example.org/kg/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        CONSTRUCT {
            ?person foaf:name ?name .
            ?person ex:wonAward ?award .
            ?award rdfs:label ?award_label .
        }
        WHERE {
            ?person a foaf:Person ;
                    foaf:name ?name ;
                    ex:wonAward ?award .
            ?award rdfs:label ?award_label .
        }
    """)

    # ── 6. Property Paths ─────────────────────────────────────────────

    run_query(g, "6. PROPERTY PATHS: Transitive concept hierarchy (broader+)", """
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

        SELECT ?concept_label ?ancestor_label
        WHERE {
            ?concept skos:prefLabel ?concept_label ;
                     skos:broader+ ?ancestor .
            ?ancestor skos:prefLabel ?ancestor_label .
        }
        ORDER BY ?concept_label
    """)

    run_query(g, "6b. PROPERTY PATHS: People connected via knows (1 or 2 hops)", """
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>

        SELECT ?name1 ?name2
        WHERE {
            ?p1 foaf:name ?name1 .
            ?p1 foaf:knows{1,2} ?p2 .
            ?p2 foaf:name ?name2 .
            FILTER (?p1 != ?p2)
        }
        ORDER BY ?name1
    """)

    # ── Bonus: SPARQL with UNION ──────────────────────────────────────

    run_query(g, "BONUS: People who work at a University OR a ResearchLab", """
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        PREFIX ex: <http://example.org/kg/>

        SELECT ?name ?org_name ?org_type
        WHERE {
            ?person a foaf:Person ;
                    foaf:name ?name ;
                    ex:worksAt ?org .
            ?org foaf:name ?org_name ;
                 ex:orgType ?org_type .
            FILTER (?org_type IN ("University", "ResearchLab"))
        }
        ORDER BY ?org_type ?name
    """)

    print("\nDone. All SPARQL queries executed successfully.")


if __name__ == "__main__":
    main()
