"""Create an RDF graph from scratch using rdflib.

Demonstrates:
- Defining custom and standard namespaces (FOAF, RDFS, OWL, SKOS)
- Adding triples about people, organizations, and concepts
- Serializing to Turtle, JSON-LD, and N-Triples formats
- Printing graph statistics
"""

import sys
from pathlib import Path

from rdflib import Graph, Literal, Namespace, URIRef, BNode
from rdflib.namespace import RDF, RDFS, OWL, FOAF, SKOS, XSD

# ── Path setup ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Custom Namespaces ────────────────────────────────────────────────
EX = Namespace("http://example.org/kg/")
TECH = Namespace("http://example.org/tech/")
ORG = Namespace("http://example.org/org/")


def create_rdf_graph() -> Graph:
    """Build an RDF graph about people, organizations, and AI concepts.

    Returns:
        A populated rdflib Graph.
    """
    g = Graph()

    # Bind namespace prefixes for readable serialization
    g.bind("ex", EX)
    g.bind("tech", TECH)
    g.bind("org", ORG)
    g.bind("foaf", FOAF)
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)
    g.bind("skos", SKOS)

    # ── People ────────────────────────────────────────────────────────

    # Geoffrey Hinton
    g.add((EX.GeoffreyHinton, RDF.type, FOAF.Person))
    g.add((EX.GeoffreyHinton, FOAF.name, Literal("Geoffrey Hinton")))
    g.add((EX.GeoffreyHinton, FOAF.title, Literal("Professor")))
    g.add((EX.GeoffreyHinton, EX.birthYear, Literal(1947, datatype=XSD.integer)))
    g.add((EX.GeoffreyHinton, EX.worksAt, ORG.UniversityOfToronto))
    g.add((EX.GeoffreyHinton, EX.workedAt, ORG.Google))
    g.add((EX.GeoffreyHinton, EX.researchArea, TECH.DeepLearning))
    g.add((EX.GeoffreyHinton, EX.researchArea, TECH.NeuralNetworks))
    g.add((EX.GeoffreyHinton, EX.invented, TECH.Backpropagation))
    g.add((EX.GeoffreyHinton, EX.wonAward, EX.NobelPrizePhysics2024))

    # Yann LeCun
    g.add((EX.YannLeCun, RDF.type, FOAF.Person))
    g.add((EX.YannLeCun, FOAF.name, Literal("Yann LeCun")))
    g.add((EX.YannLeCun, EX.worksAt, ORG.Meta))
    g.add((EX.YannLeCun, EX.worksAt, ORG.NYU))
    g.add((EX.YannLeCun, EX.researchArea, TECH.ComputerVision))
    g.add((EX.YannLeCun, EX.researchArea, TECH.DeepLearning))
    g.add((EX.YannLeCun, EX.invented, TECH.CNN))
    g.add((EX.YannLeCun, EX.wonAward, EX.TuringAward2018))

    # Yoshua Bengio
    g.add((EX.YoshuaBengio, RDF.type, FOAF.Person))
    g.add((EX.YoshuaBengio, FOAF.name, Literal("Yoshua Bengio")))
    g.add((EX.YoshuaBengio, EX.worksAt, ORG.Mila))
    g.add((EX.YoshuaBengio, EX.researchArea, TECH.DeepLearning))
    g.add((EX.YoshuaBengio, EX.researchArea, TECH.NLP))
    g.add((EX.YoshuaBengio, EX.wonAward, EX.TuringAward2018))

    # Fei-Fei Li
    g.add((EX.FeiFeiLi, RDF.type, FOAF.Person))
    g.add((EX.FeiFeiLi, FOAF.name, Literal("Fei-Fei Li")))
    g.add((EX.FeiFeiLi, EX.worksAt, ORG.Stanford))
    g.add((EX.FeiFeiLi, EX.researchArea, TECH.ComputerVision))
    g.add((EX.FeiFeiLi, EX.created, TECH.ImageNet))

    # Tim Berners-Lee
    g.add((EX.TimBernersLee, RDF.type, FOAF.Person))
    g.add((EX.TimBernersLee, FOAF.name, Literal("Tim Berners-Lee")))
    g.add((EX.TimBernersLee, EX.worksAt, ORG.MIT))
    g.add((EX.TimBernersLee, EX.invented, TECH.WorldWideWeb))
    g.add((EX.TimBernersLee, EX.invented, TECH.LinkedData))
    g.add((EX.TimBernersLee, EX.researchArea, TECH.SemanticWeb))

    # ── Organizations ─────────────────────────────────────────────────

    g.add((ORG.Google, RDF.type, FOAF.Organization))
    g.add((ORG.Google, FOAF.name, Literal("Google")))
    g.add((ORG.Google, EX.headquarters, Literal("Mountain View, CA")))

    g.add((ORG.Meta, RDF.type, FOAF.Organization))
    g.add((ORG.Meta, FOAF.name, Literal("Meta")))

    g.add((ORG.Stanford, RDF.type, FOAF.Organization))
    g.add((ORG.Stanford, FOAF.name, Literal("Stanford University")))
    g.add((ORG.Stanford, RDF.type, EX.University))

    g.add((ORG.MIT, RDF.type, FOAF.Organization))
    g.add((ORG.MIT, FOAF.name, Literal("MIT")))
    g.add((ORG.MIT, RDF.type, EX.University))

    g.add((ORG.UniversityOfToronto, RDF.type, FOAF.Organization))
    g.add((ORG.UniversityOfToronto, FOAF.name, Literal("University of Toronto")))
    g.add((ORG.UniversityOfToronto, RDF.type, EX.University))

    g.add((ORG.NYU, RDF.type, FOAF.Organization))
    g.add((ORG.NYU, FOAF.name, Literal("New York University")))
    g.add((ORG.NYU, RDF.type, EX.University))

    g.add((ORG.Mila, RDF.type, FOAF.Organization))
    g.add((ORG.Mila, FOAF.name, Literal("Mila - Quebec AI Institute")))

    # ── Technologies / Concepts ───────────────────────────────────────

    concepts = {
        TECH.DeepLearning: ("Deep Learning", TECH.MachineLearning),
        TECH.MachineLearning: ("Machine Learning", TECH.ArtificialIntelligence),
        TECH.ArtificialIntelligence: ("Artificial Intelligence", None),
        TECH.NeuralNetworks: ("Neural Networks", TECH.DeepLearning),
        TECH.CNN: ("Convolutional Neural Network", TECH.NeuralNetworks),
        TECH.Transformer: ("Transformer Architecture", TECH.NeuralNetworks),
        TECH.NLP: ("Natural Language Processing", TECH.ArtificialIntelligence),
        TECH.ComputerVision: ("Computer Vision", TECH.ArtificialIntelligence),
        TECH.KnowledgeGraph: ("Knowledge Graph", TECH.ArtificialIntelligence),
        TECH.SemanticWeb: ("Semantic Web", None),
        TECH.LinkedData: ("Linked Data", TECH.SemanticWeb),
        TECH.ImageNet: ("ImageNet", TECH.ComputerVision),
        TECH.Backpropagation: ("Backpropagation", TECH.NeuralNetworks),
        TECH.WorldWideWeb: ("World Wide Web", None),
    }

    for concept_uri, (label, broader) in concepts.items():
        g.add((concept_uri, RDF.type, SKOS.Concept))
        g.add((concept_uri, SKOS.prefLabel, Literal(label)))
        if broader:
            g.add((concept_uri, SKOS.broader, broader))
            g.add((broader, SKOS.narrower, concept_uri))

    # ── Awards ────────────────────────────────────────────────────────

    g.add((EX.TuringAward2018, RDF.type, EX.Award))
    g.add((EX.TuringAward2018, RDFS.label, Literal("Turing Award 2018")))
    g.add((EX.TuringAward2018, EX.year, Literal(2018, datatype=XSD.integer)))
    g.add((EX.TuringAward2018, EX.field, Literal("Deep Learning")))

    g.add((EX.NobelPrizePhysics2024, RDF.type, EX.Award))
    g.add((EX.NobelPrizePhysics2024, RDFS.label, Literal("Nobel Prize in Physics 2024")))
    g.add((EX.NobelPrizePhysics2024, EX.year, Literal(2024, datatype=XSD.integer)))

    # ── Relationships between people ──────────────────────────────────

    g.add((EX.GeoffreyHinton, FOAF.knows, EX.YannLeCun))
    g.add((EX.GeoffreyHinton, FOAF.knows, EX.YoshuaBengio))
    g.add((EX.YannLeCun, FOAF.knows, EX.YoshuaBengio))
    g.add((EX.GeoffreyHinton, EX.mentored, EX.YannLeCun))

    return g


def print_graph_stats(g: Graph) -> None:
    """Print statistics about the RDF graph."""
    print("=" * 60)
    print("RDF GRAPH STATISTICS")
    print("=" * 60)

    print(f"\nTotal triples: {len(g)}")

    # Count subjects, predicates, objects
    subjects = set(g.subjects())
    predicates = set(g.predicates())
    objects = set(g.objects())
    print(f"Unique subjects:   {len(subjects)}")
    print(f"Unique predicates: {len(predicates)}")
    print(f"Unique objects:    {len(objects)}")

    # Count by RDF type
    print("\n--- Entities by Type ---")
    type_counts: dict[str, int] = {}
    for s, _, o in g.triples((None, RDF.type, None)):
        type_label = str(o).split("/")[-1]
        type_counts[type_label] = type_counts.get(type_label, 0) + 1

    for type_name, count in sorted(type_counts.items()):
        print(f"  {type_name:25s}: {count}")

    # List all predicates used
    print("\n--- Predicates Used ---")
    for p in sorted(predicates, key=str):
        count = len(list(g.triples((None, p, None))))
        p_label = str(p).split("/")[-1].split("#")[-1]
        print(f"  {p_label:25s}: {count} triples")


def serialize_graph(g: Graph) -> None:
    """Serialize the graph to multiple formats and save to output/."""

    # Turtle format (most readable)
    turtle_path = OUTPUT_DIR / "knowledge_graph.ttl"
    g.serialize(destination=str(turtle_path), format="turtle")
    print(f"\nSaved Turtle format to {turtle_path}")

    # Print a preview
    turtle_str = g.serialize(format="turtle")
    lines = turtle_str.strip().split("\n")
    print(f"  ({len(lines)} lines)")
    print("  Preview (first 20 lines):")
    for line in lines[:20]:
        print(f"    {line}")
    print("    ...")

    # JSON-LD format
    jsonld_path = OUTPUT_DIR / "knowledge_graph.jsonld"
    g.serialize(destination=str(jsonld_path), format="json-ld")
    print(f"\nSaved JSON-LD format to {jsonld_path}")

    # N-Triples format
    nt_path = OUTPUT_DIR / "knowledge_graph.nt"
    g.serialize(destination=str(nt_path), format="nt")
    print(f"Saved N-Triples format to {nt_path}")

    # Show size comparison
    print("\n--- File Sizes ---")
    for path in [turtle_path, jsonld_path, nt_path]:
        size = path.stat().st_size
        print(f"  {path.name:30s}: {size:,} bytes")


def demo_graph_navigation(g: Graph) -> None:
    """Demonstrate navigating the RDF graph."""
    print("\n" + "=" * 60)
    print("GRAPH NAVIGATION DEMO")
    print("=" * 60)

    # Find all people
    print("\n--- All People ---")
    for person in g.subjects(RDF.type, FOAF.Person):
        name = g.value(person, FOAF.name)
        print(f"  {name}")

        # Find what they work at
        for org in g.objects(person, EX.worksAt):
            org_name = g.value(org, FOAF.name)
            print(f"    works at: {org_name}")

        # Find their research areas
        areas = [
            str(g.value(area, SKOS.prefLabel))
            for area in g.objects(person, EX.researchArea)
            if g.value(area, SKOS.prefLabel)
        ]
        if areas:
            print(f"    research: {', '.join(areas)}")

    # Navigate SKOS concept hierarchy
    print("\n--- Concept Hierarchy (via SKOS broader/narrower) ---")
    top_concepts = [
        s for s in g.subjects(RDF.type, SKOS.Concept)
        if not list(g.objects(s, SKOS.broader))
    ]
    for concept in top_concepts:
        _print_concept_tree(g, concept, indent=2)


def _print_concept_tree(g: Graph, concept, indent: int = 0) -> None:
    """Recursively print a concept tree."""
    label = g.value(concept, SKOS.prefLabel) or str(concept).split("/")[-1]
    print(f"{'  ' * indent}{label}")
    for narrower in g.objects(concept, SKOS.narrower):
        _print_concept_tree(g, narrower, indent + 1)


def main():
    print("Creating RDF knowledge graph...\n")
    g = create_rdf_graph()

    print_graph_stats(g)
    demo_graph_navigation(g)
    serialize_graph(g)

    print("\nDone.")


if __name__ == "__main__":
    main()
