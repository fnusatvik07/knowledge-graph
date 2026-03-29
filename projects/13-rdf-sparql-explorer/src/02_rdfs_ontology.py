"""Build an RDFS ontology with rdflib.

Demonstrates:
- Defining classes (Person, Organization, Technology)
- Defining properties with domain/range constraints
- Subclass relationships (Researcher subClassOf Person)
- Validating instances against the ontology
- How RDFS reasoning infers types
"""

import sys
from pathlib import Path

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD, FOAF

# ── Path setup ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Namespaces
EX = Namespace("http://example.org/ontology/")
DATA = Namespace("http://example.org/data/")


def build_ontology() -> Graph:
    """Build an RDFS ontology defining classes and properties.

    Returns:
        An rdflib Graph containing the ontology.
    """
    g = Graph()
    g.bind("ex", EX)
    g.bind("data", DATA)
    g.bind("foaf", FOAF)
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)

    # ── Classes ───────────────────────────────────────────────────────

    # Top-level classes
    g.add((EX.Person, RDF.type, RDFS.Class))
    g.add((EX.Person, RDFS.label, Literal("Person")))
    g.add((EX.Person, RDFS.comment, Literal("A human being.")))

    g.add((EX.Organization, RDF.type, RDFS.Class))
    g.add((EX.Organization, RDFS.label, Literal("Organization")))
    g.add((EX.Organization, RDFS.comment, Literal("A structured group of people.")))

    g.add((EX.Technology, RDF.type, RDFS.Class))
    g.add((EX.Technology, RDFS.label, Literal("Technology")))
    g.add((EX.Technology, RDFS.comment, Literal("A technology, tool, or methodology.")))

    g.add((EX.Award, RDF.type, RDFS.Class))
    g.add((EX.Award, RDFS.label, Literal("Award")))

    g.add((EX.Publication, RDF.type, RDFS.Class))
    g.add((EX.Publication, RDFS.label, Literal("Publication")))

    # Subclasses of Person
    g.add((EX.Researcher, RDF.type, RDFS.Class))
    g.add((EX.Researcher, RDFS.subClassOf, EX.Person))
    g.add((EX.Researcher, RDFS.label, Literal("Researcher")))
    g.add((EX.Researcher, RDFS.comment, Literal("A person who conducts research.")))

    g.add((EX.Professor, RDF.type, RDFS.Class))
    g.add((EX.Professor, RDFS.subClassOf, EX.Researcher))
    g.add((EX.Professor, RDFS.label, Literal("Professor")))

    g.add((EX.Student, RDF.type, RDFS.Class))
    g.add((EX.Student, RDFS.subClassOf, EX.Person))
    g.add((EX.Student, RDFS.label, Literal("Student")))

    g.add((EX.GradStudent, RDF.type, RDFS.Class))
    g.add((EX.GradStudent, RDFS.subClassOf, EX.Student))
    g.add((EX.GradStudent, RDFS.subClassOf, EX.Researcher))
    g.add((EX.GradStudent, RDFS.label, Literal("Graduate Student")))

    # Subclasses of Organization
    g.add((EX.University, RDF.type, RDFS.Class))
    g.add((EX.University, RDFS.subClassOf, EX.Organization))
    g.add((EX.University, RDFS.label, Literal("University")))

    g.add((EX.Company, RDF.type, RDFS.Class))
    g.add((EX.Company, RDFS.subClassOf, EX.Organization))
    g.add((EX.Company, RDFS.label, Literal("Company")))

    g.add((EX.ResearchLab, RDF.type, RDFS.Class))
    g.add((EX.ResearchLab, RDFS.subClassOf, EX.Organization))
    g.add((EX.ResearchLab, RDFS.label, Literal("Research Lab")))

    # Subclasses of Technology
    g.add((EX.Algorithm, RDF.type, RDFS.Class))
    g.add((EX.Algorithm, RDFS.subClassOf, EX.Technology))
    g.add((EX.Algorithm, RDFS.label, Literal("Algorithm")))

    g.add((EX.Framework, RDF.type, RDFS.Class))
    g.add((EX.Framework, RDFS.subClassOf, EX.Technology))
    g.add((EX.Framework, RDFS.label, Literal("Framework")))

    # ── Properties ────────────────────────────────────────────────────

    # worksAt: Person -> Organization
    g.add((EX.worksAt, RDF.type, RDF.Property))
    g.add((EX.worksAt, RDFS.domain, EX.Person))
    g.add((EX.worksAt, RDFS.range, EX.Organization))
    g.add((EX.worksAt, RDFS.label, Literal("works at")))

    # researchArea: Researcher -> Technology
    g.add((EX.researchArea, RDF.type, RDF.Property))
    g.add((EX.researchArea, RDFS.domain, EX.Researcher))
    g.add((EX.researchArea, RDFS.range, EX.Technology))
    g.add((EX.researchArea, RDFS.label, Literal("research area")))

    # invented: Person -> Technology
    g.add((EX.invented, RDF.type, RDF.Property))
    g.add((EX.invented, RDFS.domain, EX.Person))
    g.add((EX.invented, RDFS.range, EX.Technology))
    g.add((EX.invented, RDFS.label, Literal("invented")))

    # supervisedBy: Student -> Professor
    g.add((EX.supervisedBy, RDF.type, RDF.Property))
    g.add((EX.supervisedBy, RDFS.domain, EX.Student))
    g.add((EX.supervisedBy, RDFS.range, EX.Professor))
    g.add((EX.supervisedBy, RDFS.label, Literal("supervised by")))

    # authored: Researcher -> Publication
    g.add((EX.authored, RDF.type, RDF.Property))
    g.add((EX.authored, RDFS.domain, EX.Researcher))
    g.add((EX.authored, RDFS.range, EX.Publication))
    g.add((EX.authored, RDFS.label, Literal("authored")))

    # wonAward: Person -> Award
    g.add((EX.wonAward, RDF.type, RDF.Property))
    g.add((EX.wonAward, RDFS.domain, EX.Person))
    g.add((EX.wonAward, RDFS.range, EX.Award))
    g.add((EX.wonAward, RDFS.label, Literal("won award")))

    # usestech: Organization -> Technology
    g.add((EX.usesTech, RDF.type, RDF.Property))
    g.add((EX.usesTech, RDFS.domain, EX.Organization))
    g.add((EX.usesTech, RDFS.range, EX.Technology))
    g.add((EX.usesTech, RDFS.label, Literal("uses technology")))

    # partOf: Technology -> Technology (for hierarchies)
    g.add((EX.partOf, RDF.type, RDF.Property))
    g.add((EX.partOf, RDFS.domain, EX.Technology))
    g.add((EX.partOf, RDFS.range, EX.Technology))
    g.add((EX.partOf, RDFS.label, Literal("part of")))

    return g


def add_instance_data(g: Graph) -> Graph:
    """Add instance data (ABox) conforming to the ontology (TBox).

    Returns:
        The graph with instance data added.
    """
    # ── People instances ──────────────────────────────────────────────

    g.add((DATA.Hinton, RDF.type, EX.Professor))
    g.add((DATA.Hinton, RDFS.label, Literal("Geoffrey Hinton")))
    g.add((DATA.Hinton, EX.worksAt, DATA.UofT))
    g.add((DATA.Hinton, EX.researchArea, DATA.DeepLearning))
    g.add((DATA.Hinton, EX.invented, DATA.Backprop))
    g.add((DATA.Hinton, EX.wonAward, DATA.TuringAward))

    g.add((DATA.LeCun, RDF.type, EX.Professor))
    g.add((DATA.LeCun, RDFS.label, Literal("Yann LeCun")))
    g.add((DATA.LeCun, EX.worksAt, DATA.NYU))
    g.add((DATA.LeCun, EX.researchArea, DATA.ComputerVision))
    g.add((DATA.LeCun, EX.invented, DATA.CNN))

    g.add((DATA.AlicePhD, RDF.type, EX.GradStudent))
    g.add((DATA.AlicePhD, RDFS.label, Literal("Alice (PhD Student)")))
    g.add((DATA.AlicePhD, EX.worksAt, DATA.UofT))
    g.add((DATA.AlicePhD, EX.supervisedBy, DATA.Hinton))
    g.add((DATA.AlicePhD, EX.researchArea, DATA.NeuralArchSearch))

    # ── Organization instances ────────────────────────────────────────

    g.add((DATA.UofT, RDF.type, EX.University))
    g.add((DATA.UofT, RDFS.label, Literal("University of Toronto")))

    g.add((DATA.NYU, RDF.type, EX.University))
    g.add((DATA.NYU, RDFS.label, Literal("New York University")))

    g.add((DATA.GoogleBrain, RDF.type, EX.ResearchLab))
    g.add((DATA.GoogleBrain, RDFS.label, Literal("Google Brain")))
    g.add((DATA.GoogleBrain, EX.usesTech, DATA.TensorFlow))

    # ── Technology instances ──────────────────────────────────────────

    g.add((DATA.DeepLearning, RDF.type, EX.Technology))
    g.add((DATA.DeepLearning, RDFS.label, Literal("Deep Learning")))

    g.add((DATA.ComputerVision, RDF.type, EX.Technology))
    g.add((DATA.ComputerVision, RDFS.label, Literal("Computer Vision")))

    g.add((DATA.Backprop, RDF.type, EX.Algorithm))
    g.add((DATA.Backprop, RDFS.label, Literal("Backpropagation")))
    g.add((DATA.Backprop, EX.partOf, DATA.DeepLearning))

    g.add((DATA.CNN, RDF.type, EX.Algorithm))
    g.add((DATA.CNN, RDFS.label, Literal("Convolutional Neural Network")))
    g.add((DATA.CNN, EX.partOf, DATA.DeepLearning))

    g.add((DATA.NeuralArchSearch, RDF.type, EX.Algorithm))
    g.add((DATA.NeuralArchSearch, RDFS.label, Literal("Neural Architecture Search")))
    g.add((DATA.NeuralArchSearch, EX.partOf, DATA.DeepLearning))

    g.add((DATA.TensorFlow, RDF.type, EX.Framework))
    g.add((DATA.TensorFlow, RDFS.label, Literal("TensorFlow")))

    # ── Award instances ───────────────────────────────────────────────

    g.add((DATA.TuringAward, RDF.type, EX.Award))
    g.add((DATA.TuringAward, RDFS.label, Literal("Turing Award")))

    # ── Publication instances ─────────────────────────────────────────

    g.add((DATA.AttentionPaper, RDF.type, EX.Publication))
    g.add((DATA.AttentionPaper, RDFS.label, Literal("Attention Is All You Need")))
    g.add((DATA.Hinton, EX.authored, DATA.AttentionPaper))

    return g


def validate_instances(g: Graph) -> None:
    """Validate instances against domain/range constraints.

    Checks that property usage respects declared domain and range.
    """
    print("\n" + "=" * 60)
    print("ONTOLOGY VALIDATION")
    print("=" * 60)

    issues = []
    valid_count = 0

    # Get all properties with domain/range
    for prop in g.subjects(RDF.type, RDF.Property):
        prop_label = g.value(prop, RDFS.label) or str(prop).split("/")[-1]
        domain = g.value(prop, RDFS.domain)
        range_ = g.value(prop, RDFS.range)

        # Check all usage of this property
        for s, _, o in g.triples((None, prop, None)):
            # Check domain
            if domain:
                s_types = set(g.objects(s, RDF.type))
                # Include superclasses via RDFS reasoning
                all_types = set()
                for t in s_types:
                    all_types.add(t)
                    for super_class in g.transitive_objects(t, RDFS.subClassOf):
                        all_types.add(super_class)

                if domain not in all_types and all_types:
                    s_label = g.value(s, RDFS.label) or str(s).split("/")[-1]
                    issues.append(
                        f"  Domain violation: {s_label} uses '{prop_label}' "
                        f"but is not a {str(domain).split('/')[-1]} "
                        f"(types: {[str(t).split('/')[-1] for t in s_types]})"
                    )
                else:
                    valid_count += 1

            # Check range (only for URI objects)
            if range_ and isinstance(o, URIRef):
                o_types = set(g.objects(o, RDF.type))
                all_types = set()
                for t in o_types:
                    all_types.add(t)
                    for super_class in g.transitive_objects(t, RDFS.subClassOf):
                        all_types.add(super_class)

                if range_ not in all_types and all_types:
                    o_label = g.value(o, RDFS.label) or str(o).split("/")[-1]
                    issues.append(
                        f"  Range violation: '{prop_label}' points to {o_label} "
                        f"which is not a {str(range_).split('/')[-1]} "
                        f"(types: {[str(t).split('/')[-1] for t in o_types]})"
                    )
                else:
                    valid_count += 1

    print(f"\nValid property usages: {valid_count}")
    if issues:
        print(f"\nIssues found ({len(issues)}):")
        for issue in issues:
            print(issue)
    else:
        print("No validation issues found.")


def demonstrate_rdfs_reasoning(g: Graph) -> None:
    """Show how RDFS reasoning infers additional types."""
    print("\n" + "=" * 60)
    print("RDFS REASONING DEMO")
    print("=" * 60)

    # Show subclass hierarchy
    print("\n--- Class Hierarchy ---")
    top_classes = [
        s for s in g.subjects(RDF.type, RDFS.Class)
        if not list(g.objects(s, RDFS.subClassOf))
    ]

    def print_class_tree(cls, indent=0):
        label = g.value(cls, RDFS.label) or str(cls).split("/")[-1]
        print(f"{'  ' * indent}{label}")
        for sub in g.subjects(RDFS.subClassOf, cls):
            print_class_tree(sub, indent + 1)

    for cls in sorted(top_classes, key=str):
        print_class_tree(cls)

    # RDFS inference: GradStudent is a Researcher AND a Student
    print("\n--- Type Inference for GradStudent ---")
    print("AlicePhD is declared as: GradStudent")
    print("Via RDFS subClassOf reasoning, AlicePhD is also:")
    alice_type = EX.GradStudent
    inferred = set()
    for super_class in g.transitive_objects(alice_type, RDFS.subClassOf):
        if super_class != alice_type:
            label = g.value(super_class, RDFS.label) or str(super_class).split("/")[-1]
            inferred.add(label)
    for t in sorted(inferred, key=str):
        print(f"  - {t}")

    # Domain inference: using researchArea implies the subject is a Researcher
    print("\n--- Domain Inference ---")
    domain = g.value(EX.researchArea, RDFS.domain)
    if domain:
        domain_label = g.value(domain, RDFS.label) or str(domain).split("/")[-1]
        print(f"Property 'researchArea' has domain: {domain_label}")
        print("Therefore, anyone using 'researchArea' can be inferred to be a Researcher:")
        for s in g.subjects(EX.researchArea, None):
            s_label = g.value(s, RDFS.label) or str(s).split("/")[-1]
            s_types = [
                str(g.value(t, RDFS.label) or t).split("/")[-1]
                for t in g.objects(s, RDF.type)
            ]
            print(f"  {s_label} (declared types: {s_types}) -> inferred Researcher")


def main():
    print("Building RDFS ontology...\n")
    g = build_ontology()

    ontology_triples = len(g)
    print(f"Ontology (TBox): {ontology_triples} triples")

    # Show ontology structure
    classes = list(g.subjects(RDF.type, RDFS.Class))
    properties = list(g.subjects(RDF.type, RDF.Property))
    print(f"  Classes: {len(classes)}")
    print(f"  Properties: {len(properties)}")

    print("\n--- Properties with Domain/Range ---")
    for prop in sorted(properties, key=str):
        label = g.value(prop, RDFS.label) or str(prop).split("/")[-1]
        domain = g.value(prop, RDFS.domain)
        range_ = g.value(prop, RDFS.range)
        d_label = str(domain).split("/")[-1] if domain else "?"
        r_label = str(range_).split("/")[-1] if range_ else "?"
        print(f"  {label:20s}: {d_label} -> {r_label}")

    # Add instance data
    print("\nAdding instance data...")
    g = add_instance_data(g)
    instance_triples = len(g) - ontology_triples
    print(f"Instance data (ABox): {instance_triples} triples")
    print(f"Total triples: {len(g)}")

    # Validate
    validate_instances(g)

    # Demonstrate reasoning
    demonstrate_rdfs_reasoning(g)

    # Save ontology
    output_path = OUTPUT_DIR / "ontology.ttl"
    g.serialize(destination=str(output_path), format="turtle")
    print(f"\nOntology saved to {output_path}")


if __name__ == "__main__":
    main()
