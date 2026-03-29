"""Convert between property graph (NetworkX) and RDF (rdflib).

Demonstrates:
- Loading a NetworkX property graph
- Converting to RDF triples
- Running SPARQL on the converted graph
- Converting RDF back to property graph
- Comparing query expressiveness: Cypher vs SPARQL
- Discussing trade-offs between the two approaches
"""

import json
import sys
from pathlib import Path

import networkx as nx
from rdflib import Graph as RDFGraph, Literal, Namespace, URIRef, BNode
from rdflib.namespace import RDF, RDFS, XSD, FOAF

# ── Path setup ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Namespace for converted entities
EX = Namespace("http://example.org/converted/")
PROP = Namespace("http://example.org/property/")
REL = Namespace("http://example.org/relationship/")


def build_sample_property_graph() -> nx.DiGraph:
    """Build a sample property graph about AI researchers.

    This represents what you might find in Neo4j or a prior project.
    """
    G = nx.DiGraph()

    # People (with properties on the node)
    G.add_node("hinton", type="Person", name="Geoffrey Hinton", birth_year=1947, title="Professor")
    G.add_node("lecun", type="Person", name="Yann LeCun", birth_year=1960, title="Professor")
    G.add_node("bengio", type="Person", name="Yoshua Bengio", birth_year=1964, title="Professor")
    G.add_node("ng", type="Person", name="Andrew Ng", birth_year=1976, title="Professor")

    # Organizations
    G.add_node("utoronto", type="Organization", name="University of Toronto", org_type="University")
    G.add_node("meta", type="Organization", name="Meta", org_type="Company")
    G.add_node("mila", type="Organization", name="Mila", org_type="ResearchLab")
    G.add_node("stanford", type="Organization", name="Stanford University", org_type="University")

    # Technologies
    G.add_node("dl", type="Technology", name="Deep Learning", year_invented=2006)
    G.add_node("cnn", type="Technology", name="Convolutional Neural Networks", year_invented=1989)
    G.add_node("nlp", type="Technology", name="Natural Language Processing")

    # Relationships (with properties on the edge)
    G.add_edge("hinton", "utoronto", type="WORKS_AT", since=1987, role="Professor")
    G.add_edge("lecun", "meta", type="WORKS_AT", since=2013, role="Chief AI Scientist")
    G.add_edge("bengio", "mila", type="WORKS_AT", since=1993, role="Director")
    G.add_edge("ng", "stanford", type="WORKS_AT", since=2002, role="Professor")

    G.add_edge("hinton", "dl", type="RESEARCHES", contribution="Backpropagation")
    G.add_edge("lecun", "cnn", type="RESEARCHES", contribution="LeNet")
    G.add_edge("lecun", "dl", type="RESEARCHES", contribution="Self-supervised learning")
    G.add_edge("bengio", "nlp", type="RESEARCHES", contribution="Attention mechanisms")
    G.add_edge("bengio", "dl", type="RESEARCHES", contribution="Generative models")

    G.add_edge("hinton", "lecun", type="MENTORED", year=1987)
    G.add_edge("hinton", "bengio", type="COLLABORATES_WITH")
    G.add_edge("lecun", "bengio", type="COLLABORATES_WITH")

    return G


def property_graph_to_rdf(G: nx.DiGraph) -> RDFGraph:
    """Convert a NetworkX property graph to an RDF graph.

    Strategy:
    - Each node becomes a URI: ex:{node_id}
    - Node type becomes rdf:type with a class URI
    - Node properties become property triples
    - Each edge becomes a triple: subject -> predicate -> object
    - Edge properties are represented using reification (intermediate nodes)

    Returns:
        An rdflib Graph with the converted data.
    """
    rdf = RDFGraph()
    rdf.bind("ex", EX)
    rdf.bind("prop", PROP)
    rdf.bind("rel", REL)
    rdf.bind("foaf", FOAF)

    # Convert nodes
    for node_id, attrs in G.nodes(data=True):
        node_uri = EX[node_id]

        # Type
        node_type = attrs.get("type")
        if node_type:
            rdf.add((node_uri, RDF.type, EX[node_type]))

        # Properties
        for key, value in attrs.items():
            if key == "type":
                continue
            if key == "name":
                rdf.add((node_uri, RDFS.label, Literal(str(value))))
                rdf.add((node_uri, FOAF.name, Literal(str(value))))
            elif isinstance(value, int):
                rdf.add((node_uri, PROP[key], Literal(value, datatype=XSD.integer)))
            elif isinstance(value, float):
                rdf.add((node_uri, PROP[key], Literal(value, datatype=XSD.float)))
            else:
                rdf.add((node_uri, PROP[key], Literal(str(value))))

    # Convert edges
    edge_counter = 0
    for src, tgt, attrs in G.edges(data=True):
        src_uri = EX[src]
        tgt_uri = EX[tgt]
        edge_type = attrs.get("type", "RELATED_TO")

        # Simple triple for the relationship
        rdf.add((src_uri, REL[edge_type], tgt_uri))

        # If edge has properties beyond type, use reification
        edge_props = {k: v for k, v in attrs.items() if k != "type"}
        if edge_props:
            edge_counter += 1
            stmt = EX[f"edge_{edge_counter}"]
            rdf.add((stmt, RDF.type, RDF.Statement))
            rdf.add((stmt, RDF.subject, src_uri))
            rdf.add((stmt, RDF.predicate, REL[edge_type]))
            rdf.add((stmt, RDF.object, tgt_uri))
            for key, value in edge_props.items():
                if isinstance(value, int):
                    rdf.add((stmt, PROP[key], Literal(value, datatype=XSD.integer)))
                else:
                    rdf.add((stmt, PROP[key], Literal(str(value))))

    return rdf


def rdf_to_property_graph(rdf: RDFGraph) -> nx.DiGraph:
    """Convert an RDF graph back to a NetworkX property graph.

    Strategy:
    - Each unique subject/object URI becomes a node
    - rdf:type triples set the node type
    - rdfs:label / foaf:name set the node name
    - Other property triples (with literal objects) become node attributes
    - Triples with URI objects in the rel: namespace become edges
    - Reified statements recover edge properties

    Returns:
        A NetworkX DiGraph.
    """
    G = nx.DiGraph()

    # First pass: identify all entity URIs (subjects and objects that are URIs)
    entity_uris = set()
    for s, p, o in rdf:
        if isinstance(s, URIRef) and str(s).startswith(str(EX)):
            # Skip reification nodes
            if not str(s).split("/")[-1].startswith("edge_"):
                entity_uris.add(s)
        if isinstance(o, URIRef) and str(o).startswith(str(EX)):
            if not str(o).split("/")[-1].startswith("edge_"):
                entity_uris.add(o)

    # Second pass: build nodes with attributes
    for uri in entity_uris:
        node_id = str(uri).split("/")[-1]
        attrs = {}

        for p, o in rdf.predicate_objects(uri):
            p_str = str(p)
            if p == RDF.type and str(o).startswith(str(EX)):
                attrs["type"] = str(o).split("/")[-1]
            elif p == RDFS.label or p == FOAF.name:
                attrs["name"] = str(o)
            elif str(p).startswith(str(PROP)):
                prop_name = str(p).split("/")[-1]
                attrs[prop_name] = o.toPython() if isinstance(o, Literal) else str(o)

        G.add_node(node_id, **attrs)

    # Third pass: build edges from rel: triples
    for s, p, o in rdf:
        if str(p).startswith(str(REL)) and isinstance(o, URIRef):
            src_id = str(s).split("/")[-1]
            tgt_id = str(o).split("/")[-1]
            edge_type = str(p).split("/")[-1]
            if src_id in G.nodes and tgt_id in G.nodes:
                G.add_edge(src_id, tgt_id, type=edge_type)

    # Fourth pass: recover edge properties from reified statements
    for stmt in rdf.subjects(RDF.type, RDF.Statement):
        subj = rdf.value(stmt, RDF.subject)
        pred = rdf.value(stmt, RDF.predicate)
        obj = rdf.value(stmt, RDF.object)
        if subj and obj:
            src_id = str(subj).split("/")[-1]
            tgt_id = str(obj).split("/")[-1]
            if G.has_edge(src_id, tgt_id):
                for p, o in rdf.predicate_objects(stmt):
                    if str(p).startswith(str(PROP)):
                        prop_name = str(p).split("/")[-1]
                        G.edges[src_id, tgt_id][prop_name] = (
                            o.toPython() if isinstance(o, Literal) else str(o)
                        )

    return G


def run_sparql_on_converted(rdf: RDFGraph) -> None:
    """Run SPARQL queries on the converted RDF graph."""
    print("\n" + "=" * 60)
    print("  SPARQL QUERIES ON CONVERTED GRAPH")
    print("=" * 60)

    # Query 1: Find all people and where they work
    query1 = """
    PREFIX ex: <http://example.org/converted/>
    PREFIX rel: <http://example.org/relationship/>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>

    SELECT ?person_name ?org_name
    WHERE {
        ?person a ex:Person ;
                foaf:name ?person_name .
        ?person rel:WORKS_AT ?org .
        ?org foaf:name ?org_name .
    }
    ORDER BY ?person_name
    """
    print("\n  Query: People and their organizations")
    for row in rdf.query(query1):
        print(f"    {row[0]:25s} works at {row[1]}")

    # Query 2: Find all research contributions
    query2 = """
    PREFIX ex: <http://example.org/converted/>
    PREFIX rel: <http://example.org/relationship/>
    PREFIX prop: <http://example.org/property/>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

    SELECT ?person_name ?tech_name ?contribution
    WHERE {
        ?person a ex:Person ;
                foaf:name ?person_name .
        ?person rel:RESEARCHES ?tech .
        ?tech foaf:name ?tech_name .
        OPTIONAL {
            ?stmt rdf:subject ?person ;
                  rdf:predicate rel:RESEARCHES ;
                  rdf:object ?tech ;
                  prop:contribution ?contribution .
        }
    }
    ORDER BY ?person_name
    """
    print("\n  Query: Research contributions (with reified edge properties)")
    for row in rdf.query(query2):
        contribution = row[2] if row[2] else "(none)"
        print(f"    {row[0]:25s} researches {row[1]:35s} contribution: {contribution}")


def compare_query_expressiveness() -> None:
    """Compare Cypher and SPARQL for equivalent queries."""
    print("\n" + "=" * 60)
    print("  CYPHER vs SPARQL COMPARISON")
    print("=" * 60)

    comparisons = [
        {
            "description": "Find all people and their organizations",
            "cypher": """MATCH (p:Person)-[:WORKS_AT]->(o:Organization)
RETURN p.name, o.name""",
            "sparql": """SELECT ?name ?org
WHERE {
  ?p a ex:Person ; foaf:name ?name ;
     rel:WORKS_AT ?o .
  ?o foaf:name ?org .
}""",
        },
        {
            "description": "Find people researching Deep Learning",
            "cypher": """MATCH (p:Person)-[:RESEARCHES]->(t:Technology {name: "Deep Learning"})
RETURN p.name""",
            "sparql": """SELECT ?name
WHERE {
  ?p a ex:Person ; foaf:name ?name ;
     rel:RESEARCHES ?t .
  ?t foaf:name "Deep Learning" .
}""",
        },
        {
            "description": "Find mentoring chains (transitive)",
            "cypher": """MATCH path = (a:Person)-[:MENTORED*1..3]->(b:Person)
RETURN a.name, b.name, length(path)""",
            "sparql": """SELECT ?a_name ?b_name
WHERE {
  ?a foaf:name ?a_name .
  ?a rel:MENTORED+ ?b .
  ?b foaf:name ?b_name .
}""",
        },
        {
            "description": "Find people with edge properties (role at org)",
            "cypher": """MATCH (p:Person)-[r:WORKS_AT]->(o:Organization)
RETURN p.name, o.name, r.role, r.since""",
            "sparql": """SELECT ?name ?org ?role ?since
WHERE {
  ?p rel:WORKS_AT ?o .
  ?p foaf:name ?name . ?o foaf:name ?org .
  ?stmt rdf:subject ?p ; rdf:predicate rel:WORKS_AT ;
        rdf:object ?o ; prop:role ?role .
  OPTIONAL { ?stmt prop:since ?since }
}""",
            "note": "Edge properties require reification in RDF, making queries more verbose.",
        },
    ]

    for i, comp in enumerate(comparisons, 1):
        print(f"\n  {i}. {comp['description']}")
        print(f"\n  Cypher:")
        for line in comp["cypher"].split("\n"):
            print(f"    {line}")
        print(f"\n  SPARQL:")
        for line in comp["sparql"].split("\n"):
            print(f"    {line}")
        if "note" in comp:
            print(f"\n  Note: {comp['note']}")


def discuss_tradeoffs() -> None:
    """Print a comparison of property graph vs RDF trade-offs."""
    print("\n" + "=" * 60)
    print("  TRADE-OFFS: PROPERTY GRAPHS vs RDF")
    print("=" * 60)

    print("""
  PROPERTY GRAPHS (Neo4j, NetworkX)
  + Edge properties are first-class (natural for weighted/attributed edges)
  + Cypher queries are often more concise and intuitive
  + Better performance for traversal-heavy workloads
  + Schema-optional: flexible and easy to evolve
  - No global identifiers (entities are local to the database)
  - No standard interchange format (until GQL standard)
  - Limited reasoning / inference capabilities

  RDF (rdflib, SPARQL endpoints)
  + Global identifiers (URIs) enable linked data across the web
  + SPARQL is a W3C standard with wide tool support
  + RDFS/OWL reasoning can infer new facts from existing data
  + Standard serialization formats (Turtle, JSON-LD, N-Triples)
  + Federated queries can span multiple endpoints
  - Edge properties require reification (verbose)
  - Steeper learning curve for SPARQL
  - Can be slower for deep traversals

  WHEN TO USE WHICH?
  - Property graph: application-specific KGs, recommendation engines,
    fraud detection, social networks
  - RDF: data integration, linked data, ontology-driven domains,
    biomedical KGs, public knowledge bases (Wikidata, DBpedia)
  - Both: many real-world systems use property graphs internally
    and export as RDF for interoperability
    """)


def main():
    print("Property Graph vs RDF Conversion Demo\n")

    # Build property graph
    print("Building sample property graph...")
    G = build_sample_property_graph()
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")

    # Show original property graph
    print("\n--- Original Property Graph ---")
    for node_id, attrs in G.nodes(data=True):
        print(f"  ({node_id}) {attrs}")
    print()
    for src, tgt, attrs in G.edges(data=True):
        print(f"  ({src})-[{attrs}]->({tgt})")

    # Convert to RDF
    print("\n--- Converting to RDF ---")
    rdf = property_graph_to_rdf(G)
    print(f"  RDF triples: {len(rdf)}")

    # Save the converted RDF
    rdf_path = OUTPUT_DIR / "converted_from_property_graph.ttl"
    rdf.serialize(destination=str(rdf_path), format="turtle")
    print(f"  Saved to {rdf_path}")

    # Show some RDF triples
    print("\n  Sample triples:")
    for i, (s, p, o) in enumerate(sorted(rdf, key=lambda x: str(x[0]))):
        if i >= 15:
            print(f"    ... ({len(rdf) - 15} more)")
            break
        s_label = str(s).split("/")[-1]
        p_label = str(p).split("/")[-1].split("#")[-1]
        o_label = str(o).split("/")[-1] if isinstance(o, URIRef) else str(o)
        print(f"    {s_label:15s}  {p_label:15s}  {o_label}")

    # Run SPARQL on converted graph
    run_sparql_on_converted(rdf)

    # Convert back to property graph
    print("\n--- Converting Back to Property Graph ---")
    G_back = rdf_to_property_graph(rdf)
    print(f"  Nodes: {G_back.number_of_nodes()}")
    print(f"  Edges: {G_back.number_of_edges()}")

    # Compare
    print("\n--- Round-Trip Comparison ---")
    print(f"  Original:  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Converted: {G_back.number_of_nodes()} nodes, {G_back.number_of_edges()} edges")

    # Check if nodes match
    original_nodes = set(G.nodes())
    converted_nodes = set(G_back.nodes())
    print(f"  Nodes preserved: {original_nodes == converted_nodes}")
    if original_nodes != converted_nodes:
        print(f"    Missing: {original_nodes - converted_nodes}")
        print(f"    Extra: {converted_nodes - original_nodes}")

    # Compare queries
    compare_query_expressiveness()

    # Trade-offs discussion
    discuss_tradeoffs()


if __name__ == "__main__":
    main()
