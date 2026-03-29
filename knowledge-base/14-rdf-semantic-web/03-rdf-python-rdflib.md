# Hands-On RDF with Python rdflib

## Overview

rdflib is the standard Python library for working with RDF. It provides tools for creating, parsing, serializing, and querying RDF graphs. This tutorial covers practical usage from basic triple manipulation to building and querying a complete knowledge graph.

- **Documentation**: https://rdflib.readthedocs.io/
- **Repository**: https://github.com/RDFLib/rdflib

---

## Installation

```bash
pip install rdflib

# Optional: JSON-LD support
pip install rdflib-jsonld

# Verify
python -c "import rdflib; print(rdflib.__version__)"
```

---

## Creating RDF Graphs

### Basic Graph Creation

```python
from rdflib import Graph, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, XSD, FOAF, OWL

# Create an empty graph
g = Graph()

# Add triples using URIRef (full URIs)
subject = URIRef("http://example.org/person/einstein")
predicate = URIRef("http://xmlns.com/foaf/0.1/name")
obj = Literal("Albert Einstein")

g.add((subject, predicate, obj))

print(f"Graph has {len(g)} triples")
```

### Using Namespaces

Namespaces make code much cleaner:

```python
from rdflib import Namespace

# Define custom namespaces
EX = Namespace("http://example.org/")
PERSON = Namespace("http://example.org/person/")
PLACE = Namespace("http://example.org/place/")

# Bind prefixes for readable serialization
g.bind("ex", EX)
g.bind("person", PERSON)
g.bind("place", PLACE)
g.bind("foaf", FOAF)

# Add triples using namespace shortcuts
g.add((PERSON.einstein, RDF.type, EX.Scientist))
g.add((PERSON.einstein, FOAF.name, Literal("Albert Einstein")))
g.add((PERSON.einstein, EX.birthPlace, PLACE.Ulm))
g.add((PERSON.einstein, EX.birthDate, Literal("1879-03-14", datatype=XSD.date)))
g.add((PERSON.einstein, EX.field, EX.Physics))
g.add((PERSON.einstein, EX.field, EX.Mathematics))  # multi-valued
```

### Typed Literals

```python
# String (default)
g.add((PERSON.einstein, FOAF.name, Literal("Albert Einstein")))

# Language-tagged string
g.add((PERSON.einstein, RDFS.label, Literal("Albert Einstein", lang="en")))
g.add((PERSON.einstein, RDFS.label, Literal("Albert Einstein", lang="de")))

# Integer
g.add((PERSON.einstein, EX.age, Literal(76, datatype=XSD.integer)))

# Float
g.add((PERSON.einstein, EX.hIndex, Literal(95.5, datatype=XSD.float)))

# Date
g.add((PERSON.einstein, EX.birthDate, Literal("1879-03-14", datatype=XSD.date)))

# Boolean
g.add((PERSON.einstein, EX.isAlive, Literal(False, datatype=XSD.boolean)))
```

### Blank Nodes

For anonymous resources (e.g., addresses without a specific URI):

```python
address = BNode()
g.add((PERSON.einstein, EX.address, address))
g.add((address, EX.street, Literal("112 Mercer Street")))
g.add((address, EX.city, PLACE.Princeton))
g.add((address, EX.country, PLACE.USA))
```

---

## Serialization

### To Turtle

```python
turtle_str = g.serialize(format="turtle")
print(turtle_str)

# Save to file
g.serialize(destination="knowledge_graph.ttl", format="turtle")
```

### To JSON-LD

```python
jsonld_str = g.serialize(format="json-ld", indent=2)
print(jsonld_str)

g.serialize(destination="knowledge_graph.jsonld", format="json-ld")
```

### To N-Triples

```python
nt_str = g.serialize(format="nt")
g.serialize(destination="knowledge_graph.nt", format="nt")
```

### To RDF/XML

```python
xml_str = g.serialize(format="xml")
g.serialize(destination="knowledge_graph.rdf", format="xml")
```

---

## Parsing RDF Data

### From File

```python
g = Graph()

# Auto-detect format from extension
g.parse("knowledge_graph.ttl")

# Explicit format
g.parse("data.rdf", format="xml")
g.parse("data.jsonld", format="json-ld")
g.parse("data.nt", format="nt")

print(f"Loaded {len(g)} triples")
```

### From URL

```python
g = Graph()
g.parse("http://dbpedia.org/resource/Albert_Einstein", format="xml")
print(f"Loaded {len(g)} triples about Einstein from DBpedia")
```

### From String

```python
turtle_data = """
@prefix ex: <http://example.org/> .
@prefix foaf: <http://xmlns.com/foaf/0.1/> .

ex:alice foaf:name "Alice" ;
         foaf:knows ex:bob .
ex:bob foaf:name "Bob" .
"""

g = Graph()
g.parse(data=turtle_data, format="turtle")
```

---

## Querying: Triple Patterns

### Iterate All Triples

```python
for s, p, o in g:
    print(f"{s} -- {p} --> {o}")
```

### Pattern Matching with triples()

```python
# All triples about Einstein
for s, p, o in g.triples((PERSON.einstein, None, None)):
    print(f"  {p}: {o}")

# All scientists
for s, p, o in g.triples((None, RDF.type, EX.Scientist)):
    print(f"  Scientist: {s}")

# All birth places
for s, p, o in g.triples((None, EX.birthPlace, None)):
    print(f"  {s} born in {o}")
```

### Convenience Methods

```python
# Get all objects for a subject+predicate
for obj in g.objects(PERSON.einstein, FOAF.name):
    print(f"Name: {obj}")

# Get all subjects for a predicate+object
for subj in g.subjects(RDF.type, EX.Scientist):
    print(f"Scientist: {subj}")

# Get all predicates between two resources
for pred in g.predicates(PERSON.einstein, PLACE.Ulm):
    print(f"Relation: {pred}")

# Check if a triple exists
if (PERSON.einstein, RDF.type, EX.Scientist) in g:
    print("Einstein is a scientist!")
```

---

## SPARQL Queries on Local Graphs

rdflib supports full SPARQL 1.1:

### SELECT

```python
results = g.query("""
    PREFIX ex: <http://example.org/>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>

    SELECT ?name ?birthPlace WHERE {
        ?person a ex:Scientist ;
                foaf:name ?name ;
                ex:birthPlace ?place .
        BIND(STRAFTER(STR(?place), "http://example.org/place/") AS ?birthPlace)
    }
""")

for row in results:
    print(f"{row.name} -- born in {row.birthPlace}")
```

### ASK

```python
result = g.query("""
    ASK {
        <http://example.org/person/einstein> a <http://example.org/Scientist>
    }
""")
print(f"Einstein is a scientist: {bool(result)}")
```

### CONSTRUCT

```python
new_graph = g.query("""
    PREFIX ex: <http://example.org/>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>

    CONSTRUCT {
        ?person ex:summary ?name .
    }
    WHERE {
        ?person a ex:Scientist ;
                foaf:name ?name .
    }
""").graph

print(new_graph.serialize(format="turtle"))
```

### Parameterized Queries with initBindings

```python
results = g.query(
    """
    SELECT ?predicate ?object WHERE {
        ?entity ?predicate ?object .
    }
    """,
    initBindings={"entity": PERSON.einstein}
)

for row in results:
    print(f"  {row.predicate}: {row.object}")
```

---

## Loading External RDF Data

### From DBpedia

```python
g = Graph()
g.parse("http://dbpedia.org/data/Python_(programming_language).ttl", format="turtle")

# Query the loaded data
results = g.query("""
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>

    SELECT ?name ?abstract WHERE {
        ?s foaf:name ?name ;
           dbo:abstract ?abstract .
        FILTER(LANG(?abstract) = "en")
    }
""")

for row in results:
    print(f"Name: {row.name}")
    print(f"Abstract: {str(row.abstract)[:200]}...")
```

### From Wikidata (via SPARQLWrapper)

```python
from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql.setQuery("""
    SELECT ?item ?label ?desc WHERE {
        ?item wdt:P31 wd:Q9143 .
        ?item rdfs:label ?label .
        ?item schema:description ?desc .
        FILTER(LANG(?label) = "en")
        FILTER(LANG(?desc) = "en")
    }
    LIMIT 20
""")
sparql.setReturnFormat(JSON)
results = sparql.query().convert()

# Convert to rdflib graph
g = Graph()
EX = Namespace("http://example.org/lang/")
g.bind("ex", EX)

for r in results["results"]["bindings"]:
    item_uri = URIRef(r["item"]["value"])
    g.add((item_uri, RDFS.label, Literal(r["label"]["value"], lang="en")))
    g.add((item_uri, RDFS.comment, Literal(r["desc"]["value"], lang="en")))
    g.add((item_uri, RDF.type, EX.ProgrammingLanguage))

print(f"Built graph with {len(g)} triples")
```

---

## Converting Between Property Graph and RDF

### Property Graph (dict) to RDF

```python
# A property graph as Python dicts
nodes = [
    {"id": "python", "type": "Language", "name": "Python", "year": 1991},
    {"id": "guido", "type": "Person", "name": "Guido van Rossum"},
    {"id": "google", "type": "Company", "name": "Google"},
]

edges = [
    {"source": "python", "target": "guido", "type": "created_by"},
    {"source": "guido", "target": "google", "type": "works_at"},
]

# Convert to RDF
g = Graph()
EX = Namespace("http://example.org/")
g.bind("ex", EX)

for node in nodes:
    uri = EX[node["id"]]
    g.add((uri, RDF.type, EX[node["type"]]))
    g.add((uri, FOAF.name, Literal(node["name"])))
    if "year" in node:
        g.add((uri, EX.year, Literal(node["year"], datatype=XSD.integer)))

for edge in edges:
    g.add((EX[edge["source"]], EX[edge["type"]], EX[edge["target"]]))

print(g.serialize(format="turtle"))
```

### RDF to Property Graph (NetworkX)

```python
import networkx as nx

G = nx.DiGraph()

# Add nodes with properties
for s in set(g.subjects()):
    props = {}
    for p, o in g.predicate_objects(s):
        key = str(p).split("/")[-1].split("#")[-1]
        if isinstance(o, Literal):
            props[key] = str(o)
    G.add_node(str(s), **props)

# Add edges
for s, p, o in g:
    if isinstance(o, URIRef):
        edge_type = str(p).split("/")[-1].split("#")[-1]
        G.add_edge(str(s), str(o), relation=edge_type)

print(f"NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
```

---

## Building a Complete KG: Example

Let us build a small knowledge graph about programming languages and query it:

```python
from rdflib import Graph, Namespace, Literal
from rdflib.namespace import RDF, RDFS, XSD, FOAF

g = Graph()

# Namespaces
EX = Namespace("http://example.org/")
LANG = Namespace("http://example.org/lang/")
PERSON = Namespace("http://example.org/person/")
COMPANY = Namespace("http://example.org/company/")

g.bind("ex", EX)
g.bind("lang", LANG)
g.bind("person", PERSON)
g.bind("company", COMPANY)

# Schema (RDFS)
g.add((EX.ProgrammingLanguage, RDF.type, RDFS.Class))
g.add((EX.Person, RDF.type, RDFS.Class))
g.add((EX.Company, RDF.type, RDFS.Class))
g.add((EX.createdBy, RDF.type, RDF.Property))
g.add((EX.createdBy, RDFS.domain, EX.ProgrammingLanguage))
g.add((EX.createdBy, RDFS.range, EX.Person))

# Languages
languages = [
    ("python", "Python", 1991, "guido"),
    ("java", "Java", 1995, "james_gosling"),
    ("javascript", "JavaScript", 1995, "brendan_eich"),
    ("rust", "Rust", 2010, "graydon_hoare"),
    ("go", "Go", 2009, "rob_pike"),
]

for lid, name, year, creator in languages:
    lang_uri = LANG[lid]
    g.add((lang_uri, RDF.type, EX.ProgrammingLanguage))
    g.add((lang_uri, FOAF.name, Literal(name)))
    g.add((lang_uri, EX.year, Literal(year, datatype=XSD.integer)))
    g.add((lang_uri, EX.createdBy, PERSON[creator]))

# People
people = [
    ("guido", "Guido van Rossum", "google"),
    ("james_gosling", "James Gosling", "oracle"),
    ("brendan_eich", "Brendan Eich", "mozilla"),
    ("graydon_hoare", "Graydon Hoare", "mozilla"),
    ("rob_pike", "Rob Pike", "google"),
]

for pid, name, company in people:
    person_uri = PERSON[pid]
    g.add((person_uri, RDF.type, EX.Person))
    g.add((person_uri, FOAF.name, Literal(name)))
    g.add((person_uri, EX.worksAt, COMPANY[company]))

# Companies
for cid, name in [("google", "Google"), ("oracle", "Oracle"), ("mozilla", "Mozilla")]:
    g.add((COMPANY[cid], RDF.type, EX.Company))
    g.add((COMPANY[cid], FOAF.name, Literal(name)))

print(f"Total triples: {len(g)}")
print(g.serialize(format="turtle"))
```

### Querying the KG

```python
# Q1: Languages created after 2000
print("=== Languages after 2000 ===")
for row in g.query("""
    PREFIX ex: <http://example.org/>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    SELECT ?name ?year WHERE {
        ?lang a ex:ProgrammingLanguage ;
              foaf:name ?name ;
              ex:year ?year .
        FILTER (?year > 2000)
    }
    ORDER BY ?year
"""):
    print(f"  {row.name} ({row.year})")

# Q2: Who works at Google?
print("\n=== Google employees ===")
for row in g.query("""
    PREFIX ex: <http://example.org/>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    SELECT ?personName ?langName WHERE {
        ?person ex:worksAt <http://example.org/company/google> ;
                foaf:name ?personName .
        ?lang ex:createdBy ?person ;
              foaf:name ?langName .
    }
"""):
    print(f"  {row.personName} created {row.langName}")

# Q3: Languages per company
print("\n=== Languages by company ===")
for row in g.query("""
    PREFIX ex: <http://example.org/>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    SELECT ?companyName (COUNT(?lang) AS ?count)
           (GROUP_CONCAT(?langName; separator=", ") AS ?languages) WHERE {
        ?lang a ex:ProgrammingLanguage ;
              foaf:name ?langName ;
              ex:createdBy ?person .
        ?person ex:worksAt ?company .
        ?company foaf:name ?companyName .
    }
    GROUP BY ?companyName
    ORDER BY DESC(?count)
"""):
    print(f"  {row.companyName}: {row.languages} ({row.count} languages)")
```

---

## Graph Operations

### Merging Graphs

```python
g1 = Graph()
g1.parse("graph1.ttl")

g2 = Graph()
g2.parse("graph2.ttl")

# Union
merged = g1 + g2

# Intersection
common = g1 * g2

# Difference
diff = g1 - g2
```

### Removing Triples

```python
# Remove specific triple
g.remove((PERSON.einstein, EX.field, EX.Mathematics))

# Remove all triples about a subject
g.remove((PERSON.einstein, None, None))

# Remove by pattern
for s, p, o in g.triples((None, RDF.type, EX.Scientist)):
    g.remove((s, p, o))
```

---

## Tips and Best Practices

1. **Always bind prefixes** with `g.bind()` for readable serialization
2. **Use Namespace objects** instead of raw URIRef strings
3. **Choose Turtle for human reading**, JSON-LD for web APIs, N-Triples for bulk operations
4. **Use SPARQL for complex queries** -- pattern matching with `triples()` is limited
5. **rdflib is single-threaded** -- for large-scale RDF, consider Apache Jena or Oxigraph
6. **Validate your URIs** -- malformed URIs cause subtle bugs
7. **Use `initBindings`** in SPARQL queries to safely inject parameters (avoids injection)

---

## References

- rdflib documentation: https://rdflib.readthedocs.io/
- rdflib examples: https://github.com/RDFLib/rdflib/tree/main/examples
- W3C RDF 1.1 Primer: https://www.w3.org/TR/rdf11-primer/
- JSON-LD Playground: https://json-ld.org/playground/
