# RDF & Semantic Web Fundamentals

## Overview

The Resource Description Framework (RDF) is the foundational data model of the Semantic Web. It represents information as a set of statements called **triples**, each consisting of a subject, predicate, and object. RDF provides a standardized way to describe resources on the web and enables machines to understand and reason about data.

---

## What is RDF?

### The Triple Model

Every piece of information in RDF is expressed as a triple:

```
(Subject, Predicate, Object)
```

Examples:
```
(dbr:Albert_Einstein, dbo:birthPlace, dbr:Ulm)
(dbr:Albert_Einstein, rdf:type, dbo:Scientist)
(dbr:Albert_Einstein, dbo:birthDate, "1879-03-14"^^xsd:date)
```

### Components of a Triple

1. **Subject**: The resource being described (always a URI or blank node)
2. **Predicate**: The property or relationship (always a URI)
3. **Object**: The value -- either another resource (URI/blank node) or a literal (string, number, date)

### URIs as Identifiers

RDF uses URIs (Uniform Resource Identifiers) to uniquely identify resources:

```
http://dbpedia.org/resource/Albert_Einstein    (an entity)
http://dbpedia.org/ontology/birthPlace         (a property)
http://www.w3.org/1999/02/22-rdf-syntax-ns#type  (rdf:type)
```

Prefixed notation for readability:
```
@prefix dbr: <http://dbpedia.org/resource/> .
@prefix dbo: <http://dbpedia.org/ontology/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

dbr:Albert_Einstein dbo:birthPlace dbr:Ulm .
```

### Literals

Objects can be typed literals:
```
"Albert Einstein"                           (plain string)
"1879-03-14"^^xsd:date                      (typed literal)
"76"^^xsd:integer                           (typed literal)
"Physicist and Nobel laureate"@en           (language-tagged string)
```

---

## RDF vs Property Graphs

| Feature | RDF | Property Graphs (Neo4j) |
|---------|-----|------------------------|
| **Data model** | Triples (S, P, O) | Nodes + Relationships + Properties |
| **Edge properties** | Reification (verbose) | Native (simple) |
| **Node properties** | Separate triples | Key-value pairs on nodes |
| **Schema** | RDFS / OWL (formal) | Optional, flexible |
| **Identifiers** | URIs (globally unique) | Internal IDs (local) |
| **Query language** | SPARQL | Cypher / Gremlin |
| **Standards body** | W3C | No single standard |
| **Reasoning** | Built-in (RDFS/OWL) | Requires external tools |
| **Interoperability** | Designed for web-scale linking | Application-specific |
| **Edge labels** | URIs (one per edge) | Strings (one per edge) |
| **Multi-valued properties** | Multiple triples | Arrays or multiple relationships |
| **Typical use** | Linked Data, ontologies, data integration | Application databases, analytics |

### When to Choose RDF

- You need to integrate data from multiple sources with different schemas
- You need formal reasoning (class hierarchies, constraints, inference)
- You are publishing Linked Data on the web
- Interoperability with existing Semantic Web datasets (Wikidata, DBpedia)

### When to Choose Property Graphs

- You need edge properties (weights, timestamps) without reification overhead
- Your application is self-contained (no cross-organization data sharing)
- You prioritize query performance for traversal-heavy workloads
- Your team is more comfortable with Cypher than SPARQL

---

## RDF Serialization Formats

### Turtle (Terse RDF Triple Language)

The most human-readable format. Recommended for authoring and documentation.

```turtle
@prefix dbr: <http://dbpedia.org/resource/> .
@prefix dbo: <http://dbpedia.org/ontology/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix foaf: <http://xmlns.com/foaf/0.1/> .

dbr:Albert_Einstein
    a dbo:Scientist ;
    foaf:name "Albert Einstein" ;
    dbo:birthPlace dbr:Ulm ;
    dbo:birthDate "1879-03-14"^^xsd:date ;
    dbo:field dbr:Physics, dbr:Mathematics .
```

Note: `a` is shorthand for `rdf:type`, `;` continues with the same subject, `,` continues with the same subject and predicate.

### N-Triples

One triple per line, no abbreviations. Best for streaming and large-scale processing.

```ntriples
<http://dbpedia.org/resource/Albert_Einstein> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://dbpedia.org/ontology/Scientist> .
<http://dbpedia.org/resource/Albert_Einstein> <http://xmlns.com/foaf/0.1/name> "Albert Einstein" .
<http://dbpedia.org/resource/Albert_Einstein> <http://dbpedia.org/ontology/birthPlace> <http://dbpedia.org/resource/Ulm> .
```

### JSON-LD (JSON for Linked Data)

RDF in JSON format. Ideal for web APIs and JavaScript applications.

```json
{
    "@context": {
        "name": "http://xmlns.com/foaf/0.1/name",
        "birthPlace": {
            "@id": "http://dbpedia.org/ontology/birthPlace",
            "@type": "@id"
        },
        "birthDate": {
            "@id": "http://dbpedia.org/ontology/birthDate",
            "@type": "http://www.w3.org/2001/XMLSchema#date"
        }
    },
    "@id": "http://dbpedia.org/resource/Albert_Einstein",
    "@type": "http://dbpedia.org/ontology/Scientist",
    "name": "Albert Einstein",
    "birthPlace": "http://dbpedia.org/resource/Ulm",
    "birthDate": "1879-03-14"
}
```

### RDF/XML

The original RDF serialization. Verbose and hard to read, but still widely used.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:dbo="http://dbpedia.org/ontology/"
         xmlns:foaf="http://xmlns.com/foaf/0.1/">
    <dbo:Scientist rdf:about="http://dbpedia.org/resource/Albert_Einstein">
        <foaf:name>Albert Einstein</foaf:name>
        <dbo:birthPlace rdf:resource="http://dbpedia.org/resource/Ulm"/>
    </dbo:Scientist>
</rdf:RDF>
```

### Format Comparison

| Format | Human Readable | File Size | Streaming | Web APIs |
|--------|---------------|-----------|-----------|----------|
| Turtle | Best | Small | No | No |
| N-Triples | Poor | Large | Yes | No |
| JSON-LD | Good | Medium | No | Best |
| RDF/XML | Poor | Large | Partial | No |

---

## RDFS: RDF Schema

RDFS extends RDF with vocabulary for defining classes and properties.

### Core Concepts

```turtle
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex: <http://example.org/> .

# Define classes
ex:Person a rdfs:Class .
ex:Scientist a rdfs:Class ;
    rdfs:subClassOf ex:Person .   # Scientist IS-A Person

# Define properties
ex:birthPlace a rdf:Property ;
    rdfs:domain ex:Person ;       # Subject must be a Person
    rdfs:range ex:Place .         # Object must be a Place

ex:worksAt a rdf:Property ;
    rdfs:domain ex:Person ;
    rdfs:range ex:Organization ;
    rdfs:label "works at"@en ;
    rdfs:comment "The organization where a person is employed"@en .
```

### RDFS Inference

RDFS enables automatic inference:

```
Given:
  ex:Einstein a ex:Scientist .
  ex:Scientist rdfs:subClassOf ex:Person .

Inferred:
  ex:Einstein a ex:Person .          (subclass inference)

Given:
  ex:birthPlace rdfs:range ex:Place .
  ex:Einstein ex:birthPlace ex:Ulm .

Inferred:
  ex:Ulm a ex:Place .               (range inference)
```

---

## OWL: Web Ontology Language

OWL extends RDFS with more powerful reasoning constructs.

### Key OWL Features

```turtle
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix ex: <http://example.org/> .

# Disjoint classes (a Person cannot be a Place)
ex:Person owl:disjointWith ex:Place .

# Equivalent classes
ex:Human owl:equivalentClass ex:Person .

# Inverse properties
ex:hasChild owl:inverseOf ex:hasParent .

# Transitive properties
ex:ancestorOf a owl:TransitiveProperty .
# If A ancestorOf B, and B ancestorOf C, then A ancestorOf C

# Symmetric properties
ex:marriedTo a owl:SymmetricProperty .
# If A marriedTo B, then B marriedTo A

# Functional properties (at most one value)
ex:birthDate a owl:FunctionalProperty .

# Cardinality restrictions
ex:Team rdfs:subClassOf [
    a owl:Restriction ;
    owl:onProperty ex:hasPlayer ;
    owl:minCardinality "11"^^xsd:nonNegativeInteger
] .
```

### OWL Profiles

| Profile | Reasoning | Use Case |
|---------|-----------|----------|
| OWL 2 Full | Undecidable | Theoretical |
| OWL 2 DL | Decidable, complex | Biomedical ontologies |
| OWL 2 EL | Polynomial | Large ontologies (SNOMED CT) |
| OWL 2 QL | SQL-reducible | Database-backed ontologies |
| OWL 2 RL | Rule-based | Enterprise rules |

---

## Linked Data Principles

Tim Berners-Lee's four principles for publishing Linked Data:

1. **Use URIs to name things** -- every entity gets a globally unique identifier
2. **Use HTTP URIs** -- so people can look up those identifiers
3. **Provide useful information** -- when someone looks up a URI, return RDF data
4. **Include links** -- link to other URIs so people can discover more data

### The Linked Data Cloud

Major Linked Data sources:
- **Wikidata** (wikidata.org): Structured data from Wikipedia
- **DBpedia** (dbpedia.org): Extracted from Wikipedia infoboxes
- **GeoNames** (geonames.org): Geographical data
- **FOAF** (Friend of a Friend): Social network vocabulary
- **Schema.org**: Vocabulary for structured web data (used by Google, Bing)

---

## Code Examples with rdflib

### Quick Start

```python
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS, XSD, FOAF

# Create a graph
g = Graph()

# Define namespaces
EX = Namespace("http://example.org/")
g.bind("ex", EX)
g.bind("foaf", FOAF)

# Add triples
g.add((EX.Einstein, RDF.type, EX.Scientist))
g.add((EX.Einstein, FOAF.name, Literal("Albert Einstein")))
g.add((EX.Einstein, EX.birthPlace, EX.Ulm))
g.add((EX.Einstein, EX.birthDate, Literal("1879-03-14", datatype=XSD.date)))

# Serialize
print(g.serialize(format="turtle"))

# Query
for s, p, o in g.triples((EX.Einstein, None, None)):
    print(f"  {p.split('/')[-1]}: {o}")
```

### SPARQL Query on Local Graph

```python
results = g.query("""
    PREFIX ex: <http://example.org/>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>

    SELECT ?name ?birthPlace WHERE {
        ?person a ex:Scientist ;
                foaf:name ?name ;
                ex:birthPlace ?place .
        BIND(STRAFTER(STR(?place), "http://example.org/") AS ?birthPlace)
    }
""")

for row in results:
    print(f"{row.name} born in {row.birthPlace}")
```

---

## Key Takeaways

1. **RDF is the standard for Linked Data** -- use it when interoperability and formal semantics matter
2. **Use Turtle for authoring**, JSON-LD for APIs, N-Triples for bulk processing
3. **RDFS gives you class hierarchies and property constraints** with automatic inference
4. **OWL adds powerful reasoning** but at the cost of complexity
5. **Property graphs are simpler** for application-specific use cases; RDF is better for data integration
6. **rdflib is the Python library** for working with RDF (see the dedicated tutorial)

---

## References

- W3C RDF Primer: https://www.w3.org/TR/rdf11-primer/
- W3C SPARQL 1.1: https://www.w3.org/TR/sparql11-query/
- W3C OWL 2 Overview: https://www.w3.org/TR/owl2-overview/
- rdflib documentation: https://rdflib.readthedocs.io/
- JSON-LD specification: https://www.w3.org/TR/json-ld11/
