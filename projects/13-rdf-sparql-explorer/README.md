# Project 13: RDF & SPARQL Explorer

Work with RDF triple stores and SPARQL -- the other half of the knowledge graph world. Build RDF graphs with rdflib, query them with SPARQL, explore Wikidata and DBpedia live endpoints, and compare RDF with property graph approaches.

## What This Project Does

1. **RDF Basics** -- Create an RDF graph from scratch with rdflib. Define namespaces (custom + FOAF, RDFS, OWL, SKOS), add triples, serialize to Turtle, JSON-LD, and N-Triples.
2. **RDFS Ontology** -- Build an RDFS ontology with classes, properties, domain/range constraints, and subclass hierarchies. Validate instances and demonstrate RDFS reasoning.
3. **SPARQL Local Queries** -- Run SPARQL queries on a local RDF graph: SELECT, FILTER, OPTIONAL, aggregation, CONSTRUCT, and property paths.
4. **Query Wikidata** -- Query Wikidata's live SPARQL endpoint: Nobel Prize winners, company founders, topic knowledge graphs. Handle pagination and save results.
5. **Query DBpedia** -- Query DBpedia's SPARQL endpoint: entity lookup, category browsing, and cross-referencing with Wikidata via linked data.
6. **Property Graph vs RDF** -- Convert between NetworkX property graphs and RDF. Compare query expressiveness (Cypher vs SPARQL) and discuss trade-offs.
7. **KG Enrichment from Wikidata** -- Enrich a local knowledge graph by linking entities to Wikidata, pulling additional properties, and merging the enriched data back.

## Key Concepts

### RDF (Resource Description Framework)
- Data model based on **triples**: (subject, predicate, object)
- Everything is identified by **URIs** (Uniform Resource Identifiers)
- Supports **linked data** -- graphs can reference entities across the web
- Standard serializations: Turtle, JSON-LD, N-Triples, RDF/XML

### SPARQL
- The query language for RDF, analogous to SQL for relational databases
- Pattern matching on graph triples with powerful features: OPTIONAL, FILTER, property paths, federated queries
- Used by Wikidata, DBpedia, and many public knowledge graphs

### RDF vs Property Graphs
- **RDF**: Web-standard, linked data, SPARQL, schema via RDFS/OWL
- **Property Graphs** (Neo4j, NetworkX): Richer edge attributes, Cypher queries, more intuitive for many use cases
- Both have strengths -- this project explores when to use each

## Prerequisites

- Python 3.11+
- Dependencies: `rdflib`, `SPARQLWrapper`, `networkx`, `requests`
- Shared LLM layer: `projects/shared/llm_clients.py` (for entity disambiguation)
- Internet connection required for Wikidata/DBpedia queries

## Quick Start

```bash
# Create and serialize an RDF graph
python src/01_rdf_basics.py

# Build an RDFS ontology
python src/02_rdfs_ontology.py

# Run SPARQL queries locally
python src/03_sparql_local.py

# Query Wikidata live endpoint
python src/04_query_wikidata.py

# Query DBpedia live endpoint
python src/05_query_dbpedia.py

# Compare property graphs and RDF
python src/06_property_graph_vs_rdf.py

# Enrich a local KG from Wikidata
python src/07_kg_enrichment_from_wikidata.py
```

## RDF Triple Pattern

```
@prefix foaf: <http://xmlns.com/foaf/0.1/> .
@prefix ex:   <http://example.org/> .

ex:Alice  a             foaf:Person ;
          foaf:name     "Alice" ;
          ex:worksAt    ex:MIT .
```
