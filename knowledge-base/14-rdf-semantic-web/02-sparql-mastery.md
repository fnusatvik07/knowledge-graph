# SPARQL Mastery: Querying RDF Knowledge Graphs

## Overview

SPARQL (SPARQL Protocol and RDF Query Language) is the standard query language for RDF data, analogous to SQL for relational databases. It can query any RDF-based knowledge graph, from local rdflib graphs to public endpoints like Wikidata and DBpedia.

---

## SPARQL Query Forms

SPARQL has four query forms:

| Form | Purpose | Returns |
|------|---------|---------|
| **SELECT** | Retrieve variable bindings | Table of results |
| **CONSTRUCT** | Build new RDF graph | RDF triples |
| **ASK** | Boolean existence check | true / false |
| **DESCRIBE** | Get information about a resource | RDF triples |

---

## SELECT Queries

### Basic Triple Pattern

```sparql
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbr: <http://dbpedia.org/resource/>

SELECT ?birthPlace WHERE {
    dbr:Albert_Einstein dbo:birthPlace ?birthPlace .
}
```

### Multiple Triple Patterns

```sparql
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

SELECT ?name ?birthPlace ?birthDate WHERE {
    ?person a dbo:Scientist .
    ?person foaf:name ?name .
    ?person dbo:birthPlace ?birthPlace .
    ?person dbo:birthDate ?birthDate .
}
LIMIT 10
```

### SELECT DISTINCT

```sparql
SELECT DISTINCT ?field WHERE {
    ?scientist a dbo:Scientist ;
               dbo:field ?field .
}
ORDER BY ?field
```

---

## FILTER

Filter results based on conditions:

```sparql
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT ?scientist ?name ?birthDate WHERE {
    ?scientist a dbo:Scientist ;
               foaf:name ?name ;
               dbo:birthDate ?birthDate .
    FILTER (?birthDate > "1900-01-01"^^xsd:date)
    FILTER (LANG(?name) = "en")
}
```

### Common Filter Functions

```sparql
# String matching
FILTER (CONTAINS(?name, "Einstein"))
FILTER (STRSTARTS(?name, "Albert"))
FILTER (REGEX(?name, "^Albert", "i"))   # case-insensitive regex

# Numeric comparison
FILTER (?age > 30 && ?age < 80)

# Type checking
FILTER (isURI(?x))
FILTER (isLiteral(?x))
FILTER (DATATYPE(?x) = xsd:integer)

# Existence (using BOUND)
FILTER (BOUND(?optionalVar))
FILTER (!BOUND(?optionalVar))

# String operations
BIND(STRLEN(?name) AS ?nameLength)
BIND(UCASE(?name) AS ?upperName)
BIND(STRAFTER(STR(?uri), "resource/") AS ?localName)
```

---

## OPTIONAL

Include results even when some patterns do not match:

```sparql
SELECT ?scientist ?name ?spouse WHERE {
    ?scientist a dbo:Scientist ;
               foaf:name ?name .
    OPTIONAL { ?scientist dbo:spouse ?spouse }
}
```

Without OPTIONAL, scientists without a spouse would be excluded entirely. With OPTIONAL, they appear with a null value for `?spouse`.

### Multiple OPTIONALs

```sparql
SELECT ?person ?name ?birth ?death WHERE {
    ?person a dbo:Person ;
            foaf:name ?name .
    OPTIONAL { ?person dbo:birthDate ?birth }
    OPTIONAL { ?person dbo:deathDate ?death }
}
```

---

## UNION

Combine results from alternative patterns:

```sparql
SELECT ?person ?name WHERE {
    {
        ?person a dbo:Scientist ;
                foaf:name ?name .
    }
    UNION
    {
        ?person a dbo:Philosopher ;
                foaf:name ?name .
    }
}
```

This returns all scientists and all philosophers.

---

## Aggregations

### COUNT

```sparql
SELECT (COUNT(?scientist) AS ?total) WHERE {
    ?scientist a dbo:Scientist .
}
```

### GROUP BY

```sparql
SELECT ?field (COUNT(?scientist) AS ?count) WHERE {
    ?scientist a dbo:Scientist ;
               dbo:field ?field .
}
GROUP BY ?field
ORDER BY DESC(?count)
LIMIT 10
```

### HAVING

Filter groups after aggregation:

```sparql
SELECT ?university (COUNT(?alumnus) AS ?alumni_count) WHERE {
    ?alumnus dbo:almaMater ?university .
    ?alumnus a dbo:Scientist .
}
GROUP BY ?university
HAVING (COUNT(?alumnus) > 50)
ORDER BY DESC(?alumni_count)
```

### Other Aggregates

```sparql
SELECT
    ?field
    (COUNT(?s) AS ?count)
    (MIN(?birthDate) AS ?earliest)
    (MAX(?birthDate) AS ?latest)
    (AVG(?age) AS ?avgAge)
    (GROUP_CONCAT(?name; separator=", ") AS ?names)
WHERE {
    ?s a dbo:Scientist ;
       dbo:field ?field ;
       foaf:name ?name ;
       dbo:birthDate ?birthDate .
    BIND(YEAR(NOW()) - YEAR(?birthDate) AS ?age)
}
GROUP BY ?field
```

---

## Subqueries

Nest queries for complex logic:

```sparql
# Find scientists who have published more papers than average
SELECT ?scientist ?name ?paperCount WHERE {
    ?scientist a dbo:Scientist ;
               foaf:name ?name .
    {
        SELECT ?scientist (COUNT(?paper) AS ?paperCount) WHERE {
            ?paper dbo:author ?scientist .
        }
        GROUP BY ?scientist
    }
    {
        SELECT (AVG(?cnt) AS ?avgPapers) WHERE {
            SELECT ?s (COUNT(?p) AS ?cnt) WHERE {
                ?p dbo:author ?s .
                ?s a dbo:Scientist .
            }
            GROUP BY ?s
        }
    }
    FILTER (?paperCount > ?avgPapers)
}
ORDER BY DESC(?paperCount)
LIMIT 20
```

---

## Property Paths

Navigate the graph using path expressions:

```sparql
# Direct predicate
?x foaf:knows ?y .

# Sequence (A knows someone who knows B)
?x foaf:knows/foaf:knows ?y .

# Alternative (born in or lives in)
?x (dbo:birthPlace|dbo:residence) ?place .

# Zero or more hops (transitive closure)
?x rdfs:subClassOf* ?superClass .

# One or more hops
?x rdfs:subClassOf+ ?superClass .

# Inverse path
?x ^dbo:birthPlace ?person .    # equivalent to: ?person dbo:birthPlace ?x

# Negated property set
?x !rdf:type ?y .               # any predicate except rdf:type
```

### Example: Class Hierarchy Traversal

```sparql
# Find all superclasses of Scientist (transitive)
SELECT ?superClass WHERE {
    dbo:Scientist rdfs:subClassOf* ?superClass .
}
```

### Example: Multi-Hop Paths

```sparql
# Find people within 3 hops of "knows" from Einstein
SELECT DISTINCT ?person ?name WHERE {
    dbr:Albert_Einstein (foaf:knows){1,3} ?person .
    ?person foaf:name ?name .
}
```

---

## CONSTRUCT Queries

Build new RDF graphs from query results:

```sparql
PREFIX ex: <http://example.org/>

CONSTRUCT {
    ?scientist ex:researchArea ?field .
    ?scientist ex:label ?name .
}
WHERE {
    ?scientist a dbo:Scientist ;
               foaf:name ?name ;
               dbo:field ?field .
    FILTER (LANG(?name) = "en")
}
```

This creates a new, simplified RDF graph from the source data.

---

## ASK Queries

Check if a pattern exists:

```sparql
ASK {
    dbr:Albert_Einstein dbo:birthPlace dbr:Ulm .
}
# Returns: true
```

---

## DESCRIBE Queries

Get all known information about a resource:

```sparql
DESCRIBE dbr:Albert_Einstein
```

Returns all triples where Einstein is subject or object (implementation-dependent).

---

## Federated Queries (SERVICE)

Query remote SPARQL endpoints from within a query:

```sparql
# Query Wikidata from within a DBpedia query
SELECT ?scientist ?name ?wikidataID WHERE {
    ?scientist a dbo:Scientist ;
               foaf:name ?name ;
               owl:sameAs ?wikidataID .
    FILTER (STRSTARTS(STR(?wikidataID), "http://www.wikidata.org/"))

    SERVICE <https://query.wikidata.org/sparql> {
        ?wikidataID wdt:P69 ?university .
    }
}
LIMIT 10
```

---

## Querying Wikidata

Wikidata SPARQL endpoint: https://query.wikidata.org/

Wikidata uses its own property identifiers (P-numbers and Q-numbers):

```sparql
# Wikidata: Find all Nobel Prize winners in Physics
SELECT ?person ?personLabel ?year WHERE {
    ?person wdt:P166 wd:Q38104 .          # P166 = award received, Q38104 = Nobel Prize in Physics
    ?person wdt:P569 ?birthDate .          # P569 = date of birth
    BIND(YEAR(?birthDate) AS ?year)

    SERVICE wikibase:label {
        bd:serviceParam wikibase:language "en" .
    }
}
ORDER BY ?year
LIMIT 20
```

### Wikidata: Countries by Population

```sparql
SELECT ?country ?countryLabel ?population WHERE {
    ?country wdt:P31 wd:Q6256 .           # instance of country
    ?country wdt:P1082 ?population .       # population

    SERVICE wikibase:label {
        bd:serviceParam wikibase:language "en" .
    }
}
ORDER BY DESC(?population)
LIMIT 20
```

### Wikidata: Programming Languages and Creators

```sparql
SELECT ?lang ?langLabel ?creator ?creatorLabel ?year WHERE {
    ?lang wdt:P31 wd:Q9143 .              # instance of programming language
    ?lang wdt:P178 ?creator .              # developer
    OPTIONAL {
        ?lang wdt:P571 ?inception .        # inception date
        BIND(YEAR(?inception) AS ?year)
    }

    SERVICE wikibase:label {
        bd:serviceParam wikibase:language "en" .
    }
}
ORDER BY ?year
LIMIT 30
```

---

## Querying DBpedia

DBpedia SPARQL endpoint: https://dbpedia.org/sparql

```sparql
# DBpedia: Cities with population over 10 million
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbr: <http://dbpedia.org/resource/>

SELECT ?city ?name ?population WHERE {
    ?city a dbo:City ;
          foaf:name ?name ;
          dbo:populationTotal ?population .
    FILTER (?population > 10000000)
    FILTER (LANG(?name) = "en")
}
ORDER BY DESC(?population)
LIMIT 20
```

### DBpedia: Films by Director

```sparql
SELECT ?film ?filmName ?year WHERE {
    ?film a dbo:Film ;
          foaf:name ?filmName ;
          dbo:director dbr:Christopher_Nolan .
    OPTIONAL { ?film dbo:releaseDate ?date . BIND(YEAR(?date) AS ?year) }
    FILTER (LANG(?filmName) = "en")
}
ORDER BY ?year
```

---

## Python Access via SPARQLWrapper

```python
from SPARQLWrapper import SPARQLWrapper, JSON

# Query Wikidata
sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql.setQuery("""
    SELECT ?item ?itemLabel ?population WHERE {
        ?item wdt:P31 wd:Q6256 .
        ?item wdt:P1082 ?population .
        SERVICE wikibase:label {
            bd:serviceParam wikibase:language "en" .
        }
    }
    ORDER BY DESC(?population)
    LIMIT 10
""")
sparql.setReturnFormat(JSON)
results = sparql.query().convert()

for result in results["results"]["bindings"]:
    name = result["itemLabel"]["value"]
    pop = int(result["population"]["value"])
    print(f"{name}: {pop:,}")
```

### Query DBpedia from Python

```python
sparql = SPARQLWrapper("https://dbpedia.org/sparql")
sparql.setQuery("""
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>

    SELECT ?name ?abstract WHERE {
        dbr:Python_(programming_language) foaf:name ?name ;
                                          dbo:abstract ?abstract .
        FILTER (LANG(?abstract) = "en")
    }
""")
sparql.setReturnFormat(JSON)
results = sparql.query().convert()

for r in results["results"]["bindings"]:
    print(f"Name: {r['name']['value']}")
    print(f"Abstract: {r['abstract']['value'][:200]}...")
```

### Converting Results to Pandas

```python
import pandas as pd

def sparql_to_dataframe(endpoint, query):
    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    bindings = results["results"]["bindings"]
    columns = results["head"]["vars"]

    data = []
    for binding in bindings:
        row = {col: binding.get(col, {}).get("value", None) for col in columns}
        data.append(row)

    return pd.DataFrame(data, columns=columns)

df = sparql_to_dataframe(
    "https://query.wikidata.org/sparql",
    """
    SELECT ?lang ?langLabel WHERE {
        ?lang wdt:P31 wd:Q9143 .
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
    }
    LIMIT 50
    """
)
print(df.head(10))
```

---

## SPARQL Tips and Best Practices

1. **Always use LIMIT** when exploring unknown datasets -- unbounded queries can time out
2. **Use FILTER(LANG(?x) = "en")** to avoid duplicate results in different languages
3. **Use OPTIONAL** for properties that may not exist on all resources
4. **Property paths** (`/`, `*`, `+`) are powerful but can be slow on large graphs
5. **Federated queries** (SERVICE) add latency -- minimize data transferred
6. **Wikidata's label service** is essential: always include the `SERVICE wikibase:label` block
7. **Test queries** at the endpoint's web interface before embedding in code
8. **Set timeouts** in SPARQLWrapper for production use: `sparql.setTimeout(30)`

---

## References

- W3C SPARQL 1.1 Specification: https://www.w3.org/TR/sparql11-query/
- Wikidata Query Service: https://query.wikidata.org/
- DBpedia SPARQL Endpoint: https://dbpedia.org/sparql
- SPARQLWrapper: https://sparqlwrapper.readthedocs.io/
- Wikidata SPARQL Tutorial: https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial
