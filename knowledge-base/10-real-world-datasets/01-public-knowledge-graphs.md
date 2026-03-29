# Public Knowledge Graphs

Large-scale, publicly available knowledge graphs are invaluable for learning, benchmarking, and bootstrapping domain-specific applications. This section covers the major public KGs, how to access them, and practical Python examples for working with each.

## General-Purpose Knowledge Graphs

### Wikidata

**URL**: https://www.wikidata.org/
**Size**: 100M+ items, 1.5B+ statements
**Query Endpoint**: https://query.wikidata.org/
**Format**: RDF / JSON
**License**: CC0 (public domain)

Wikidata is the largest open, community-maintained knowledge graph. It powers Wikipedia infoboxes and is used by Google, Apple, and others.

#### Structure

- **Items**: Identified by Q-IDs (e.g., Q937 = Albert Einstein)
- **Properties**: Identified by P-IDs (e.g., P106 = occupation)
- **Statements**: Item-Property-Value triples with qualifiers and references

#### SPARQL Query Example

```sparql
# Find all Nobel Prize winners in Physics
SELECT ?person ?personLabel ?year WHERE {
  ?person wdt:P166 wd:Q38104 .  # P166 = award received, Q38104 = Nobel Prize in Physics
  ?person wdt:P166 ?award .
  OPTIONAL {
    ?person p:P166 ?statement .
    ?statement ps:P166 wd:Q38104 .
    ?statement pq:P585 ?date .
    BIND(YEAR(?date) AS ?year)
  }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
}
ORDER BY ?year
LIMIT 50
```

#### Python Access

```python
# Using the Wikidata SPARQL endpoint
import requests

def query_wikidata(sparql_query: str) -> list[dict]:
    """Execute a SPARQL query against Wikidata."""
    url = "https://query.wikidata.org/sparql"
    headers = {"Accept": "application/json", "User-Agent": "KGResearch/1.0"}
    response = requests.get(url, params={"query": sparql_query}, headers=headers)
    response.raise_for_status()
    data = response.json()
    return data["results"]["bindings"]

results = query_wikidata("""
    SELECT ?person ?personLabel WHERE {
        ?person wdt:P106 wd:Q170790 .  # occupation = mathematician
        ?person wdt:P27 wd:Q30 .       # country of citizenship = USA
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
    }
    LIMIT 20
""")
for r in results:
    print(r["personLabel"]["value"])
```

```python
# Using the Wikidata API (for individual items)
def get_wikidata_entity(qid: str) -> dict:
    """Get a Wikidata entity by its Q-ID."""
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    response = requests.get(url)
    return response.json()["entities"][qid]

einstein = get_wikidata_entity("Q937")
print(einstein["labels"]["en"]["value"])  # "Albert Einstein"
```

### DBpedia

**URL**: https://www.dbpedia.org/
**Size**: 6M+ entities, 1.7B+ triples
**SPARQL Endpoint**: https://dbpedia.org/sparql
**Format**: RDF
**License**: CC BY-SA 3.0

DBpedia extracts structured information from Wikipedia infoboxes. It is more curated than Wikidata but updated less frequently.

#### Python Access

```python
from SPARQLWrapper import SPARQLWrapper, JSON

def query_dbpedia(sparql_query: str) -> list[dict]:
    """Execute a SPARQL query against DBpedia."""
    sparql = SPARQLWrapper("https://dbpedia.org/sparql")
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results["results"]["bindings"]

results = query_dbpedia("""
    SELECT ?person ?abstract WHERE {
        ?person a dbo:Scientist ;
                dbo:birthPlace dbr:Warsaw ;
                dbo:abstract ?abstract .
        FILTER (lang(?abstract) = 'en')
    }
    LIMIT 10
""")
```

### YAGO

**URL**: https://yago-knowledge.org/
**Size**: 50M+ entities, 2B+ facts
**Format**: RDF / TSV
**License**: CC BY 4.0

YAGO (Yet Another Great Ontology) combines Wikidata, Wikipedia, and WordNet into a curated knowledge graph with a clean taxonomy.

#### Download

```bash
# Download YAGO 4.5 (latest)
wget https://yago-knowledge.org/data/yago4.5/yago-facts.tsv.gz
wget https://yago-knowledge.org/data/yago4.5/yago-taxonomy.tsv.gz
```

#### Python Access

```python
import pandas as pd

# Load YAGO facts
facts = pd.read_csv("yago-facts.tsv.gz", sep="\t", compression="gzip",
                     names=["subject", "predicate", "object"])
print(f"Loaded {len(facts)} facts")
print(facts.head())
```

### ConceptNet

**URL**: https://conceptnet.io/
**Size**: 21M+ edges, 8M+ nodes
**API**: https://api.conceptnet.io/
**Format**: JSON API / CSV dump
**License**: CC BY-SA 4.0

ConceptNet is a commonsense knowledge graph. Unlike Wikidata/DBpedia (which focus on encyclopedic facts), ConceptNet captures everyday knowledge like "a dog is a pet" and "rain makes things wet."

#### Python Access

```python
import requests

def conceptnet_lookup(concept: str, limit: int = 10) -> list[dict]:
    """Look up a concept in ConceptNet."""
    url = f"https://api.conceptnet.io/c/en/{concept}"
    response = requests.get(url, params={"limit": limit})
    data = response.json()
    edges = []
    for edge in data.get("edges", []):
        edges.append({
            "start": edge["start"]["label"],
            "relation": edge["rel"]["label"],
            "end": edge["end"]["label"],
            "weight": edge["weight"],
        })
    return edges

results = conceptnet_lookup("knowledge_graph")
for r in results:
    print(f"  {r['start']} --[{r['relation']}]--> {r['end']} (weight: {r['weight']:.1f})")
```

### Freebase (Deprecated but Influential)

**Status**: Deprecated by Google in 2016, data migrated to Wikidata
**Legacy Data**: Available as a dump on Google's developer site
**Format**: RDF / N-Triples
**Impact**: Powered Google Knowledge Panels, basis for many KG benchmarks (FB15k, FB15k-237)

Freebase is no longer active but its data and benchmarks remain important. Many KG embedding benchmarks (see [02 - Benchmark Datasets](./02-benchmark-datasets.md)) use Freebase-derived datasets.

```python
# Freebase data is now in Wikidata -- use Wikidata's P646 property to find Freebase IDs
# Example: Find the Wikidata item for a Freebase ID
results = query_wikidata("""
    SELECT ?item ?itemLabel WHERE {
        ?item wdt:P646 "/m/0jcx" .  # Freebase ID for "Albert Einstein"
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
    }
""")
```

### NELL (Never-Ending Language Learner)

**URL**: http://rtw.ml.cmu.edu/
**Organization**: Carnegie Mellon University
**Size**: 50M+ beliefs extracted from the web
**Format**: CSV / TSV
**Approach**: Continuous automated extraction from the web since 2010

NELL is a system that has been continuously reading the web and extracting structured knowledge since January 2010.

```python
# Download NELL beliefs
# http://rtw.ml.cmu.edu/rtw/resources
import pandas as pd

nell = pd.read_csv("NELL.08m.1115.esv.csv.gz", sep="\t", compression="gzip")
# Columns: Entity, Relation, Value, Iteration, Probability, Source, ...
high_confidence = nell[nell["Probability"] > 0.95]
print(f"High-confidence facts: {len(high_confidence)}")
```

## Domain-Specific Knowledge Graphs

### UMLS (Unified Medical Language System)

**URL**: https://www.nlm.nih.gov/research/umls/
**Domain**: Biomedical
**Size**: 4M+ concepts, 15M+ relationships
**Access**: Free with UMLS license (registration required)

```python
# Access UMLS via the API
import requests

def search_umls(term: str, api_key: str) -> list[dict]:
    """Search UMLS for a medical concept."""
    url = "https://uts-ws.nlm.nih.gov/rest/search/current"
    params = {"string": term, "apiKey": api_key, "returnIdType": "concept"}
    response = requests.get(url, params=params)
    return response.json()["result"]["results"]

# results = search_umls("diabetes", "your-api-key")
```

### Gene Ontology (GO)

**URL**: http://geneontology.org/
**Domain**: Molecular biology
**Size**: 45,000+ terms, 100,000+ relationships
**Format**: OBO / OWL / JSON
**License**: CC BY 4.0

```python
# Load Gene Ontology with pronto
import pronto

go = pronto.Ontology("http://purl.obolibrary.org/obo/go.obo")
term = go["GO:0008150"]  # biological_process
print(f"{term.id}: {term.name}")
for child in term.subclasses(distance=1):
    print(f"  - {child.name}")
```

### ChEBI (Chemical Entities of Biological Interest)

**URL**: https://www.ebi.ac.uk/chebi/
**Domain**: Chemistry / biochemistry
**Size**: 60,000+ entities
**Format**: OBO / OWL / SDF
**License**: CC BY 4.0

```python
# Access ChEBI via the API
import requests

def search_chebi(name: str) -> list[dict]:
    url = "https://www.ebi.ac.uk/webservices/chebi/2.0/test/getLiteEntity"
    params = {"search": name, "searchCategory": "ALL", "maximumResults": 5}
    response = requests.get(url, params=params)
    return response.json()
```

## Loading Public KGs into Your Graph Database

### Wikidata Subset to Neo4j

```python
from langchain_community.graphs import Neo4jGraph

graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="password")

# Load Wikidata scientists into Neo4j
scientists = query_wikidata("""
    SELECT ?person ?personLabel ?birthDate WHERE {
        ?person wdt:P106 wd:Q901 .  # occupation = scientist
        ?person wdt:P569 ?birthDate .
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
    }
    LIMIT 1000
""")

for s in scientists:
    name = s["personLabel"]["value"]
    qid = s["person"]["value"].split("/")[-1]
    graph.query(
        "MERGE (p:Person {wikidata_id: $qid}) SET p.name = $name",
        params={"qid": qid, "name": name}
    )
```

> **LangChain Neo4j integration**: https://python.langchain.com/docs/integrations/graphs/neo4j_cypher/

## Summary Table

| Knowledge Graph | Domain | Size | Access | Format | License |
|----------------|--------|------|--------|--------|---------|
| Wikidata | General | 100M+ items | SPARQL / API | RDF / JSON | CC0 |
| DBpedia | General | 6M+ entities | SPARQL | RDF | CC BY-SA |
| YAGO | General | 50M+ entities | Download | RDF / TSV | CC BY |
| ConceptNet | Commonsense | 21M+ edges | API / Download | JSON / CSV | CC BY-SA |
| Freebase | General | Deprecated | Via Wikidata | N-Triples | CC BY |
| NELL | General | 50M+ beliefs | Download | CSV | Research |
| UMLS | Biomedical | 4M+ concepts | API (licensed) | Various | Licensed |
| Gene Ontology | Biology | 45K+ terms | Download | OBO / OWL | CC BY |
| ChEBI | Chemistry | 60K+ entities | API / Download | OBO / SDF | CC BY |

## Next Steps

- [02 - Benchmark Datasets](./02-benchmark-datasets.md) -- standard datasets for evaluating KG systems
- [Entity Extraction](../02-kg-construction/01-entity-extraction.md) -- build your own KG from text
