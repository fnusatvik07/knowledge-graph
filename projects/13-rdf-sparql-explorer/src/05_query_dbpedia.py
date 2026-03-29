"""Query DBpedia's SPARQL endpoint.

Demonstrates:
- Getting information about a specific entity
- Finding related entities by category
- Comparing DBpedia results with Wikidata
- How linked data connects different knowledge graphs
"""

import json
import sys
import time
from pathlib import Path

from SPARQLWrapper import SPARQLWrapper, JSON

# ── Path setup ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

DBPEDIA_ENDPOINT = "https://dbpedia.org/sparql"
USER_AGENT = "KnowledgeGraphsProject/1.0 (Educational; Python SPARQLWrapper)"


def create_dbpedia_client() -> SPARQLWrapper:
    """Create a SPARQLWrapper client for DBpedia."""
    sparql = SPARQLWrapper(DBPEDIA_ENDPOINT)
    sparql.setReturnFormat(JSON)
    sparql.addCustomHttpHeader("User-Agent", USER_AGENT)
    return sparql


def run_dbpedia_query(sparql: SPARQLWrapper, title: str, query: str) -> list[dict]:
    """Execute a SPARQL query against DBpedia and print results."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
    except Exception as e:
        print(f"  Error: {e}")
        return []

    bindings = results["results"]["bindings"]
    if not bindings:
        print("  (no results)")
        return []

    vars_ = results["head"]["vars"]

    # Print header
    header = "  " + " | ".join(f"{v:35s}" for v in vars_)
    print(header)
    print("  " + "-" * min(len(header), 120))

    rows = []
    for binding in bindings:
        row = {}
        values = []
        for var in vars_:
            if var in binding:
                val = binding[var]["value"]
                if val.startswith("http://dbpedia.org/resource/"):
                    display = val.replace("http://dbpedia.org/resource/", "dbr:")
                elif val.startswith("http://dbpedia.org/ontology/"):
                    display = val.replace("http://dbpedia.org/ontology/", "dbo:")
                elif len(val) > 35:
                    display = val[:32] + "..."
                else:
                    display = val
                row[var] = val
                values.append(display)
            else:
                row[var] = None
                values.append("(none)")
        rows.append(row)
        print("  " + " | ".join(f"{v:35s}" for v in values))

    print(f"\n  ({len(rows)} results)")
    return rows


def query_entity_info(sparql: SPARQLWrapper) -> list[dict]:
    """Get detailed information about a specific entity (Alan Turing)."""
    query = """
    PREFIX dbr: <http://dbpedia.org/resource/>
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>

    SELECT ?property ?value
    WHERE {
        dbr:Alan_Turing ?property ?value .
        FILTER(
            ?property IN (
                rdfs:label,
                dbo:abstract,
                dbo:birthDate,
                dbo:birthPlace,
                dbo:field,
                dbo:knownFor,
                dbo:award,
                dbo:almaMater,
                foaf:name
            )
        )
        FILTER(LANG(?value) = "en" || !isLiteral(?value))
    }
    LIMIT 30
    """
    return run_dbpedia_query(sparql, "Entity Info: Alan Turing", query)


def query_by_category(sparql: SPARQLWrapper) -> list[dict]:
    """Find entities related by category (Machine Learning researchers)."""
    query = """
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX dct: <http://purl.org/dc/terms/>
    PREFIX dbc: <http://dbpedia.org/resource/Category:>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?person ?name ?field
    WHERE {
        ?person dct:subject dbc:Machine_learning_researchers ;
                a dbo:Person .
        OPTIONAL { ?person foaf:name ?name . FILTER(LANG(?name) = "en") }
        OPTIONAL {
            ?person rdfs:label ?name .
            FILTER(LANG(?name) = "en")
        }
        OPTIONAL { ?person dbo:field ?fieldRes .
                   ?fieldRes rdfs:label ?field .
                   FILTER(LANG(?field) = "en") }
    }
    ORDER BY ?name
    LIMIT 20
    """
    return run_dbpedia_query(sparql, "Machine Learning Researchers (by Category)", query)


def query_programming_languages(sparql: SPARQLWrapper) -> list[dict]:
    """Find programming languages and their designers on DBpedia."""
    query = """
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?lang ?name ?designerName ?year
    WHERE {
        ?lang a dbo:ProgrammingLanguage ;
              rdfs:label ?name .
        FILTER(LANG(?name) = "en")
        OPTIONAL {
            ?lang dbo:designer ?designer .
            ?designer rdfs:label ?designerName .
            FILTER(LANG(?designerName) = "en")
        }
        OPTIONAL {
            ?lang dbo:releaseDate ?date .
            BIND(YEAR(?date) AS ?year)
        }
    }
    ORDER BY ?name
    LIMIT 20
    """
    return run_dbpedia_query(sparql, "Programming Languages on DBpedia", query)


def compare_dbpedia_wikidata(sparql_dbpedia: SPARQLWrapper) -> None:
    """Compare DBpedia and Wikidata for the same entity.

    Shows how linked data connects different knowledge graphs via
    owl:sameAs and other linking predicates.
    """
    print(f"\n{'=' * 60}")
    print(f"  COMPARING DBpedia and Wikidata")
    print(f"{'=' * 60}")

    # Find Wikidata links for DBpedia entities
    query = """
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX dbr: <http://dbpedia.org/resource/>

    SELECT ?entity ?label ?wikidataID
    WHERE {
        VALUES ?entity {
            dbr:Geoffrey_Hinton
            dbr:Yann_LeCun
            dbr:Yoshua_Bengio
            dbr:Deep_learning
            dbr:Knowledge_graph
        }
        ?entity rdfs:label ?label .
        FILTER(LANG(?label) = "en")
        OPTIONAL {
            ?entity owl:sameAs ?wikidataID .
            FILTER(STRSTARTS(STR(?wikidataID), "http://www.wikidata.org/"))
        }
    }
    """
    results = run_dbpedia_query(sparql_dbpedia, "DBpedia <-> Wikidata Links (owl:sameAs)", query)

    print("\n  Key insight: DBpedia and Wikidata describe the same entities.")
    print("  The owl:sameAs predicate links equivalent entities across KGs.")
    print("  This is the foundation of Linked Data and the Semantic Web.")

    if results:
        print("\n  Cross-KG entity mapping:")
        for r in results:
            entity = r.get("entity", "").split("/")[-1]
            wikidata = r.get("wikidataID", "")
            if wikidata:
                wikidata_id = wikidata.split("/")[-1]
                print(f"    dbr:{entity:30s} <-> wd:{wikidata_id}")


def show_linked_data_principles() -> None:
    """Explain how linked data connects different knowledge graphs."""
    print(f"\n{'=' * 60}")
    print(f"  LINKED DATA PRINCIPLES")
    print(f"{'=' * 60}")

    print("""
  The Linked Data principles (Tim Berners-Lee, 2006):

  1. Use URIs to name things
     - DBpedia:  http://dbpedia.org/resource/Alan_Turing
     - Wikidata: http://www.wikidata.org/entity/Q7251

  2. Use HTTP URIs so people can look up those names
     - Dereference the URI to get structured data about the entity

  3. When someone looks up a URI, provide useful RDF information
     - Properties, types, labels, descriptions in multiple languages

  4. Include links to other URIs so people can discover more things
     - owl:sameAs links connect equivalent entities across KGs
     - rdfs:seeAlso points to related resources

  Key linking predicates:
    owl:sameAs       - Entities are the same thing
    skos:exactMatch  - Concepts are equivalent
    skos:closeMatch  - Concepts are similar
    rdfs:seeAlso     - Related resources
    schema:sameAs    - Schema.org equivalent
    """)


def main():
    print("Querying DBpedia SPARQL endpoint...")
    print(f"Endpoint: {DBPEDIA_ENDPOINT}\n")

    sparql = create_dbpedia_client()
    all_results = {}

    # Query 1: Entity information
    results = query_entity_info(sparql)
    all_results["alan_turing"] = results
    time.sleep(1)

    # Query 2: Category-based discovery
    results = query_by_category(sparql)
    all_results["ml_researchers"] = results
    time.sleep(1)

    # Query 3: Programming languages
    results = query_programming_languages(sparql)
    all_results["programming_languages"] = results
    time.sleep(1)

    # Query 4: Compare with Wikidata
    compare_dbpedia_wikidata(sparql)
    time.sleep(1)

    # Linked Data principles
    show_linked_data_principles()

    # Save results
    output_path = OUTPUT_DIR / "dbpedia_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    print("\nDone. All DBpedia queries completed.")


if __name__ == "__main__":
    main()
