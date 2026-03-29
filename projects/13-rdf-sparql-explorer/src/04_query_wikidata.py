"""Query Wikidata's live SPARQL endpoint using SPARQLWrapper.

Demonstrates:
- Nobel Prize winners in Physics
- Companies founded by a specific person
- Knowledge graph of a topic
- Federated query example
- Pagination for large result sets
- Saving results as local JSON
"""

import json
import sys
import time
from pathlib import Path

from SPARQLWrapper import SPARQLWrapper, JSON, XML

# ── Path setup ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "KnowledgeGraphsProject/1.0 (Educational; Python SPARQLWrapper)"


def create_sparql_client() -> SPARQLWrapper:
    """Create a SPARQLWrapper client for Wikidata."""
    sparql = SPARQLWrapper(WIKIDATA_ENDPOINT)
    sparql.setReturnFormat(JSON)
    # Wikidata requires a User-Agent header
    sparql.addCustomHttpHeader("User-Agent", USER_AGENT)
    return sparql


def run_wikidata_query(sparql: SPARQLWrapper, title: str, query: str) -> list[dict]:
    """Execute a SPARQL query against Wikidata and print results.

    Returns:
        List of result dicts.
    """
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

    # Get variable names
    vars_ = results["head"]["vars"]

    # Print header
    header = "  " + " | ".join(f"{v:30s}" for v in vars_)
    print(header)
    print("  " + "-" * min(len(header), 120))

    # Print rows
    rows = []
    for binding in bindings:
        row = {}
        values = []
        for var in vars_:
            if var in binding:
                val = binding[var]["value"]
                # Shorten URIs for display
                if val.startswith("http://www.wikidata.org/entity/"):
                    display = val.split("/")[-1]
                elif len(val) > 30:
                    display = val[:27] + "..."
                else:
                    display = val
                row[var] = val
                values.append(display)
            else:
                row[var] = None
                values.append("(none)")
        rows.append(row)
        print("  " + " | ".join(f"{v:30s}" for v in values))

    print(f"\n  ({len(rows)} results)")
    return rows


def query_nobel_physics(sparql: SPARQLWrapper) -> list[dict]:
    """Find Nobel Prize winners in Physics (recent ones)."""
    query = """
    SELECT ?winner ?winnerLabel ?year ?countryLabel
    WHERE {
        ?winner wdt:P166 wd:Q38104 .      # awarded Nobel Prize in Physics
        ?winner wdt:P166 ?statement .
        OPTIONAL { ?winner wdt:P27 ?country . }
        OPTIONAL {
            ?winner p:P166 ?stmt .
            ?stmt ps:P166 wd:Q38104 ;
                  pq:P585 ?date .
            BIND(YEAR(?date) AS ?year)
        }
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    ORDER BY DESC(?year)
    LIMIT 20
    """
    return run_wikidata_query(sparql, "Nobel Prize Winners in Physics (recent 20)", query)


def query_companies_by_founder(sparql: SPARQLWrapper) -> list[dict]:
    """Find companies founded by Elon Musk."""
    query = """
    SELECT ?company ?companyLabel ?foundedYear ?industryLabel
    WHERE {
        ?company wdt:P112 wd:Q317521 .    # founded by Elon Musk (Q317521)
        OPTIONAL {
            ?company wdt:P571 ?founded .
            BIND(YEAR(?founded) AS ?foundedYear)
        }
        OPTIONAL { ?company wdt:P452 ?industry . }
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    ORDER BY ?foundedYear
    """
    return run_wikidata_query(sparql, "Companies Founded by Elon Musk", query)


def query_topic_knowledge_graph(sparql: SPARQLWrapper) -> list[dict]:
    """Get the knowledge graph around 'Knowledge Graph' on Wikidata."""
    query = """
    SELECT ?property ?propertyLabel ?value ?valueLabel
    WHERE {
        wd:Q33002955 ?prop ?value .       # Q33002955 = Knowledge graph
        ?property wikibase:directClaim ?prop .
        FILTER(isIRI(?value))
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    LIMIT 30
    """
    return run_wikidata_query(
        sparql,
        "Knowledge Graph of 'Knowledge Graph' (Wikidata Q33002955)",
        query,
    )


def query_ai_researchers(sparql: SPARQLWrapper) -> list[dict]:
    """Find notable AI researchers and their affiliations."""
    query = """
    SELECT ?person ?personLabel ?employerLabel ?awardLabel
    WHERE {
        ?person wdt:P106 wd:Q901 .        # occupation: scientist
        ?person wdt:P101 wd:Q11660 .       # field of work: artificial intelligence
        OPTIONAL { ?person wdt:P108 ?employer . }
        OPTIONAL { ?person wdt:P166 ?award . }
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    LIMIT 20
    """
    return run_wikidata_query(sparql, "AI Researchers on Wikidata", query)


def query_with_pagination(sparql: SPARQLWrapper, page_size: int = 10, max_pages: int = 3) -> list[dict]:
    """Demonstrate pagination for large result sets."""
    print(f"\n{'=' * 60}")
    print(f"  Paginated Query: Programming Languages")
    print(f"{'=' * 60}")

    all_results = []

    for page in range(max_pages):
        offset = page * page_size
        query = f"""
        SELECT ?lang ?langLabel ?designerLabel ?year
        WHERE {{
            ?lang wdt:P31 wd:Q9143 .          # instance of: programming language
            OPTIONAL {{
                ?lang wdt:P287 ?designer .
            }}
            OPTIONAL {{
                ?lang wdt:P571 ?inception .
                BIND(YEAR(?inception) AS ?year)
            }}
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        ORDER BY ?langLabel
        LIMIT {page_size}
        OFFSET {offset}
        """

        sparql.setQuery(query)
        try:
            results = sparql.query().convert()
            bindings = results["results"]["bindings"]
        except Exception as e:
            print(f"  Page {page + 1} error: {e}")
            break

        if not bindings:
            print(f"  Page {page + 1}: no more results")
            break

        print(f"\n  Page {page + 1} ({len(bindings)} results):")
        for binding in bindings:
            name = binding.get("langLabel", {}).get("value", "?")
            designer = binding.get("designerLabel", {}).get("value", "?")
            year = binding.get("year", {}).get("value", "?")
            print(f"    {name:30s}  designer: {designer:25s}  year: {year}")
            all_results.append({
                "name": name,
                "designer": designer,
                "year": year,
            })

        # Be polite to the endpoint
        time.sleep(1)

    print(f"\n  Total results across {min(max_pages, page + 1)} pages: {len(all_results)}")
    return all_results


def save_results(results: dict, filename: str) -> None:
    """Save query results as JSON."""
    output_path = OUTPUT_DIR / filename
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")


def main():
    print("Querying Wikidata SPARQL endpoint...")
    print(f"Endpoint: {WIKIDATA_ENDPOINT}\n")

    sparql = create_sparql_client()
    all_results = {}

    # Query 1: Nobel Prize winners
    results = query_nobel_physics(sparql)
    all_results["nobel_physics"] = results
    time.sleep(1)  # rate limiting

    # Query 2: Companies by founder
    results = query_companies_by_founder(sparql)
    all_results["musk_companies"] = results
    time.sleep(1)

    # Query 3: Knowledge graph of a topic
    results = query_topic_knowledge_graph(sparql)
    all_results["kg_topic"] = results
    time.sleep(1)

    # Query 4: AI researchers
    results = query_ai_researchers(sparql)
    all_results["ai_researchers"] = results
    time.sleep(1)

    # Query 5: Paginated results
    paginated = query_with_pagination(sparql, page_size=10, max_pages=3)
    all_results["programming_languages"] = paginated

    # Save all results
    save_results(all_results, "wikidata_results.json")

    print("\nDone. All Wikidata queries completed.")


if __name__ == "__main__":
    main()
