"""Enrich a local knowledge graph with data from Wikidata.

Workflow:
1. Take entities from a local RDF graph
2. Match them to Wikidata entities (label matching + LLM disambiguation)
3. Pull additional properties from Wikidata
4. Add enriched data back to the local graph
5. Show before/after graph statistics
"""

import json
import sys
import time
from pathlib import Path

from rdflib import Graph as RDFGraph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, OWL, FOAF, SKOS, XSD
from SPARQLWrapper import SPARQLWrapper, JSON

# ── Path setup ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared.llm_clients import chat_completion

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

EX = Namespace("http://example.org/kg/")
WD = Namespace("http://www.wikidata.org/entity/")
WDT = Namespace("http://www.wikidata.org/prop/direct/")

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "KnowledgeGraphsProject/1.0 (Educational; Python SPARQLWrapper)"


def build_local_graph() -> RDFGraph:
    """Build a small local RDF graph with entities to enrich."""
    g = RDFGraph()
    g.bind("ex", EX)
    g.bind("foaf", FOAF)
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)

    # People -- we know their names but want more data from Wikidata
    people = [
        ("GeoffreyHinton", "Geoffrey Hinton"),
        ("YannLeCun", "Yann LeCun"),
        ("YoshuaBengio", "Yoshua Bengio"),
        ("AlanTuring", "Alan Turing"),
        ("TimBernersLee", "Tim Berners-Lee"),
    ]
    for entity_id, name in people:
        uri = EX[entity_id]
        g.add((uri, RDF.type, FOAF.Person))
        g.add((uri, FOAF.name, Literal(name)))
        g.add((uri, RDFS.label, Literal(name, lang="en")))

    # Organizations
    orgs = [
        ("Google", "Google"),
        ("MIT", "Massachusetts Institute of Technology"),
        ("Stanford", "Stanford University"),
    ]
    for entity_id, name in orgs:
        uri = EX[entity_id]
        g.add((uri, RDF.type, FOAF.Organization))
        g.add((uri, FOAF.name, Literal(name)))
        g.add((uri, RDFS.label, Literal(name, lang="en")))

    # Technologies
    techs = [
        ("DeepLearning", "Deep learning"),
        ("KnowledgeGraph", "Knowledge graph"),
    ]
    for entity_id, name in techs:
        uri = EX[entity_id]
        g.add((uri, RDF.type, SKOS.Concept))
        g.add((uri, SKOS.prefLabel, Literal(name, lang="en")))
        g.add((uri, RDFS.label, Literal(name, lang="en")))

    return g


def search_wikidata_entity(label: str, entity_type: str | None = None) -> list[dict]:
    """Search Wikidata for entities matching a label.

    Uses the Wikidata SPARQL endpoint to find candidate matches.

    Returns:
        List of candidate entities with QID, label, and description.
    """
    sparql = SPARQLWrapper(WIKIDATA_ENDPOINT)
    sparql.addCustomHttpHeader("User-Agent", USER_AGENT)
    sparql.setReturnFormat(JSON)

    # Search by label with optional type constraint
    type_filter = ""
    if entity_type == "Person":
        type_filter = "?item wdt:P31 wd:Q5 ."  # instance of: human
    elif entity_type == "Organization":
        type_filter = """
        { ?item wdt:P31/wdt:P279* wd:Q43229 . }  # instance of: organization (or subclass)
        """

    query = f"""
    SELECT ?item ?itemLabel ?itemDescription
    WHERE {{
        ?item rdfs:label "{label}"@en .
        {type_filter}
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 5
    """

    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
        candidates = []
        for binding in results["results"]["bindings"]:
            candidates.append({
                "qid": binding["item"]["value"].split("/")[-1],
                "uri": binding["item"]["value"],
                "label": binding.get("itemLabel", {}).get("value", ""),
                "description": binding.get("itemDescription", {}).get("value", ""),
            })
        return candidates
    except Exception as e:
        print(f"    Search error: {e}")
        return []


def disambiguate_entity(
    local_label: str,
    candidates: list[dict],
    provider: str = "openai",
    model: str | None = None,
) -> dict | None:
    """Use LLM to disambiguate among Wikidata candidates.

    Returns:
        The best matching candidate, or None if no good match.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    # Build candidate descriptions for the LLM
    candidate_text = "\n".join(
        f"  {i+1}. {c['qid']}: {c['label']} -- {c.get('description', 'no description')}"
        for i, c in enumerate(candidates)
    )

    prompt = f"""I have a local knowledge graph entity labeled "{local_label}".
I found these Wikidata candidates:

{candidate_text}

Which candidate (1-{len(candidates)}) is the best match for "{local_label}"?
If none match well, say "none".
Respond with just the number or "none"."""

    response = chat_completion(
        prompt=prompt,
        system="You are a knowledge graph entity disambiguation expert. Respond with just a number or 'none'.",
        provider=provider,
        model=model,
    )

    response = response.strip().lower()
    if response == "none":
        return None
    try:
        idx = int(response) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]
    except ValueError:
        pass

    # Default to first candidate if LLM response is unclear
    return candidates[0]


def fetch_wikidata_properties(qid: str) -> dict:
    """Fetch additional properties for a Wikidata entity.

    Returns:
        Dict with property labels and values.
    """
    sparql = SPARQLWrapper(WIKIDATA_ENDPOINT)
    sparql.addCustomHttpHeader("User-Agent", USER_AGENT)
    sparql.setReturnFormat(JSON)

    query = f"""
    SELECT ?propLabel ?value ?valueLabel
    WHERE {{
        wd:{qid} ?prop ?value .
        ?property wikibase:directClaim ?prop .
        ?property rdfs:label ?propLabel .
        FILTER(LANG(?propLabel) = "en")
        FILTER(
            ?prop IN (
                wdt:P31,   # instance of
                wdt:P569,  # date of birth
                wdt:P570,  # date of death
                wdt:P27,   # country of citizenship
                wdt:P19,   # place of birth
                wdt:P69,   # educated at
                wdt:P108,  # employer
                wdt:P166,  # award received
                wdt:P101,  # field of work
                wdt:P856,  # official website
                wdt:P571,  # inception (for orgs)
                wdt:P159,  # headquarters location
                wdt:P17,   # country
                wdt:P1454  # legal form
            )
        )
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 30
    """

    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
        properties = {}
        for binding in results["results"]["bindings"]:
            prop_label = binding["propLabel"]["value"]
            value_label = binding.get("valueLabel", {}).get("value", "")
            value_raw = binding["value"]["value"]

            if prop_label not in properties:
                properties[prop_label] = []
            properties[prop_label].append({
                "value": value_raw,
                "label": value_label,
            })
        return properties
    except Exception as e:
        print(f"    Property fetch error: {e}")
        return {}


def enrich_graph(
    g: RDFGraph,
    provider: str = "openai",
    model: str | None = None,
) -> tuple[RDFGraph, dict]:
    """Enrich the local graph with Wikidata data.

    Returns:
        Tuple of (enriched graph, enrichment report).
    """
    report = {"matched": [], "unmatched": [], "properties_added": 0}

    # Find all entities with labels
    entities = []
    for s in set(g.subjects()):
        label = g.value(s, RDFS.label) or g.value(s, FOAF.name)
        if label:
            # Determine type
            types = list(g.objects(s, RDF.type))
            entity_type = None
            for t in types:
                t_str = str(t)
                if "Person" in t_str:
                    entity_type = "Person"
                elif "Organization" in t_str:
                    entity_type = "Organization"
            entities.append((s, str(label), entity_type))

    print(f"\nFound {len(entities)} entities to enrich\n")

    for uri, label, entity_type in entities:
        print(f"  Enriching: {label} (type: {entity_type or 'unknown'})")

        # Search Wikidata
        candidates = search_wikidata_entity(label, entity_type)
        time.sleep(0.5)  # rate limiting

        if not candidates:
            print(f"    No Wikidata matches found")
            report["unmatched"].append(label)
            continue

        # Disambiguate
        match = disambiguate_entity(label, candidates, provider=provider, model=model)
        if not match:
            print(f"    No confident match")
            report["unmatched"].append(label)
            continue

        print(f"    Matched to: {match['qid']} ({match.get('description', '')})")

        # Add owl:sameAs link
        wikidata_uri = URIRef(match["uri"])
        g.add((uri, OWL.sameAs, wikidata_uri))

        # Fetch additional properties
        properties = fetch_wikidata_properties(match["qid"])
        time.sleep(0.5)

        props_added = 0
        for prop_label, values in properties.items():
            for val in values:
                # Add as enrichment properties
                prop_uri = EX[f"wikidata_{prop_label.replace(' ', '_')}"]
                value_text = val.get("label") or val.get("value", "")
                if value_text:
                    g.add((uri, prop_uri, Literal(value_text)))
                    props_added += 1

        print(f"    Added {props_added} properties from Wikidata")
        report["properties_added"] += props_added
        report["matched"].append({
            "local_label": label,
            "wikidata_qid": match["qid"],
            "wikidata_label": match.get("label", ""),
            "properties_added": props_added,
        })

    return g, report


def print_graph_comparison(before_count: int, after_count: int, report: dict) -> None:
    """Print before/after graph statistics."""
    print("\n" + "=" * 60)
    print("  ENRICHMENT RESULTS")
    print("=" * 60)

    print(f"\n  Before enrichment: {before_count} triples")
    print(f"  After enrichment:  {after_count} triples")
    print(f"  New triples added: {after_count - before_count}")

    print(f"\n  Entities matched:   {len(report['matched'])}")
    print(f"  Entities unmatched: {len(report['unmatched'])}")
    print(f"  Properties added:   {report['properties_added']}")

    if report["matched"]:
        print("\n  Matched entities:")
        for match in report["matched"]:
            print(f"    {match['local_label']:30s} -> wd:{match['wikidata_qid']} "
                  f"(+{match['properties_added']} properties)")

    if report["unmatched"]:
        print(f"\n  Unmatched entities: {', '.join(report['unmatched'])}")


def main():
    print("KG Enrichment from Wikidata\n")

    # Build local graph
    print("Building local knowledge graph...")
    g = build_local_graph()
    before_count = len(g)
    print(f"  Local graph: {before_count} triples")

    # Show initial entities
    print("\n  Entities in local graph:")
    for s in sorted(set(g.subjects()), key=str):
        label = g.value(s, RDFS.label) or g.value(s, FOAF.name) or g.value(s, SKOS.prefLabel)
        if label:
            types = [str(t).split("/")[-1].split("#")[-1] for t in g.objects(s, RDF.type)]
            print(f"    {label:35s}  types: {', '.join(types)}")

    # Enrich
    print("\nStarting enrichment process...")
    g, report = enrich_graph(g)
    after_count = len(g)

    # Print comparison
    print_graph_comparison(before_count, after_count, report)

    # Save enriched graph
    output_path = OUTPUT_DIR / "enriched_graph.ttl"
    g.serialize(destination=str(output_path), format="turtle")
    print(f"\n  Enriched graph saved to {output_path}")

    # Save report
    report_path = OUTPUT_DIR / "enrichment_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Enrichment report saved to {report_path}")

    # Show some enriched data
    print("\n--- Sample Enriched Data ---")
    for match_info in report.get("matched", [])[:3]:
        label = match_info["local_label"]
        print(f"\n  {label}:")
        # Find the entity URI
        for s in g.subjects(RDFS.label, Literal(label, lang="en")):
            for p, o in g.predicate_objects(s):
                p_label = str(p).split("/")[-1].split("#")[-1]
                if p_label.startswith("wikidata_"):
                    print(f"    {p_label:40s} = {o}")
            break

    print("\nDone.")


if __name__ == "__main__":
    main()
