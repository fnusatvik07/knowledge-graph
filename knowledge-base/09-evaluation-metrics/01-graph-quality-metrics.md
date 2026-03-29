# Graph Quality Metrics

How do you know if your knowledge graph is any good? This section covers the metrics, formulas, and code for evaluating KG quality across multiple dimensions: extraction accuracy, structural integrity, and coverage.

## Dimensions of Graph Quality

| Dimension | What It Measures | Key Metrics |
|-----------|-----------------|-------------|
| **Extraction Accuracy** | Are the entities and relations correct? | Precision, Recall, F1 |
| **Entity Resolution** | Are duplicates merged correctly? | Pairwise F1, cluster purity |
| **Schema Conformance** | Does the graph follow the ontology? | Conformance rate, violation count |
| **Completeness** | Does the graph capture all relevant knowledge? | Coverage ratio, missing entity rate |
| **Freshness** | Is the graph up to date? | Staleness score, update latency |

## Entity Extraction Metrics

### Precision, Recall, and F1

The fundamental metrics for evaluating entity extraction quality.

```
Precision = True Positives / (True Positives + False Positives)
           "Of all entities extracted, how many are correct?"

Recall    = True Positives / (True Positives + False Negatives)
           "Of all entities that should be extracted, how many were found?"

F1        = 2 * (Precision * Recall) / (Precision + Recall)
           "Harmonic mean of precision and recall"
```

### Computing Extraction Metrics

```python
from dataclasses import dataclass

@dataclass
class ExtractionMetrics:
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int

def compute_extraction_metrics(
    extracted: set[str],
    ground_truth: set[str],
    normalize: bool = True
) -> ExtractionMetrics:
    """Compute precision, recall, F1 for entity extraction.

    Args:
        extracted: Set of extracted entity names
        ground_truth: Set of ground truth entity names
        normalize: If True, lowercase and strip all names before comparison
    """
    if normalize:
        extracted = {e.lower().strip() for e in extracted}
        ground_truth = {g.lower().strip() for g in ground_truth}

    tp = len(extracted & ground_truth)
    fp = len(extracted - ground_truth)
    fn = len(ground_truth - extracted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return ExtractionMetrics(
        precision=precision, recall=recall, f1=f1,
        true_positives=tp, false_positives=fp, false_negatives=fn
    )

# Example usage
extracted_entities = {"Marie Curie", "University of Paris", "Nobel Prize", "France", "radium"}
ground_truth_entities = {"Marie Curie", "University of Paris", "Nobel Prize in Physics", "Pierre Curie"}

metrics = compute_extraction_metrics(extracted_entities, ground_truth_entities)
print(f"Precision: {metrics.precision:.2f}")  # 0.40 (2/5)
print(f"Recall: {metrics.recall:.2f}")        # 0.50 (2/4)
print(f"F1: {metrics.f1:.2f}")                # 0.44
```

### Type-Aware Metrics

Evaluate extraction per entity type for more granular insight:

```python
from collections import defaultdict

def compute_typed_metrics(
    extracted: list[dict],   # [{"name": "...", "type": "PERSON"}, ...]
    ground_truth: list[dict]
) -> dict[str, ExtractionMetrics]:
    """Compute metrics per entity type."""
    extracted_by_type = defaultdict(set)
    truth_by_type = defaultdict(set)

    for e in extracted:
        extracted_by_type[e["type"]].add(e["name"].lower().strip())
    for g in ground_truth:
        truth_by_type[g["type"]].add(g["name"].lower().strip())

    all_types = set(extracted_by_type.keys()) | set(truth_by_type.keys())
    results = {}
    for t in all_types:
        results[t] = compute_extraction_metrics(
            extracted_by_type[t], truth_by_type[t], normalize=False
        )
    return results

# Example
typed_metrics = compute_typed_metrics(
    [{"name": "Marie Curie", "type": "PERSON"}, {"name": "France", "type": "LOCATION"}],
    [{"name": "Marie Curie", "type": "PERSON"}, {"name": "Pierre Curie", "type": "PERSON"}]
)
for entity_type, m in typed_metrics.items():
    print(f"  {entity_type}: P={m.precision:.2f} R={m.recall:.2f} F1={m.f1:.2f}")
```

## Relationship Accuracy

Relationships are harder to evaluate because they involve matching source, target, and type.

### Relationship Matching

```python
from typing import NamedTuple

class RelTriple(NamedTuple):
    source: str
    relation: str
    target: str

def normalize_triple(t: RelTriple) -> RelTriple:
    return RelTriple(
        t.source.lower().strip(),
        t.relation.lower().strip().replace(" ", "_"),
        t.target.lower().strip()
    )

def compute_relationship_metrics(
    extracted: list[RelTriple],
    ground_truth: list[RelTriple]
) -> ExtractionMetrics:
    """Compute metrics for relationship extraction."""
    ext_set = {normalize_triple(t) for t in extracted}
    gt_set = {normalize_triple(t) for t in ground_truth}
    return compute_extraction_metrics(ext_set, gt_set, normalize=False)

# Example
extracted_rels = [
    RelTriple("Marie Curie", "WORKED_AT", "University of Paris"),
    RelTriple("Marie Curie", "DISCOVERED", "Radium"),
    RelTriple("Marie Curie", "BORN_IN", "Warsaw"),  # Correct
]
truth_rels = [
    RelTriple("Marie Curie", "WORKED_AT", "University of Paris"),
    RelTriple("Marie Curie", "BORN_IN", "Warsaw"),
    RelTriple("Marie Curie", "MARRIED_TO", "Pierre Curie"),  # Missed
]

rel_metrics = compute_relationship_metrics(extracted_rels, truth_rels)
print(f"Relationship F1: {rel_metrics.f1:.2f}")
```

## Entity Resolution Metrics

Entity resolution (deduplication) quality measures how well your system merges duplicate references to the same real-world entity.

### Pairwise F1

```python
def entity_resolution_f1(
    predicted_clusters: list[set[str]],
    ground_truth_clusters: list[set[str]]
) -> dict:
    """Evaluate entity resolution using pairwise precision/recall/F1.

    Each cluster is a set of mentions that refer to the same entity.
    """
    def get_pairs(clusters):
        pairs = set()
        for cluster in clusters:
            members = sorted(cluster)
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    pairs.add((members[i], members[j]))
        return pairs

    pred_pairs = get_pairs(predicted_clusters)
    true_pairs = get_pairs(ground_truth_clusters)

    tp = len(pred_pairs & true_pairs)
    fp = len(pred_pairs - true_pairs)
    fn = len(true_pairs - pred_pairs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}

# Example: "M. Curie", "Marie Curie", "Curie" should all be in one cluster
predicted = [{"Marie Curie", "M. Curie", "Curie"}, {"Pierre Curie", "P. Curie"}]
ground_truth = [{"Marie Curie", "M. Curie", "Curie", "Madame Curie"}, {"Pierre Curie", "P. Curie"}]

er_metrics = entity_resolution_f1(predicted, ground_truth)
print(f"Entity Resolution Pairwise F1: {er_metrics['f1']:.2f}")
```

## Ontology Conformance

Measure how well the constructed graph adheres to the defined ontology.

```python
def check_ontology_conformance(graph_data, ontology) -> dict:
    """Check if graph nodes and relationships conform to the ontology.

    Args:
        graph_data: dict with "nodes" and "relationships"
        ontology: dict with "valid_node_types", "valid_rel_types",
                  "valid_rel_pairs" (list of (source_type, rel_type, target_type))
    """
    violations = []
    total_nodes = len(graph_data["nodes"])
    total_rels = len(graph_data["relationships"])

    # Check node types
    invalid_node_types = 0
    for node in graph_data["nodes"]:
        if node["type"] not in ontology["valid_node_types"]:
            violations.append(f"Invalid node type: {node['type']} for '{node['name']}'")
            invalid_node_types += 1

    # Check relationship types and valid source-target pairs
    invalid_rels = 0
    for rel in graph_data["relationships"]:
        if rel["type"] not in ontology["valid_rel_types"]:
            violations.append(f"Invalid relationship type: {rel['type']}")
            invalid_rels += 1
        else:
            triple = (rel["source_type"], rel["type"], rel["target_type"])
            if triple not in ontology["valid_rel_pairs"]:
                violations.append(f"Invalid relationship pattern: {triple}")
                invalid_rels += 1

    node_conformance = (total_nodes - invalid_node_types) / total_nodes if total_nodes > 0 else 1.0
    rel_conformance = (total_rels - invalid_rels) / total_rels if total_rels > 0 else 1.0

    return {
        "node_conformance": node_conformance,
        "relationship_conformance": rel_conformance,
        "overall_conformance": (node_conformance + rel_conformance) / 2,
        "violations": violations,
        "violation_count": len(violations),
    }
```

## Coverage Analysis

Measure how much of the source material is captured in the graph.

```python
def coverage_analysis(
    source_documents: list[str],
    graph_entities: set[str],
    llm=None
) -> dict:
    """Estimate graph coverage over source documents.

    Uses an LLM to extract a quick entity list from each document,
    then checks how many appear in the graph.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    normalized_graph = {e.lower().strip() for e in graph_entities}
    total_expected = 0
    found_in_graph = 0

    for doc in source_documents:
        # Quick entity extraction for coverage check
        response = llm.invoke([HumanMessage(
            content=f"List all named entities in this text, one per line:\n\n{doc[:2000]}"
        )])
        doc_entities = {e.strip().lower() for e in response.content.split('\n') if e.strip()}
        total_expected += len(doc_entities)
        found_in_graph += len(doc_entities & normalized_graph)

    coverage = found_in_graph / total_expected if total_expected > 0 else 0.0
    return {
        "coverage_ratio": coverage,
        "entities_found": found_in_graph,
        "entities_expected": total_expected,
    }
```

> **LangChain ChatOpenAI**: https://python.langchain.com/docs/integrations/chat/openai/

## Graph Structural Metrics

Beyond content quality, evaluate the graph's structural properties:

```python
import networkx as nx

def structural_metrics(G: nx.Graph) -> dict:
    """Compute structural quality metrics for a knowledge graph."""
    metrics = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": nx.density(G),
        "num_connected_components": nx.number_connected_components(G.to_undirected())
            if G.is_directed() else nx.number_connected_components(G),
        "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes()
            if G.number_of_nodes() > 0 else 0,
    }

    # Largest connected component ratio (ideally close to 1.0)
    if G.is_directed():
        largest_cc = max(nx.weakly_connected_components(G), key=len)
    else:
        largest_cc = max(nx.connected_components(G), key=len)
    metrics["largest_component_ratio"] = len(largest_cc) / G.number_of_nodes()

    # Isolated nodes (entities with no relationships -- usually a problem)
    metrics["isolated_nodes"] = len(list(nx.isolates(G)))
    metrics["isolated_node_ratio"] = metrics["isolated_nodes"] / G.number_of_nodes() \
        if G.number_of_nodes() > 0 else 0

    return metrics
```

## Putting It All Together: A Quality Report

```python
def generate_quality_report(
    extracted_entities, truth_entities,
    extracted_rels, truth_rels,
    graph: nx.DiGraph,
    ontology: dict
) -> dict:
    """Generate a comprehensive quality report for a knowledge graph."""
    entity_metrics = compute_extraction_metrics(
        {e["name"] for e in extracted_entities},
        {e["name"] for e in truth_entities}
    )
    rel_metrics = compute_relationship_metrics(extracted_rels, truth_rels)
    conformance = check_ontology_conformance(
        {"nodes": extracted_entities, "relationships": extracted_rels},
        ontology
    )
    structure = structural_metrics(graph)

    return {
        "entity_extraction": {"precision": entity_metrics.precision, "recall": entity_metrics.recall, "f1": entity_metrics.f1},
        "relationship_extraction": {"precision": rel_metrics.precision, "recall": rel_metrics.recall, "f1": rel_metrics.f1},
        "ontology_conformance": conformance["overall_conformance"],
        "violations": conformance["violation_count"],
        "structural": structure,
    }
```

## Target Benchmarks

| Metric | Poor | Acceptable | Good | Excellent |
|--------|------|-----------|------|-----------|
| Entity Extraction F1 | < 0.5 | 0.5-0.7 | 0.7-0.85 | > 0.85 |
| Relationship Extraction F1 | < 0.4 | 0.4-0.6 | 0.6-0.8 | > 0.8 |
| Ontology Conformance | < 0.8 | 0.8-0.9 | 0.9-0.95 | > 0.95 |
| Largest Component Ratio | < 0.5 | 0.5-0.7 | 0.7-0.9 | > 0.9 |
| Isolated Node Ratio | > 0.2 | 0.1-0.2 | 0.05-0.1 | < 0.05 |

## Next Steps

- [02 - RAG Evaluation](./02-rag-evaluation.md) -- evaluating the downstream RAG system
- [LLM Extraction Patterns](../02-kg-construction/04-llm-extraction-patterns.md) -- improve extraction quality
