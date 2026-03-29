# KG Embeddings for RAG Systems

## Overview

Knowledge graph embeddings and text embeddings serve complementary roles in Retrieval-Augmented Generation (RAG). Text embeddings (from models like OpenAI's text-embedding-ada-002 or Sentence-BERT) capture semantic similarity of natural language. Graph embeddings (from TransE, RotatE, etc.) capture structural relationships between entities in a knowledge graph.

Combining both creates more powerful retrieval systems that understand both what things mean and how they relate.

---

## Entity Embeddings as Retrieval Features

### The Core Idea

In a KG-augmented RAG pipeline, entity embeddings provide a parallel retrieval channel alongside text embeddings:

```
User Query
    |
    +---> Text Embedding --> Vector Store --> Text Chunks
    |
    +---> Entity Extraction --> Entity Embedding --> KG Neighbors --> Structured Context
    |
    +---> Merge Results --> LLM --> Answer
```

### Using Entity Embeddings for Candidate Retrieval

Pre-compute embeddings for all entities, then use them to find relevant entities at query time:

```python
import numpy as np
from pykeen.predict import predict_target

# Pre-compute: extract all entity embeddings
entity_emb = model.entity_representations[0]().detach().cpu().numpy()
entity_names = list(training_factory.entity_to_id.keys())

# At query time: given a known entity, find similar entities
def find_similar_entities(entity_name, top_k=10):
    entity_id = training_factory.entity_to_id[entity_name]
    query_emb = entity_emb[entity_id]

    # Cosine similarity
    similarities = entity_emb @ query_emb / (
        np.linalg.norm(entity_emb, axis=1) * np.linalg.norm(query_emb)
    )

    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [(entity_names[i], similarities[i]) for i in top_indices]

similar = find_similar_entities("Python_(programming_language)")
# Returns: [("Java_(programming_language)", 0.92), ("Ruby_(programming_language)", 0.89), ...]
```

### Building a FAISS Index over Entity Embeddings

For large KGs, use FAISS for efficient nearest-neighbor search:

```python
import faiss

# Normalize embeddings for cosine similarity
entity_emb_normalized = entity_emb / np.linalg.norm(entity_emb, axis=1, keepdims=True)

# Build FAISS index
dimension = entity_emb_normalized.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner product = cosine on normalized vectors
index.add(entity_emb_normalized.astype(np.float32))

# Query
query_vector = entity_emb_normalized[entity_id].reshape(1, -1).astype(np.float32)
distances, indices = index.search(query_vector, k=10)
```

---

## Embedding-Based Entity Disambiguation

When a user mentions "Apple," do they mean the company or the fruit? Entity embeddings help disambiguate:

```python
def disambiguate_entity(mention, context_entities, candidates):
    """
    Given a mention and surrounding context entities,
    pick the candidate whose embedding is closest to the context.

    Args:
        mention: ambiguous entity name (e.g., "Apple")
        context_entities: entities already identified in the query
        candidates: possible entities (e.g., ["Apple_Inc.", "Apple_(fruit)"])
    """
    # Average embedding of context entities
    context_ids = [training_factory.entity_to_id[e] for e in context_entities]
    context_emb = entity_emb[context_ids].mean(axis=0)

    best_score = -1
    best_candidate = None
    for candidate in candidates:
        cand_id = training_factory.entity_to_id[candidate]
        cand_emb = entity_emb[cand_id]
        score = np.dot(context_emb, cand_emb) / (
            np.linalg.norm(context_emb) * np.linalg.norm(cand_emb)
        )
        if score > best_score:
            best_score = score
            best_candidate = candidate

    return best_candidate

# Example: "Apple" with context about technology
result = disambiguate_entity(
    mention="Apple",
    context_entities=["iPhone", "Steve_Jobs", "Silicon_Valley"],
    candidates=["Apple_Inc.", "Apple_(fruit)"]
)
# Returns: "Apple_Inc."
```

---

## Combining Text Embeddings with Graph Embeddings

### LangChain Integration Pattern

Use LangChain for text-based retrieval and KG embeddings for structure-based retrieval, then merge:

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain_community.graphs import Neo4jGraph

# Text retrieval via LangChain
text_embeddings = OpenAIEmbeddings()
text_store = LangchainFAISS.from_documents(documents, text_embeddings)

# Graph retrieval via KG embeddings
def kg_retrieve(query_entities, top_k=5):
    """Retrieve related entities using KG embeddings."""
    related = set()
    for entity in query_entities:
        similar = find_similar_entities(entity, top_k=top_k)
        related.update([name for name, score in similar if score > 0.7])
    return related

# Combined retrieval
def hybrid_retrieve(query, top_k=5):
    # Step 1: Text-based retrieval
    text_results = text_store.similarity_search(query, k=top_k)

    # Step 2: Extract entities from query (via NER or entity linking)
    query_entities = extract_entities(query)

    # Step 3: KG embedding-based retrieval
    kg_entities = kg_retrieve(query_entities, top_k=top_k)

    # Step 4: Fetch KG context for related entities
    kg_context = fetch_subgraph(kg_entities)

    # Step 5: Merge
    return {
        "text_chunks": text_results,
        "kg_context": kg_context,
    }
```

### Concatenated Embedding Approach

For entities that have both text descriptions and graph structure, create a unified embedding:

```python
from sentence_transformers import SentenceTransformer

text_model = SentenceTransformer("all-MiniLM-L6-v2")

def create_hybrid_embedding(entity_name, entity_description):
    """Create a combined text + graph embedding."""
    # Text embedding from description
    text_emb = text_model.encode(entity_description)
    text_emb = text_emb / np.linalg.norm(text_emb)

    # Graph embedding from KG
    entity_id = training_factory.entity_to_id.get(entity_name)
    if entity_id is not None:
        graph_emb = entity_emb[entity_id]
        graph_emb = graph_emb / np.linalg.norm(graph_emb)
    else:
        graph_emb = np.zeros(entity_emb.shape[1])

    # Weighted concatenation
    alpha = 0.6  # weight for text
    return np.concatenate([alpha * text_emb, (1 - alpha) * graph_emb])
```

---

## When to Use Graph Embeddings vs Text Embeddings vs Both

| Scenario | Text Embeddings | Graph Embeddings | Both |
|----------|----------------|-----------------|------|
| Free-text QA over documents | Best | Not applicable | -- |
| Entity-centric QA (who, what) | Good | Better | Best |
| Relational queries (how X relates to Y) | Weak | Best | Best |
| Entity disambiguation | Moderate | Good | Best |
| Link prediction (missing facts) | Not applicable | Best | -- |
| Semantic search over KG | Good | Good | Best |
| Cold-start entities (no graph data) | Best | Not applicable | Text only |

### Decision Guide

1. **Text embeddings only**: Your data is primarily unstructured text, entities are not well-defined, or your KG is too small for meaningful embeddings
2. **Graph embeddings only**: Your task is purely structural (link prediction, entity clustering), and you have a large, well-curated KG
3. **Both**: You have a KG with entity descriptions, your queries mix factual and relational questions, and you need high-quality entity disambiguation

---

## Production Patterns

### Pre-compute Entity Embeddings

Entity embeddings should be computed offline and stored for fast retrieval:

```python
# Training pipeline (offline, runs periodically)
def train_and_export_embeddings(triples_path, output_dir):
    from pykeen.pipeline import pipeline

    result = pipeline(
        training=triples_path,
        model="RotatE",
        model_kwargs=dict(embedding_dim=200),
        training_kwargs=dict(num_epochs=200),
    )

    # Export embeddings
    entity_emb = result.model.entity_representations[0]().detach().cpu().numpy()
    entity_names = list(result.training.entity_to_id.keys())

    np.save(f"{output_dir}/entity_embeddings.npy", entity_emb)
    with open(f"{output_dir}/entity_names.json", "w") as f:
        json.dump(entity_names, f)

    # Build FAISS index
    emb_normalized = entity_emb / np.linalg.norm(entity_emb, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(emb_normalized.shape[1])
    index.add(emb_normalized.astype(np.float32))
    faiss.write_index(index, f"{output_dir}/entity_index.faiss")
```

### Serving Architecture

```
                     +-------------------+
  User Query ------> | Query Processor   |
                     | (NER + embedding) |
                     +--------+----------+
                              |
              +---------------+---------------+
              |                               |
     +--------v--------+           +----------v---------+
     | Text Vector      |           | Entity Embedding    |
     | Store (FAISS)     |           | Index (FAISS)       |
     +--------+---------+           +----------+----------+
              |                                |
              +---------------+----------------+
                              |
                     +--------v--------+
                     | Result Merger   |
                     | & Re-ranker     |
                     +--------+--------+
                              |
                     +--------v--------+
                     | LLM (with both  |
                     | text + KG ctx)  |
                     +-----------------+
```

### Incremental Updates

When new entities or triples are added to the KG:

1. **Small additions** (< 1% of KG): Re-embed only affected entities using warm-start from existing model
2. **Large additions** (> 1%): Retrain from scratch with existing weights as initialization
3. **Schedule**: Retrain embeddings on a cadence matching your KG update frequency (daily/weekly)

```python
# Warm-start: load existing model, continue training with new triples
from pykeen.models import RotatE

model = RotatE(
    triples_factory=new_training_factory,
    embedding_dim=200,
)
# Load weights from previous model where entity/relation IDs overlap
model.load_state_dict(old_state_dict, strict=False)
```

---

## LangChain Graph Integration

LangChain provides built-in graph store integrations that can be combined with embeddings.

Reference: https://python.langchain.com/docs/integrations/graphs/

```python
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI

# LangChain handles the Cypher generation
graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="password")
chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(model="gpt-4"),
    graph=graph,
    verbose=True,
)

# Your entity embeddings handle disambiguation and candidate retrieval
# before the query reaches the Cypher generation step
```

---

## Key Takeaways

1. **KG embeddings and text embeddings are complementary** -- use both when you have structured and unstructured data
2. **Pre-compute and index entity embeddings** for low-latency retrieval in production
3. **Entity disambiguation is a killer app** for KG embeddings in RAG pipelines
4. **Start with RotatE** for general-purpose entity embeddings (best balance of expressiveness and speed)
5. **LangChain's graph integrations** handle Cypher/SPARQL generation; your embeddings handle the structural reasoning layer beneath

---

## References

- LangChain Graph Integrations: https://python.langchain.com/docs/integrations/graphs/
- FAISS: https://github.com/facebookresearch/faiss
- PyKEEN: https://pykeen.readthedocs.io/
- Sentence-BERT: https://www.sbert.net/
