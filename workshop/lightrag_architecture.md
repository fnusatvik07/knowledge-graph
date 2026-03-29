# LightRAG — Architecture Deep Dive

**Paper**: [Simple and Fast Retrieval-Augmented Generation](https://arxiv.org/abs/2410.05779) (EMNLP 2025)
**GitHub**: [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) — 30K+ stars
**Package**: `pip install lightrag-hku`

---

## What Is LightRAG?

LightRAG is a **Graph RAG framework** that automatically builds a knowledge graph from your documents and uses it for retrieval — without the expensive community detection step that Microsoft's GraphRAG requires.

Think of it as: **your entire manual KG pipeline (extract → build → query) packaged into 5 lines of code.**

```python
rag = LightRAG(working_dir="./storage")
await rag.ainsert("Your document text...")
answer = await rag.aquery("Your question", param=QueryParam(mode="hybrid"))
```

---

## Architecture Overview

```
                    ┌─────────────────────────────┐
                    │         DOCUMENTS            │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │       1. CHUNKING            │
                    │  Split into 1200-token chunks │
                    │  with 100-token overlap       │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │    2. ENTITY EXTRACTION       │
                    │  LLM reads each chunk         │
                    │  Extracts: entities + relations│
                    │  Uses "gleaning" for quality   │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │      3. PROFILING             │
                    │  Generate key-value pairs:    │
                    │  • Entity keys = names        │
                    │  • Relation keys = themes     │
                    │    enhanced by connected       │
                    │    entity context              │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │    4. DEDUPLICATION            │
                    │  Merge identical entities      │
                    │  from different chunks         │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │      5. EMBEDDING             │
                    │  Embed entity/relation         │
                    │  descriptions into vectors     │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │        STORAGE                │
                    │  • Graph: NetworkX / Neo4j     │
                    │  • Vectors: NanoVectorDB       │
                    │  • KV: JSON files              │
                    └─────────────────────────────┘
```

---

## Ingestion Pipeline — Step by Step

### Step 1: Chunking

Documents are split into chunks of **1200 tokens** with **100 token overlap**.

```
Document (10,000 tokens)
    │
    ├── Chunk 1: tokens 0-1200
    ├── Chunk 2: tokens 1100-2300   ← 100 token overlap
    ├── Chunk 3: tokens 2200-3400
    └── ... etc
```

**Why overlap?** Entities near chunk boundaries appear in both chunks, so they don't get lost.

### Step 2: Entity & Relation Extraction

Each chunk is sent to the LLM with a prompt like:

```
"Given the following text, extract all entities (people, organizations,
concepts) and relationships between them. Return as structured JSON."
```

LightRAG uses a **gleaning loop** — after the first extraction, it sends the results back to the LLM asking "did you miss anything?" This catches entities that were overlooked on the first pass.

Output per chunk:
```json
{
  "entities": [
    {"name": "Sam Altman", "type": "PERSON", "description": "CEO of OpenAI"}
  ],
  "relations": [
    {"source": "Sam Altman", "target": "OpenAI", "description": "leads the company"}
  ]
}
```

### Step 3: Profiling (Key-Value Generation)

This is **what replaces community detection**. For each entity and relation, LightRAG generates searchable key-value descriptions:

**Entities** → key = entity name

**Relations** → keys = LLM-enhanced keywords that include **global themes from connected entities**

Example: The relation `(Sam Altman)--[LEADS]-->(OpenAI)` gets keys like:
- "AI leadership"
- "LLM company management"
- "Silicon Valley tech CEO"

These enriched keys allow **global retrieval** without needing community summaries.

### Step 4: Deduplication

"Sam Altman" might appear in chunks 1, 3, and 7. LightRAG merges them into a single entity node, combining descriptions.

### Step 5: Embedding

All entity descriptions and relation descriptions are embedded into vectors using the configured embedding model (default: OpenAI `text-embedding-3-small`).

---

## Query Pipeline — How Retrieval Works

```
           User Question
                │
                ▼
    ┌───────────────────────┐
    │  LLM extracts keywords │
    │  from the question      │
    └───────────┬─────────────┘
                │
        ┌───────┴────────┐
        │                │
   ┌────▼─────┐    ┌─────▼────┐
   │  LOCAL    │    │  GLOBAL  │
   │ Search    │    │ Search   │
   │ entity    │    │ relation │
   │ nodes     │    │ themes   │
   └────┬──────┘    └─────┬────┘
        │                 │
        └────────┬────────┘
                 │
    ┌────────────▼────────────┐
    │  Combine & rank results  │
    │  (optional reranker)     │
    └────────────┬────────────┘
                 │
    ┌────────────▼────────────┐
    │  LLM generates answer    │
    │  from retrieved context   │
    └──────────────────────────┘
```

### Query Modes

| Mode | What It Searches | How | Best For |
|------|-----------------|-----|----------|
| **naive** | Raw text chunks | Vector similarity on chunk embeddings | Simple fact lookup ("When was X founded?") |
| **local** | Entity nodes + neighbors | Keywords → entity vector search → expand with neighbors | Specific questions ("What does X do?") |
| **global** | Relation-level themes | Keywords → relation vector search → retrieve connected entities | Broad questions ("What are the main themes?") |
| **hybrid** | Both local + global | Runs both, merges results | Best overall quality |
| **mix** | KG + vector chunks | Graph retrieval + chunk retrieval with reranker | When you need both structure and text |

### Local vs Global — What's the Difference?

**Local search**: "Tell me about Sam Altman"
1. Keywords extracted: ["Sam Altman"]
2. Search entity vectors → finds the Sam Altman node
3. Expand: get all entities connected to Sam Altman
4. Return: Sam Altman's description + all his relationships

**Global search**: "What are the main AI industry trends?"
1. Keywords extracted: ["AI industry", "trends", "technology landscape"]
2. Search relation vectors → finds relations tagged with these themes
3. Return: broad relationship-level information across the entire graph

**Hybrid**: runs both and merges.

---

## LightRAG vs GraphRAG — The Key Difference

```
GraphRAG:                              LightRAG:

  Entities + Relations                   Entities + Relations
       │                                      │
       ▼                                      ▼
  Leiden Community Detection ← EXPENSIVE   Profile with key-value ← CHEAP
       │                                      │
       ▼                                      ▼
  LLM writes summary for     ← EXPENSIVE   Embed descriptions     ← CHEAP
  each community (~5000                       │
  tokens per community)                       ▼
       │                                   Search vectors directly
       ▼
  Map-reduce over summaries
  for global queries
```

| Aspect | GraphRAG | LightRAG |
|--------|----------|----------|
| Community detection | Yes (Leiden algorithm) | **No** |
| Community summaries | Yes (~14M tokens for large corpus) | **No** |
| Indexing cost | ~$33,000 for large corpus | **~$0.50** |
| Incremental updates | Full rebuild needed | **Just add new entities** |
| Global query quality | Best (community summaries) | Good (relation-level themes) |
| Local query quality | Good | Good |
| Query latency | 2-5 seconds | **~80ms** |

---

## Supported Backends

LightRAG is **backend-agnostic** — swap any component:

### Graph Storage

| Backend | Setup | Best For |
|---------|-------|----------|
| **NetworkXStorage** (default) | No setup needed, file-based | Prototyping, small datasets |
| **Neo4JStorage** | `docker compose up -d` | Production, large graphs |
| **PGGraphStorage** | PostgreSQL | Existing Postgres infrastructure |
| **AGEStorage** | Apache AGE on PostgreSQL | Graph + relational hybrid |

### Vector Storage

| Backend | Setup | Best For |
|---------|-------|----------|
| **NanoVectorDBStorage** (default) | No setup, file-based | Prototyping |
| **ChromaVectorDBStorage** | `pip install chromadb` | Simple production |
| **MilvusVectorDBStorage** | Milvus server | Large scale |
| **FaissVectorDBStorage** | `pip install faiss-cpu` | High performance |
| **PGVectorStorage** | PostgreSQL + pgvector | Existing Postgres |
| **QdrantVectorDBStorage** | Qdrant server | Cloud-native |

### KV Storage

| Backend | Setup | Best For |
|---------|-------|----------|
| **JsonKVStorage** (default) | No setup, JSON files | Prototyping |
| **RedisKVStorage** | Redis server | Fast access |
| **MongoKVStorage** | MongoDB server | Document-oriented |
| **PGKVStorage** | PostgreSQL | Existing Postgres |

### LLM Providers

| Provider | Config |
|----------|--------|
| **OpenAI** (default) | `gpt_4o_mini_complete` |
| **Anthropic** | Custom function |
| **Ollama** (local) | Custom function with `ollama` package |
| **Any OpenAI-compatible** | Custom base URL |

---

## Configuration Options

```python
LightRAG(
    working_dir="./storage",           # where to save everything

    # LLM
    llm_model_func=gpt_4o_mini_complete,
    llm_model_max_async=4,             # concurrent LLM calls

    # Embedding
    embedding_func=openai_embed,
    embedding_batch_num=32,            # batch size

    # Chunking
    chunk_token_size=1200,             # tokens per chunk
    chunk_overlap_token_size=100,      # overlap between chunks

    # Extraction
    entity_extract_max_gleaning=1,     # refinement loops (0 = no gleaning)

    # Storage backends
    graph_storage="NetworkXStorage",   # or "Neo4JStorage"
    vector_storage="NanoVectorDBStorage",
    kv_storage="JsonKVStorage",

    # Cache
    enable_llm_cache=True,             # cache LLM responses
)
```

### Query Parameters

```python
QueryParam(
    mode="hybrid",                     # naive, local, global, hybrid, mix
    top_k=60,                          # entities/relations to retrieve
    stream=False,                      # streaming response
    response_type="Multiple Paragraphs",  # or "Bullet Points", "Single Paragraph"
    enable_rerank=False,               # use reranker model
)
```

---

## Incremental Updates

LightRAG's killer feature — add new documents without rebuilding:

```python
# Initial indexing
await rag.ainsert(document_1)
await rag.ainsert(document_2)

# Days later — just add more
await rag.ainsert(document_3)  # no rebuild, seamlessly merged

# Query includes knowledge from ALL documents
answer = await rag.aquery("...", param=QueryParam(mode="hybrid"))
```

What happens internally:
1. New document is chunked
2. Entities/relations extracted from new chunks
3. New entities **merged** with existing graph (deduplication)
4. New embeddings added to vector index
5. Existing data is **untouched**

GraphRAG cannot do this — any new document requires rebuilding all community structures and regenerating all community summaries.

---

## Performance Benchmarks (from the paper)

Evaluated on 4 datasets. Win rates vs NaiveRAG (standard vector RAG):

| Dataset | Comprehensiveness | Diversity | Overall |
|---------|------------------|-----------|---------|
| Agriculture | 67.6% | 76.4% | 67.6% |
| CS | 61.6% | 62.0% | 61.2% |
| Legal | 83.6% | 86.4% | 84.8% |
| Mixed | 61.2% | 67.6% | ~61% |

LightRAG also outperformed GraphRAG, RQ-RAG, and HyDE on all datasets.

**Key finding from ablation**: Removing original text chunks from context surprisingly did NOT significantly hurt performance — meaning the knowledge graph alone captures most of the needed information.

---

## Recommendations from the Authors

- Use an LLM with **32B+ parameters** for entity extraction
- Context length should be **32K minimum**, ideally 64K
- **Avoid reasoning models** during indexing (they overthink)
- Recommended embedding: `BAAI/bge-m3` or `text-embedding-3-large`
- Recommended reranker: `BAAI/bge-reranker-v2-m3`
