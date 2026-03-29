# Microsoft GraphRAG: From Local to Global

## Paper Overview

**Title:** From Local to Global: A GraphRAG Approach to Query-Focused Summarization
**Authors:** Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Jonathan Larson (Microsoft Research)
**Published:** April 2024
**Repository:** [microsoft/graphrag](https://github.com/microsoft/graphrag)

GraphRAG is Microsoft Research's answer to a fundamental limitation of standard vector-based
Retrieval-Augmented Generation: the inability to answer **global, sensemaking queries** over
large document corpora. The paper introduces a knowledge-graph-powered pipeline that
combines entity/relationship extraction, community detection, and hierarchical summarization
to enable both local (specific) and global (thematic) question answering.

---

## The Problem: Why Vector RAG Fails at Global Queries

Traditional RAG systems work by:

1. Chunking documents into text segments
2. Embedding those segments into a vector space
3. Retrieving the top-k most similar chunks to a query
4. Feeding those chunks to an LLM for answer generation

This approach works well for **specific, fact-seeking queries** such as:

- "What is the capital of France?"
- "When was the company founded?"
- "What did the CEO say about revenue in Q3?"

However, it **systematically fails** at queries that require synthesizing information across
many documents or understanding the corpus as a whole:

- "What are the main themes in this dataset?"
- "How do the different research groups relate to each other?"
- "Summarize the key controversies across all community discussions."

### Why the Failure Occurs

| Failure Mode | Explanation |
|---|---|
| **Needle-in-a-haystack assumption** | Vector search assumes the answer lives in a small number of chunks. Global queries require aggregation across hundreds or thousands of chunks. |
| **No structural awareness** | Embeddings capture semantic similarity but not structural relationships between entities, communities, or themes. |
| **Context window limits** | Even with large context windows, naive stuffing of all relevant chunks is prohibitively expensive and often incoherent. |
| **No hierarchy** | Vector RAG has no mechanism for operating at different levels of abstraction. |

---

## The Solution: Knowledge Graph + Community Detection + Hierarchical Summarization

GraphRAG addresses these failures through a multi-stage architecture:

```
Documents
    |
    v
[Text Chunking]
    |
    v
[Entity & Relationship Extraction via LLM]
    |
    v
[Knowledge Graph Construction]
    |
    v
[Leiden Community Detection]
    |
    v
[Hierarchical Community Summarization]
    |
    v
[Query-Time Map-Reduce Generation]
```

The key insight is that a knowledge graph captures **structural relationships** between
entities, and community detection algorithms can identify **natural clusters** of related
entities. These clusters can then be summarized at multiple levels of granularity, enabling
answers to both specific and global queries.

---

## The Indexing Pipeline (Detailed)

### Stage 1: Text Unit Extraction

Source documents are split into overlapping text chunks called **text units**. These serve as
the atomic units of provenance -- every claim in the final output can be traced back to
specific text units.

- Default chunk size: 300 tokens (configurable)
- Overlap between chunks ensures no information is lost at boundaries
- Each text unit retains metadata about its source document

### Stage 2: Entity Extraction

An LLM (typically GPT-4 class) processes each text unit to extract **named entities** with:

- **Entity name** (normalized/canonical form)
- **Entity type** (person, organization, location, event, concept, etc.)
- **Entity description** (a brief summary of what the entity is)

The extraction prompt is domain-adaptive -- entity types can be customized for the corpus.
Multiple extraction passes (gleaning) can be configured to improve recall.

### Stage 3: Relationship Extraction

The same LLM pass also extracts **relationships** between entities:

- **Source entity**
- **Target entity**
- **Relationship description** (the nature of the connection)
- **Relationship strength** (a weight indicating importance)

### Stage 4: Graph Construction

Extracted entities and relationships are assembled into a knowledge graph:

- Duplicate entities are merged (via name normalization and description aggregation)
- Duplicate relationships are merged (descriptions concatenated, weights summed)
- The result is a weighted, undirected graph where nodes are entities and edges are
  relationships

### Stage 5: Leiden Community Detection

The **Leiden algorithm** partitions the graph into communities of densely interconnected
entities. This is applied hierarchically, producing communities at multiple resolutions:

- **Level 0:** Fine-grained communities (small clusters of tightly related entities)
- **Level 1:** Mid-level communities (aggregations of Level 0 communities)
- **Level 2+:** Coarse-grained communities (broad thematic groupings)

See [02-community-detection-leiden.md](./02-community-detection-leiden.md) for a deep dive.

### Stage 6: Community Summarization

Each community at each level receives an LLM-generated summary that captures:

- **Title:** A descriptive name for the community
- **Summary:** An overview of the community's key entities and relationships
- **Key findings:** Ranked claims supported by the community's entities
- **Rating:** An importance score (0-10) with a rationale

These summaries are the core data structure that enables global query answering.

---

## Query-Time: Map-Reduce Generation

When a global query arrives, GraphRAG uses a **map-reduce** approach:

### Map Phase

1. Select the appropriate community level based on query scope
2. Retrieve all community summaries at that level
3. For each community summary, the LLM generates a **partial answer** (or "not relevant")
4. Each partial answer includes a relevance score (0-100)

### Reduce Phase

1. Partial answers are sorted by relevance score (descending)
2. Answers are iteratively added to the context window until the token budget is exhausted
3. The LLM generates a **final synthesized answer** from the aggregated partial answers

```
Query: "What are the main themes in this dataset?"
                    |
                    v
    +------+------+------+------+
    | Comm | Comm | Comm | Comm |  ... (all community summaries)
    |  1   |  2   |  3   |  4   |
    +------+------+------+------+
        |      |      |      |
        v      v      v      v
    [Map: Generate partial answer from each community]
        |      |      |      |
        v      v      v      v
    +------+------+------+------+
    |Partial|Partial|Partial| N/A |  (some communities may be irrelevant)
    |Ans 1  |Ans 2  |Ans 3  |     |
    +------+------+------+------+
                    |
                    v
    [Reduce: Synthesize partial answers into final answer]
                    |
                    v
            Final Answer
```

---

## Results and Evaluation

The paper evaluates GraphRAG against baseline RAG (naive vector retrieval) using the
following metrics (assessed by LLM-as-judge):

### Comprehensiveness

How thoroughly the answer covers all aspects of the question.

- **GraphRAG improvement: ~26% over naive RAG**
- Community summaries capture cross-document themes that vector retrieval misses

### Diversity

How many different perspectives or dimensions the answer addresses.

- **GraphRAG improvement: ~57% over naive RAG**
- Hierarchical communities naturally surface diverse facets of complex topics

### Empowerment

How well the answer enables the user to take informed action.

- GraphRAG shows consistent improvement, particularly for sensemaking tasks

### Directness

How directly the answer addresses the question.

- Comparable to naive RAG for specific queries; superior for global queries

### Win Rates by Community Level

| Level | Comprehensiveness Win | Diversity Win |
|---|---|---|
| Level 0 (fine) | 56% | 57% |
| Level 1 (mid) | 72% | 78% |
| Level 2 (coarse) | 80% | 82% |

Higher community levels (coarser granularity) perform better for global queries because
they aggregate more information.

---

## January 2025 Update: Dynamic Community Selection (v2)

### The Problem with v1 Global Search

The original map-reduce approach sends **every** community summary to the LLM during the
map phase. For large corpora with thousands of communities, this is:

- **Expensive:** Thousands of LLM calls per query
- **Slow:** Linear scaling with corpus size
- **Wasteful:** Most communities are irrelevant to any given query

### Dynamic Community Selection

The January 2025 update introduces **Dynamic Community Selection**, which pre-filters
communities before the map phase:

1. Community summaries are embedded into a vector space
2. At query time, the query embedding is compared against community summary embeddings
3. Only the top-k most relevant communities proceed to the map phase
4. The map-reduce pipeline then operates over a much smaller set

### Results

- **79% reduction in token usage** compared to v1 global search
- **Comparable answer quality** (no statistically significant degradation)
- **Faster response times** due to fewer LLM calls
- Effectively makes global search viable for very large corpora (millions of documents)

### Additional v2 Improvements

- **DRIFT search mode:** A hybrid search that combines local and global strategies
  (see [03-local-vs-global-vs-drift.md](./03-local-vs-global-vs-drift.md))
- **Incremental indexing:** Update the graph without full re-indexing
- **Non-OpenAI model support:** Ollama, Azure, and other providers
- **Improved prompt tuning:** Auto-adapt extraction prompts to the domain

---

## Cost and Practical Considerations

### Indexing Cost

GraphRAG indexing is **significantly more expensive** than vector RAG indexing because
every text unit requires LLM processing for entity/relationship extraction:

| Corpus Size | Approximate Indexing Cost (GPT-4) | Vector RAG Indexing Cost |
|---|---|---|
| 10K tokens | ~$0.50 | ~$0.01 |
| 1M tokens | ~$50 | ~$0.50 |
| 100M tokens | ~$5,000 | ~$50 |
| 1B tokens | ~$33,000+ | ~$500 |

### When to Use GraphRAG

GraphRAG is most valuable when:

- Users need to ask **global, thematic, or sensemaking queries**
- The corpus contains **rich entity relationships** (not just isolated facts)
- **Comprehensiveness and diversity** of answers are critical
- The corpus is relatively stable (amortizing indexing cost over many queries)

GraphRAG is **less justified** when:

- All queries are simple fact lookups (vector RAG suffices)
- The corpus changes rapidly (re-indexing is expensive)
- Budget constraints are tight
- The corpus is small enough to fit in a single context window

---

## Architecture Diagram

```
                    +------------------+
                    |   Source Docs    |
                    +--------+---------+
                             |
                    +--------v---------+
                    |  Text Chunking   |
                    +--------+---------+
                             |
                    +--------v---------+
                    | Entity/Relation  |
                    |   Extraction     |
                    |   (LLM-based)    |
                    +--------+---------+
                             |
                    +--------v---------+
                    | Knowledge Graph  |
                    |  Construction    |
                    +--------+---------+
                             |
                    +--------v---------+
                    | Leiden Community  |
                    |   Detection      |
                    +--------+---------+
                             |
              +--------------+--------------+
              |              |              |
     +--------v---+  +------v-----+  +-----v------+
     | Level 0    |  | Level 1    |  | Level 2+   |
     | Communities|  | Communities|  | Communities|
     +--------+---+  +------+-----+  +-----+------+
              |              |              |
     +--------v---+  +------v-----+  +-----v------+
     | Summaries  |  | Summaries  |  | Summaries  |
     +------------+  +------------+  +------------+
              \              |              /
               +-------------+-------------+
                             |
                    +--------v---------+
                    |  Query Engine    |
                    |  (Local/Global/  |
                    |   DRIFT Search)  |
                    +------------------+
```

---

## Key Takeaways

1. **GraphRAG solves a real gap** in traditional RAG -- global sensemaking queries.
2. **Community detection is the secret sauce** -- it transforms a flat graph into a
   hierarchical structure that can be summarized at multiple levels.
3. **The cost is substantial** -- LLM-based extraction makes indexing 50-100x more
   expensive than vector-only approaches.
4. **Dynamic Community Selection (v2)** makes global search practical by reducing token
   usage by 79% without sacrificing quality.
5. **GraphRAG is complementary to vector RAG**, not a replacement -- the best systems
   use both depending on query type.

---

## References

- Edge, D. et al. (2024). "From Local to Global: A GraphRAG Approach to Query-Focused
  Summarization." arXiv:2404.16130
- Microsoft GraphRAG GitHub: https://github.com/microsoft/graphrag
- GraphRAG Accelerator: https://github.com/Azure-Samples/graphrag-accelerator
- Traag, V.A. et al. (2019). "From Louvain to Leiden: guaranteeing well-connected
  communities." Scientific Reports, 9(1), 5233.
