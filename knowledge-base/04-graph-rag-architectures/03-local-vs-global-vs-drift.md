# Local vs. Global vs. DRIFT Search in GraphRAG

## Overview

Microsoft GraphRAG provides three distinct search modes, each optimized for a different
class of query. Choosing the right mode is critical for getting high-quality answers at
reasonable cost.

```
+-------------------------------------------------------------+
|                    GraphRAG Search Modes                     |
+-------------------------------------------------------------+
|                                                             |
|  LOCAL SEARCH         GLOBAL SEARCH         DRIFT SEARCH    |
|  "Who is X?"          "What are the         "How does X     |
|  "What did X do?"      main themes?"         relate to      |
|                       "Summarize all..."     the broader     |
|  Entity-focused       Community-focused      picture?"      |
|  Neighborhood         Map-Reduce over        Hybrid:        |
|  retrieval            all communities        Local + Global  |
|                                                             |
+-------------------------------------------------------------+
```

---

## Local Search

### How It Works

Local search is designed for **specific, entity-focused queries**. It retrieves
information from the neighborhood of relevant entities in the knowledge graph.

**Retrieval pipeline:**

1. **Entity identification:** The query is analyzed to identify relevant entities
   (via embedding similarity against entity descriptions)
2. **Neighborhood expansion:** For each matched entity, retrieve:
   - The entity's description
   - All relationships connected to the entity
   - Neighboring entities (1-hop or configurable depth)
   - Community reports for communities containing the entity
   - Source text units associated with the entity
3. **Context assembly:** All retrieved information is assembled into a context window,
   prioritized by relevance
4. **Answer generation:** A single LLM call generates the answer from the assembled context

```
Query: "What role did Dr. Smith play in the CRISPR patent dispute?"

Step 1: Identify entities -> [Dr. Smith, CRISPR, patent dispute]

Step 2: Expand neighborhoods:

         [Patent Office]
              |
    [UC Berkeley]---[Patent Dispute]---[Broad Institute]
              |           |                    |
         [Doudna]    [Dr. Smith]          [Feng Zhang]
              |           |                    |
         [Cas9 Paper] [Legal Brief]     [2013 Paper]

Step 3: Assemble context from entity descriptions,
        relationship descriptions, community reports,
        and source text units

Step 4: Generate answer
```

### Strengths

- **Fast:** Single LLM call (no map-reduce overhead)
- **Precise:** Directly targets the entities mentioned in the query
- **Cost-effective:** Minimal token usage compared to global search
- **Source-grounded:** Returns specific text units as evidence
- **Works well for factual queries:** Who, what, when, where questions

### Weaknesses

- **Misses global patterns:** Cannot synthesize across unrelated parts of the graph
- **Entity-dependent:** If the query mentions no recognizable entities, retrieval fails
- **Limited scope:** Only sees the local neighborhood, not the full picture

### Example Questions Best Suited for Local Search

| Question | Why Local Works |
|---|---|
| "Who founded OpenAI?" | Specific entity lookup |
| "What are the side effects of metformin?" | Entity-focused retrieval from drug's neighborhood |
| "Describe the relationship between Company A and Company B." | Direct relationship query |
| "What happened at the 2023 board meeting?" | Specific event entity |
| "What technologies does Project X use?" | Neighborhood of a specific project |

---

## Global Search

### How It Works

Global search is designed for **broad, thematic, or sensemaking queries** that require
synthesizing information across the entire corpus. It operates on **community summaries**
using a map-reduce pattern.

**Retrieval pipeline:**

1. **Community level selection:** Choose the appropriate level in the community hierarchy
   (Level 0 for fine-grained, Level 2+ for broad themes)
2. **Community retrieval:** In v1, all communities at the selected level are retrieved.
   In v2 (Dynamic Community Selection), only the top-k most relevant communities are
   retrieved based on embedding similarity
3. **Map phase:** For each community summary, the LLM generates a partial answer
   (or indicates the community is not relevant). Each partial answer receives a
   relevance score (0-100)
4. **Reduce phase:** Partial answers are sorted by relevance, accumulated into a context
   window up to the token budget, and a final synthesized answer is generated

```
Query: "What are the main research themes across all departments?"

Step 1: Select community level -> Level 1 (mid-level)

Step 2: Retrieve communities (Dynamic Community Selection in v2)

Step 3: Map phase
    +-------------------+  +-------------------+  +-------------------+
    | Community: NLP     |  | Community: CV      |  | Community: Ethics |
    | Summary: ...       |  | Summary: ...       |  | Summary: ...      |
    +--------+----------+  +--------+----------+  +--------+----------+
             |                       |                       |
             v                       v                       v
    [Partial Answer:        [Partial Answer:        [Partial Answer:
     "NLP research           "Computer vision        "AI ethics work
      focuses on LLMs..."     centers on..."          addresses bias..."]
     Score: 85]              Score: 72]              Score: 68]

Step 4: Reduce phase
    Sort by score -> Combine top answers -> Synthesize final response
```

### Strengths

- **Handles global queries:** The only mode that can answer "what are the main themes"
- **Comprehensive:** Considers information from across the entire corpus
- **Diverse perspectives:** Naturally surfaces multiple viewpoints from different communities
- **Hierarchical flexibility:** Can operate at different granularity levels

### Weaknesses

- **Expensive:** Multiple LLM calls (one per community in the map phase, plus the reduce)
- **Slow:** Map phase can take significant time for large community counts
- **Less precise:** Summaries are abstractions -- specific details may be lost
- **No source text:** Community summaries are pre-generated; original text is not directly
  available in the answer context

### Example Questions Best Suited for Global Search

| Question | Why Global Works |
|---|---|
| "What are the main themes in this dataset?" | Requires corpus-wide synthesis |
| "Summarize the key findings across all reports." | Aggregation across all communities |
| "What are the most controversial topics?" | Needs broad perspective |
| "How has the field evolved over the past decade?" | Thematic evolution across many entities |
| "What are the strengths and weaknesses of the organization?" | Multi-faceted sensemaking |

### Community Level Selection Guide

| Query Scope | Recommended Level | Rationale |
|---|---|---|
| Very specific themes | Level 0 | Fine-grained communities preserve detail |
| Department/team-level | Level 1 | Mid-level aggregation |
| Organization-wide | Level 2+ | Broad thematic groupings |
| "Give me the big picture" | Highest available level | Maximum abstraction |

---

## DRIFT Search (Dynamic Reasoning and Inference with Flexible Traversal)

### How It Works

DRIFT search is a **hybrid approach** introduced in GraphRAG v2 (late 2024 / early 2025).
It combines the specificity of local search with the breadth of global search by
dynamically navigating between entity neighborhoods and community summaries.

DRIFT operates in three phases:

**Phase 1: Primer**

1. Identify initial entities relevant to the query (similar to local search)
2. Retrieve community reports for communities containing those entities
3. Generate an initial "primer" response that provides a broad orientation
4. Along with the primer, generate a set of **follow-up questions** that would help
   refine the answer

**Phase 2: Follow-Up**

1. For each follow-up question, perform targeted local search
2. Retrieve additional entity neighborhoods, relationships, and text units
3. Generate partial answers to each follow-up question
4. Assess which areas need further exploration

**Phase 3: Output**

1. Synthesize the primer response and all follow-up answers
2. Generate a comprehensive final answer that combines both local detail and global context
3. Provide source citations from text units encountered during exploration

```
Query: "How does the shift to remote work affect employee well-being
        and what are companies doing about it?"

Phase 1: PRIMER
    Entities found: [remote work, employee well-being, corporate policy]
    Community reports retrieved for context
    Primer: "Remote work has become a dominant theme..."
    Follow-up questions generated:
      Q1: "What mental health impacts are reported?"
      Q2: "Which companies have implemented wellness programs?"
      Q3: "How do remote work policies vary by industry?"

Phase 2: FOLLOW-UP
    Q1 -> Local search on [mental health, burnout, isolation]
        -> Partial answer with specific findings
    Q2 -> Local search on [wellness programs, corporate initiatives]
        -> Partial answer with company examples
    Q3 -> Local search on [tech industry, finance, healthcare, remote policy]
        -> Partial answer with industry comparisons

Phase 3: OUTPUT
    Synthesize primer + all follow-up answers
    -> Comprehensive response with both breadth and depth
    -> Specific examples grounded in source text
    -> Thematic context from community reports
```

### Strengths

- **Best of both worlds:** Combines local specificity with global context
- **Self-directed exploration:** The model identifies what additional information it needs
- **Source-grounded:** Unlike pure global search, DRIFT retrieves actual text units
- **Handles complex queries:** Works well for queries that are neither purely specific
  nor purely global
- **Adaptive depth:** Follow-up questions allow the search to go as deep as needed

### Weaknesses

- **Most expensive mode:** Multiple rounds of LLM calls (primer + follow-ups + synthesis)
- **Slowest mode:** Sequential phases add latency
- **Quality depends on follow-up question generation:** Poor follow-ups lead to
  incomplete answers
- **Newest and least battle-tested:** Less community experience compared to local/global

### Example Questions Best Suited for DRIFT Search

| Question | Why DRIFT Works |
|---|---|
| "How does X relate to the broader landscape of Y?" | Needs both specific (X) and global (Y) |
| "What are the implications of event A for stakeholders B, C, and D?" | Multiple entities + thematic synthesis |
| "Compare the approaches of different teams to solving problem X." | Specific entities + cross-cutting analysis |
| "What led to the failure of Project X and what lessons can be learned?" | Narrative requiring depth and context |
| "How is technology T being adopted across different sectors?" | Specific technology + broad industry view |

---

## Comparison Matrix

| Dimension | Local Search | Global Search | DRIFT Search |
|---|---|---|---|
| **Query type** | Specific, factual | Broad, thematic | Complex, multi-faceted |
| **Retrieval source** | Entity neighborhoods + text units | Community summaries | Both |
| **LLM calls** | 1 | N (one per community) + 1 | Multiple rounds |
| **Latency** | Fast (seconds) | Moderate to slow | Slow (multiple rounds) |
| **Token cost** | Low | High (reduced with DCS) | Highest |
| **Comprehensiveness** | Low-Medium | High | High |
| **Specificity** | High | Low-Medium | High |
| **Source citations** | Yes (text units) | No (summaries only) | Yes (text units) |
| **Best for** | Fact lookup | Sensemaking | Complex analysis |

---

## Decision Flowchart

```
                        [User Query]
                             |
                             v
               Is the query about specific
               entities or facts?
              /                        \
           YES                          NO
            |                            |
            v                            v
     Is broad context              Is the query asking
     also needed?                  for themes/summaries
    /             \                across the corpus?
  YES              NO            /                  \
   |                |          YES                   NO
   v                v           |                     |
[DRIFT]         [LOCAL]         v                     v
                           [GLOBAL]              [DRIFT or LOCAL]
                                              (depends on complexity)
```

### Quick Decision Rules

1. **Use Local** when the query names specific entities and asks for specific facts.
2. **Use Global** when the query asks "what are the main..." or "summarize all..."
3. **Use DRIFT** when the query is complex, referencing specific entities but also
   asking for broader context or implications.
4. **When in doubt**, start with Local (cheapest). If the answer feels incomplete,
   escalate to DRIFT. Reserve Global for explicitly corpus-wide questions.

---

## Practical Configuration

### Local Search Settings

```yaml
local_search:
  text_unit_prop: 0.5       # Proportion of context for text units
  community_prop: 0.1       # Proportion of context for community reports
  conversation_history_max_turns: 5
  top_k_entities: 10        # Number of entities to retrieve
  top_k_relationships: 10   # Number of relationships per entity
  max_tokens: 12000         # Total context token budget
```

### Global Search Settings

```yaml
global_search:
  max_tokens: 12000         # Token budget for reduce phase
  data_max_tokens: 12000    # Token budget per map phase call
  map_max_tokens: 1000      # Max tokens per partial answer
  reduce_max_tokens: 2000   # Max tokens for final answer
  concurrency: 32           # Parallel map phase calls
  dynamic_community_selection:
    enabled: true            # v2: use DCS to reduce costs
    top_k: 20               # Number of communities to select
```

### DRIFT Search Settings

```yaml
drift_search:
  max_follow_up_questions: 5  # Number of follow-up questions per round
  max_rounds: 2               # Maximum follow-up rounds
  primer_max_tokens: 2000     # Token budget for primer
  follow_up_max_tokens: 1000  # Token budget per follow-up answer
  output_max_tokens: 3000     # Token budget for final synthesis
```

---

## Cost Comparison (Approximate)

Assuming a corpus that produces 500 communities at Level 1, with GPT-4 pricing:

| Search Mode | LLM Calls | Input Tokens | Output Tokens | Estimated Cost |
|---|---|---|---|---|
| Local | 1 | ~10,000 | ~500 | ~$0.10 |
| Global (v1) | 501 | ~500,000 | ~25,000 | ~$5.00 |
| Global (v2 DCS) | 21 | ~25,000 | ~2,000 | ~$0.30 |
| DRIFT | ~8-12 | ~80,000 | ~5,000 | ~$0.80 |

Dynamic Community Selection (v2) dramatically reduces the cost of global search, making
it comparable to DRIFT for many workloads.

---

## Key Takeaways

1. **Local search** is your default -- fast, cheap, and precise for entity-focused queries.
2. **Global search** is the only option for corpus-wide sensemaking, but use v2's Dynamic
   Community Selection to manage costs.
3. **DRIFT search** bridges the gap for complex queries that need both specificity and
   breadth, at the cost of higher latency and token usage.
4. **The choice of search mode has a bigger impact on answer quality than most parameter
   tuning** -- picking the wrong mode for a query type will produce poor results
   regardless of other settings.
5. **A production system should support all three modes** and either route queries
   automatically (via a classifier) or let users select the appropriate mode.

---

## References

- Edge, D. et al. (2024). "From Local to Global: A GraphRAG Approach to Query-Focused
  Summarization." arXiv:2404.16130
- Microsoft GraphRAG documentation: https://microsoft.github.io/graphrag/
- DRIFT search announcement: Microsoft Research Blog, 2024
- Dynamic Community Selection: GraphRAG v0.5.0 release notes, January 2025
