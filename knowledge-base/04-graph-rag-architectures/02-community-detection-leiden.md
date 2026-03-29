# Community Detection with the Leiden Algorithm

## What Is Community Detection?

Community detection is the process of identifying groups of nodes in a graph that are
more densely connected to each other than to the rest of the network. In the context of
knowledge graphs and GraphRAG, these communities represent **clusters of related entities**
that share a common theme, domain, or narrative thread.

Think of it like identifying friend groups in a social network -- people within a group
interact frequently with each other but less so with people outside the group.

### Why Community Detection Matters for GraphRAG

In Microsoft's GraphRAG pipeline, community detection serves a critical architectural role:

1. **Reduces complexity:** A graph with 50,000 entities cannot be summarized directly.
   Communities break it into manageable units.
2. **Creates semantic clusters:** Entities grouped by structural connectivity tend to be
   topically coherent -- ideal for generating meaningful summaries.
3. **Enables hierarchical abstraction:** Multi-level communities allow queries to be
   answered at different levels of granularity (specific vs. thematic).
4. **Supports global search:** Community summaries are the atomic units of the map-reduce
   global search pipeline.

---

## Why Leiden? (Improvement over Louvain)

### The Louvain Algorithm (Predecessor)

The **Louvain algorithm** (Blondel et al., 2008) was the standard for community detection
for over a decade. It works by iteratively optimizing **modularity** -- a measure of how
well a network partition separates dense internal connections from sparse external ones.

However, Louvain has a critical flaw: it can produce **badly connected communities** or
even **disconnected communities**. This happens because Louvain moves individual nodes
between communities without checking whether the resulting community remains internally
connected.

### The Leiden Algorithm (Improvement)

The **Leiden algorithm** (Traag, Waltman, van Eck, 2019) fixes Louvain's connectivity
problem by adding a **refinement phase** that ensures communities are well-connected. It
also runs faster on large graphs.

| Property | Louvain | Leiden |
|---|---|---|
| Guaranteed connectivity | No | Yes |
| Speed | Fast | Faster |
| Quality (modularity) | Good | Better or equal |
| Hierarchical support | Yes | Yes |
| Stability | Can oscillate | More stable |
| Disconnected communities | Possible | Prevented |

The Leiden paper showed that Louvain could produce communities where **up to 25% of
communities were disconnected** in some benchmarks. Leiden eliminates this problem entirely.

---

## How the Leiden Algorithm Works

The Leiden algorithm operates in three phases that repeat iteratively until convergence:

### Phase 1: Local Moving

Each node is considered for moving to a neighboring community. A node moves to the
community that produces the **largest increase in modularity** (or a related quality
function). If no move improves modularity, the node stays.

```
Before:                        After Local Moving:

  A---B   E---F                  [  A---B  ]   [  E---F  ]
  |   |   |   |                  [  |   |  ]   [  |   |  ]
  C---D   G---H                  [  C---D  ]   [  G---H  ]
      \  /                           \  /
       \/                             \/
       I                          [ I ] (assigned to best community)
```

### Phase 2: Refinement

This is the key innovation of Leiden over Louvain. After the local moving phase, each
community is examined and potentially **split into well-connected subcommunities**.

The refinement phase:

1. Starts with each node in its own singleton community within the larger community
2. Merges nodes into subcommunities only if the merge maintains connectivity
3. Ensures every resulting community is internally connected

```
Louvain might produce:         Leiden refinement catches this:

  A---B       E---F             Community {A,B,C,D} is connected: OK
  |   |       |   |             Community {E,F,G,H} is connected: OK
  C---D       G---H
  [--- Community 1 ---]         If Louvain had merged A-B-E-F into one
  (disconnected!)               community, refinement would split it.
```

### Phase 3: Aggregation

Communities identified in the refinement phase are collapsed into **super-nodes**. The
edges between communities become edges between super-nodes (with aggregated weights).
This creates a new, smaller graph on which the algorithm repeats.

```
Original Graph:                 Aggregated Graph:

  [  A---B  ]---[  E---F  ]        [ C1 ]---[ C2 ]
  [  |   |  ]   [  |   |  ]           |         |
  [  C---D  ]   [  G---H  ]           |         |
       |             |              [ C3 ]---[ C4 ]
  [  I---J  ]---[  K---L  ]
  [  |   |  ]   [  |   |  ]
  [  M---N  ]   [  O---P  ]
```

### Iterative Convergence

The three phases repeat on the aggregated graph:

```
Iteration 1: Original graph (N nodes)
    -> Local Moving -> Refinement -> Aggregation
    -> Produces Level 0 communities + smaller graph

Iteration 2: Aggregated graph (fewer nodes)
    -> Local Moving -> Refinement -> Aggregation
    -> Produces Level 1 communities + even smaller graph

Iteration 3: Further aggregated graph
    -> Local Moving -> Refinement -> Aggregation
    -> Produces Level 2 communities

... continues until no further improvement
```

---

## Hierarchical Communities in GraphRAG

The iterative nature of Leiden naturally produces a **hierarchy** of communities at
different granularities. GraphRAG exploits this hierarchy:

### Level 0: Fine-Grained Communities

- Typically 3-15 entities per community
- Represent very specific topics or narratives
- Example: A community about a specific research project and its authors
- Best for: Detailed, specific queries

### Level 1: Mid-Level Communities

- Typically 15-100 entities per community
- Represent broader topics or domains
- Example: A community about all machine learning research at an institution
- Best for: Queries requiring moderate synthesis

### Level 2+: Coarse-Grained Communities

- Potentially hundreds of entities per community
- Represent high-level themes across the corpus
- Example: A community about all AI/ML research across multiple institutions
- Best for: Global sensemaking queries

### Conceptual Hierarchy Diagram

```
Level 2 (Coarse):     [=======  AI Research  ========]   [===  Policy  ===]
                       /            |            \              |
Level 1 (Mid):    [NLP Research] [CV Research] [RL Research] [AI Ethics]
                   /   |   \        |    \         |          /     \
Level 0 (Fine): [LLM] [MT] [NER] [OD] [Seg]  [MARL]    [Bias] [Safety]
                  |     |    |     |     |       |         |       |
Entities:       GPT  BLEU  CoNLL YOLO  SAM   AlphaGo   FairML  RLHF
                BERT  WMT  OntoN COCO  U-Net  MuZero   AIF360  ConstitAI
                T5    ...  spaCy  ...   ...    ...      ...     ...
```

Each level provides a different "zoom level" for answering queries. The query engine
selects the appropriate level based on the scope of the question.

---

## Resolution Parameter Tuning

The Leiden algorithm accepts a **resolution parameter** (gamma) that controls the
granularity of detected communities:

- **Low resolution (gamma < 1.0):** Fewer, larger communities. Merges more aggressively.
- **Default resolution (gamma = 1.0):** Standard modularity optimization.
- **High resolution (gamma > 1.0):** More, smaller communities. Splits more aggressively.

### Impact on GraphRAG

| Resolution | Community Count | Avg Community Size | Summary Depth | Cost |
|---|---|---|---|---|
| 0.5 | Few | Large | Broad overviews | Lower |
| 1.0 | Moderate | Medium | Balanced | Moderate |
| 2.0 | Many | Small | Detailed | Higher |

**Practical guidance:**

- For **broad sensemaking** tasks, lower resolution produces better global summaries
- For **detailed analysis** tasks, higher resolution preserves more specificity
- GraphRAG defaults to resolution = 1.0, which works well for most corpora
- Very large corpora (100K+ entities) may benefit from slightly lower resolution to
  keep community counts manageable
- Very specialized corpora may benefit from higher resolution to separate closely
  related but distinct topics

### Tuning in Practice

```python
# In GraphRAG settings (settings.yaml or equivalent)
cluster_graph:
  max_cluster_size: 10    # Maximum entities per community at Level 0
  strategy:
    type: leiden
    resolution: 1.0       # Adjust this parameter
    max_levels: 4          # Maximum hierarchy depth
```

The `max_cluster_size` parameter is also important -- if a community exceeds this size,
the algorithm forces further subdivision. This ensures that community summaries remain
focused and manageable for the LLM.

---

## How Communities Become Summarization Units

After community detection, GraphRAG generates a **structured summary** for each community
at each hierarchical level. This is the bridge between graph structure and natural language.

### Summarization Input

For each community, the LLM receives:

1. **All entities** in the community (names, types, descriptions)
2. **All relationships** between entities in the community (descriptions, weights)
3. **Representative text units** (source text that mentions the entities)

### Summarization Output

The LLM produces a structured report containing:

```
Community Report: [Title]
================================

Summary:
  A 2-3 paragraph overview of what this community represents,
  its key entities, and their relationships.

Rating: [0-10]
Rating Explanation:
  Why this community matters in the context of the dataset.

Key Findings:
  1. [Finding with supporting entity/relationship evidence]
  2. [Finding with supporting entity/relationship evidence]
  3. [Finding with supporting entity/relationship evidence]
  ...
```

### Example Community Summary

```
Community Report: CRISPR Gene Editing Research at MIT
=====================================================

Summary:
  This community centers on CRISPR-Cas9 gene editing research
  conducted at MIT, primarily led by Dr. Feng Zhang and
  collaborators at the Broad Institute. The group has made
  foundational contributions to CRISPR technology for mammalian
  cell editing and has been involved in key patent disputes
  with UC Berkeley.

Rating: 8
Rating Explanation:
  High importance due to the foundational nature of CRISPR
  technology and its broad implications for medicine and
  biotechnology.

Key Findings:
  1. Feng Zhang's lab demonstrated CRISPR-Cas9 editing in
     mammalian cells in 2013, a milestone that enabled
     therapeutic applications. [Source: text_unit_42, text_unit_87]
  2. Patent disputes between MIT/Broad Institute and UC Berkeley
     have shaped the intellectual property landscape for CRISPR.
     [Source: text_unit_156, text_unit_203]
  3. Spin-off company Editas Medicine was founded to
     commercialize the technology. [Source: text_unit_301]
```

---

## Community Detection Beyond Leiden

While Leiden is the default and recommended algorithm for GraphRAG, other community
detection methods exist:

| Algorithm | Strengths | Weaknesses | Use Case |
|---|---|---|---|
| **Leiden** | Fast, guaranteed connectivity, hierarchical | Requires resolution tuning | Default for GraphRAG |
| **Louvain** | Widely available, fast | Disconnected communities possible | Legacy systems |
| **Infomap** | Information-theoretic, good for flow networks | Less intuitive tuning | Communication networks |
| **Label Propagation** | Very fast, no resolution parameter | Non-deterministic, lower quality | Huge graphs where speed is critical |
| **Spectral Clustering** | Mathematically principled | Slow for large graphs, requires k | Small to medium graphs |
| **Girvan-Newman** | Edge betweenness based, intuitive | Very slow (O(m^2 n)) | Small graphs, pedagogical |

---

## Implementation Notes

### Libraries

- **Python:** `leidenalg` package (wraps C++ implementation)
- **igraph:** Leiden is available via `igraph` in Python, R, and C
- **NetworkX:** No native Leiden support; use `leidenalg` with conversion
- **GraphRAG:** Uses `graspologic` which wraps `leidenalg`

### Performance Characteristics

| Graph Size (nodes) | Leiden Runtime | Memory |
|---|---|---|
| 1,000 | < 1 second | < 10 MB |
| 10,000 | ~1-5 seconds | ~50 MB |
| 100,000 | ~10-30 seconds | ~500 MB |
| 1,000,000 | ~1-5 minutes | ~5 GB |

Leiden scales roughly linearly with the number of edges, making it practical for
knowledge graphs up to millions of nodes.

### Common Pitfalls

1. **Singleton communities:** If many entities end up alone, extraction quality may be
   poor (not enough relationships) or resolution may be too high.
2. **Giant communities:** If one community contains most entities, the graph may be too
   densely connected or resolution may be too low.
3. **Unstable across runs:** Leiden is slightly non-deterministic. Set a random seed for
   reproducibility.
4. **Weighted vs. unweighted:** GraphRAG uses relationship weights. Ensure weights are
   meaningful (not all 1.0) for best community structure.

---

## Key Takeaways

1. **Community detection transforms a flat graph into a hierarchical structure** suitable
   for multi-level summarization.
2. **Leiden improves on Louvain** by guaranteeing that all detected communities are
   internally well-connected.
3. **The three phases** (local moving, refinement, aggregation) repeat to produce a
   hierarchy of communities at increasing granularity.
4. **Resolution tuning** controls the trade-off between community size and count --
   adjust based on corpus characteristics and query needs.
5. **Community summaries are the core data structure** that enables GraphRAG's global
   search capability. Without community detection, there is no hierarchical
   summarization, and without hierarchical summarization, there is no global search.

---

## References

- Traag, V.A., Waltman, L., & van Eck, N.J. (2019). "From Louvain to Leiden:
  guaranteeing well-connected communities." Scientific Reports, 9(1), 5233.
- Blondel, V.D. et al. (2008). "Fast unfolding of communities in large networks."
  Journal of Statistical Mechanics, P10008.
- Edge, D. et al. (2024). "From Local to Global: A GraphRAG Approach to Query-Focused
  Summarization." arXiv:2404.16130.
- leidenalg Python package: https://github.com/vtraag/leidenalg
