import { motion } from 'framer-motion'
import { useState } from 'react'
import AnimatedCard from '../components/AnimatedCard'
import CodeBlock from '../components/CodeBlock'
import { Zap, Database, GitBranch, Brain, Layers, Search, Clock, BarChart3 } from 'lucide-react'

const techniques = [
  {
    id: 'lightrag',
    name: 'LightRAG',
    tagline: 'Graph RAG without the cost',
    color: '#F39C12',
    icon: Zap,
    paper: 'arxiv.org/abs/2410.05779 (EMNLP 2025)',
    github: 'github.com/HKUDS/LightRAG (30K+ stars)',
    what: 'LightRAG builds a knowledge graph from your documents (just like GraphRAG) but skips the expensive community detection step. Instead, it uses smart key-value indexing on entity and relationship descriptions. The result: 100x cheaper indexing, instant incremental updates, and 70-90% of GraphRAG\'s quality.',
    how: [
      'Chunk the document (default 1200 tokens)',
      'LLM extracts entities + relationships from each chunk',
      'Profile each entity/relation with key-value descriptions',
      'Deduplicate — merge identical entities across chunks',
      'Embed all descriptions into vectors for retrieval',
    ],
    modes: [
      { name: 'Naive', desc: 'Standard chunk retrieval (baseline)', best: 'Simple fact lookup' },
      { name: 'Local', desc: 'Search entity nodes + immediate neighbors', best: '"What does X do?"' },
      { name: 'Global', desc: 'Search relation-level themes across entities', best: '"What are the main themes?"' },
      { name: 'Hybrid', desc: 'Both local + global combined', best: 'Best overall quality' },
    ],
    why_different: 'GraphRAG uses Leiden community detection to cluster entities, then generates LLM-written summary reports for each community. This costs ~14 million tokens for a large corpus. LightRAG replaces this with profiling functions that enhance relation keys with global themes from connected entities — achieving similar global understanding at 1/100th the cost.',
    incremental: 'When you add new documents, LightRAG just extracts new entities and merges them into the existing graph. No rebuild. GraphRAG would need to tear down and reconstruct all community reports from scratch.',
    cost: '$0.50 for 500 pages',
    quality: '70-90% of GraphRAG',
    code: `from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

rag = LightRAG(
    working_dir="./storage",
    llm_model_func=gpt_4o_mini_complete,
    embedding_func=openai_embed,
)
await rag.initialize_storages()
await rag.ainsert("Your document text...")

# Query in hybrid mode (local + global)
answer = await rag.aquery(
    "What are the key findings?",
    param=QueryParam(mode="hybrid")
)`,
  },
  {
    id: 'graphrag',
    name: 'Microsoft GraphRAG',
    tagline: 'The gold standard for global queries',
    color: '#6C5CE7',
    icon: Brain,
    paper: 'arxiv.org/abs/2404.16130 (Microsoft Research, 2024)',
    github: 'github.com/microsoft/graphrag',
    what: 'Microsoft\'s GraphRAG builds a knowledge graph, applies Leiden community detection to cluster related entities, generates LLM-written summary reports at each hierarchy level, and uses map-reduce over these summaries to answer global questions. It\'s the highest-quality approach for corpus-wide reasoning but also the most expensive.',
    how: [
      'Extract entities + relationships from every text chunk using LLM',
      'Build a graph and apply Leiden community detection',
      'Generate LLM summary reports for each community at multiple hierarchy levels',
      'Local search: retrieve entity neighborhoods + community reports',
      'Global search: map-reduce over all community summaries',
    ],
    modes: [
      { name: 'Local', desc: 'Entity-focused retrieval with community context', best: '"Tell me about entity X"' },
      { name: 'Global', desc: 'Map-reduce over community summaries', best: '"What are the themes across all documents?"' },
      { name: 'DRIFT', desc: 'Starts global, drills down locally', best: 'Complex questions needing both breadth and depth' },
    ],
    why_different: 'GraphRAG is the only approach that can reliably answer "What are the main themes across 10,000 documents?" because it pre-computes hierarchical summaries. No other approach has this global understanding — but it comes at a steep cost.',
    incremental: 'Adding new documents requires rebuilding community structures and regenerating all summary reports. This is the biggest drawback for production systems with frequently updated data.',
    cost: '$33K+ for large corpora',
    quality: '100% (the benchmark)',
    code: `from graphrag.index import create_pipeline_config
from graphrag.query.structured_search.local_search import LocalSearch

# GraphRAG requires a full indexing pipeline
# Run: python -m graphrag.index --root ./ragtest
# Then query:
search = LocalSearch(...)
result = await search.asearch("Your question")`,
  },
  {
    id: 'hybrid',
    name: 'Hybrid RAG',
    tagline: 'The production answer',
    color: '#4ECDC4',
    icon: Layers,
    paper: 'arxiv.org/abs/2408.04948 (HybridRAG, 2024)',
    github: 'No single library — it\'s an architecture pattern',
    what: 'Hybrid RAG combines vector search and graph traversal. The graph tells you WHERE to look (which entities and relationships are relevant), then vector search retrieves the original text chunks for those entities. The LLM gets both structured relationships AND exact text with numbers and details. This is the production default in 2026.',
    how: [
      'Build both: a knowledge graph (Neo4j) AND a vector index (ChromaDB)',
      'User asks a question',
      'Graph traversal finds relevant entities + their relationships',
      'Vector search retrieves original text chunks mentioning those entities',
      'LLM answers with BOTH: graph structure + exact text details',
    ],
    modes: [
      { name: 'Graph-first', desc: 'Graph finds entities, vector retrieves their text', best: 'Relationship + detail questions' },
      { name: 'Vector-first', desc: 'Vector finds chunks, graph expands with connections', best: 'Fact + context questions' },
      { name: 'Parallel', desc: 'Both run simultaneously, results fused with RRF', best: 'Best overall but higher latency' },
    ],
    why_different: 'Classic RAG keeps text but loses connections. Graph RAG finds connections but loses exact details (numbers get dropped during extraction). Hybrid RAG combines both — the graph provides the structure, the text provides the substance. In our tests, Hybrid RAG was the only approach that could both trace multi-hop paths AND cite exact dollar amounts.',
    incremental: 'Adding new documents: extract entities and MERGE into graph (no rebuild), add new chunks to vector store. Both operations are incremental.',
    cost: 'Moderate (extraction cost + vector indexing)',
    quality: 'Highest overall (graph structure + text detail)',
    code: `# Step 1: Graph finds relevant entities
entities = ask_llm("Extract key entities from: " + question)
graph_triples = neo4j.run(
    "MATCH (a)-[r]->(b) WHERE a.name CONTAINS $term...",
    term=entities[0]
)

# Step 2: Vector search retrieves text for those entities
for entity in discovered_entities:
    chunks = chromadb.query(entity_name, n_results=1)

# Step 3: LLM answers with BOTH
answer = ask_llm(f"""
Graph relationships: {graph_triples}
Original text: {chunks}
Question: {question}
""")`,
  },
  {
    id: 'community',
    name: 'Community Detection',
    tagline: 'Finding clusters in knowledge graphs',
    color: '#FF6B6B',
    icon: GitBranch,
    paper: 'Leiden algorithm (Traag et al., 2019)',
    github: 'Used internally by GraphRAG',
    what: 'Community detection groups densely connected entities into clusters. In a knowledge graph about a company, you might get communities like "Leadership Team", "Products", "Investors", "Competitors". GraphRAG generates LLM summaries for each community, enabling global questions like "What are the key themes?" LightRAG skips this entirely.',
    how: [
      'Build the entity-relationship graph',
      'Apply the Leiden algorithm (improvement over Louvain)',
      'Entities cluster into communities at multiple hierarchy levels',
      'Level 0: fine-grained (2-10 entities) — "The CTO team"',
      'Level 1+: broader groupings — "Engineering division"',
      'Generate LLM summary for each community',
    ],
    modes: [
      { name: 'Level 0', desc: 'Fine-grained communities (2-10 entities)', best: 'Detailed topic clusters' },
      { name: 'Level 1', desc: 'Broader groupings', best: 'Department-level themes' },
      { name: 'Level 2+', desc: 'High-level themes', best: 'Corpus-wide summarization' },
    ],
    why_different: 'Community detection is what makes GraphRAG\'s global search possible. Without it (like in LightRAG), you can still do entity-level and relationship-level queries, but corpus-wide thematic summarization is weaker. The trade-off: community detection + summary generation is the most expensive step in the entire pipeline.',
    incremental: 'Any new entity potentially changes community boundaries, requiring full recomputation. This is why LightRAG chose to skip it.',
    cost: 'High (Leiden algorithm + LLM summaries per community)',
    quality: 'Essential for global/thematic queries',
    code: `import networkx as nx
from community import community_louvain  # or leidenalg

# Build graph from extracted entities
G = nx.Graph()
G.add_edges_from([(r.source, r.target) for r in relationships])

# Detect communities
partition = community_louvain.best_partition(G, resolution=1.0)

# Group entities by community
communities = {}
for entity, comm_id in partition.items():
    communities.setdefault(comm_id, []).append(entity)

# Generate summary for each community
for comm_id, members in communities.items():
    summary = ask_llm(f"Summarize this group: {members}")`,
  },
  {
    id: 'embeddings',
    name: 'KG Embeddings',
    tagline: 'Predicting missing links',
    color: '#DDA0DD',
    icon: BarChart3,
    paper: 'TransE (Bordes et al., 2013), RotatE (Sun et al., 2019)',
    github: 'github.com/pykeen/pykeen',
    what: 'Knowledge graph embeddings map entities and relations into continuous vector spaces. The key idea: if (Einstein, developed, Relativity) is true, then the embedding of "Einstein" + the embedding of "developed" should be close to the embedding of "Relativity". This enables link prediction — discovering relationships that should exist but aren\'t in the graph yet.',
    how: [
      'Define a scoring function: TransE uses h + r ≈ t (translation)',
      'RotatE uses rotation in complex space (handles more relation patterns)',
      'Train on known triples, negative sampling for unknown triples',
      'Evaluate with MRR (Mean Reciprocal Rank) and Hits@K',
      'Use trained embeddings for link prediction: given (h, r, ?), predict t',
    ],
    modes: [
      { name: 'TransE', desc: 'Relations as translations: h + r ≈ t', best: 'Simple, fast, good baseline' },
      { name: 'RotatE', desc: 'Relations as rotations in complex space', best: 'Handles symmetric/antisymmetric relations' },
      { name: 'ComplEx', desc: 'Complex-valued embeddings', best: 'Asymmetric relations' },
      { name: 'DistMult', desc: 'Bilinear diagonal model', best: 'Symmetric relations, simple' },
    ],
    why_different: 'KG embeddings are about completing the graph, not answering questions. They predict MISSING edges — "Drug X probably treats Disease Y because similar drugs treat similar diseases." This is different from Graph RAG which retrieves existing knowledge. Embeddings are used in drug discovery, recommendation systems, and knowledge base completion.',
    incremental: 'Adding new entities requires retraining or fine-tuning the embedding model.',
    cost: 'Low (training is compute, not LLM calls)',
    quality: 'Depends on task — excellent for link prediction',
    code: `from pykeen.pipeline import pipeline

# Train TransE on FB15k-237 benchmark
result = pipeline(
    dataset="FB15k-237",
    model="TransE",
    training_kwargs=dict(num_epochs=100),
    evaluation_kwargs=dict(batch_size=256),
)

# Predict missing links
predictions = result.model.predict_tails(
    head="Einstein", relation="developed"
)`,
  },
  {
    id: 'temporal',
    name: 'Temporal Knowledge Graphs',
    tagline: 'Facts that change over time',
    color: '#98FB98',
    icon: Clock,
    paper: 'Graphiti (Zep, 2025) — arxiv.org/abs/2501.13956',
    github: 'github.com/getzep/graphiti (24K+ stars)',
    what: 'Standard knowledge graphs assume facts are permanent. But "CEO of OpenAI" was Sam Altman, then briefly Mira Murati, then Sam Altman again. Temporal KGs attach time validity to every fact: valid_from, valid_to, source, confidence. This enables point-in-time queries ("Who was CEO in November 2023?") and fact invalidation when new information contradicts old facts.',
    how: [
      'Every fact gets temporal metadata: valid_from, valid_to, source, confidence',
      'New facts can invalidate old ones (CEO changed → old CEO fact gets valid_to timestamp)',
      'Point-in-time queries: "What was true at time T?"',
      'Graphiti: hybrid retrieval (semantic + BM25 + graph traversal) with NO LLM at read time',
      'P95 retrieval latency: 300ms',
    ],
    modes: [
      { name: 'Current', desc: 'Only facts valid right now', best: 'Default queries' },
      { name: 'Point-in-time', desc: 'Facts valid at a specific date', best: '"Who was CEO in 2023?"' },
      { name: 'Timeline', desc: 'How a fact evolved over time', best: '"Show the CEO history"' },
    ],
    why_different: 'Most KG approaches treat facts as immutable. In reality, facts change constantly — people change jobs, companies get acquired, products get discontinued. Temporal KGs are essential for AI agent memory (the agent needs to know what\'s current, not what was true 6 months ago) and for any domain where information evolves.',
    incremental: 'By design — new facts are always added incrementally. Old facts are never deleted, just timestamped as no longer valid.',
    cost: 'Low (metadata overhead only)',
    quality: 'Essential for time-sensitive domains',
    code: `# Simplified temporal KG pattern
def add_fact(neo4j, subject, predicate, object, source):
    neo4j.run("""
        MERGE (s:Entity {name: $subj})
        MERGE (o:Entity {name: $obj})
        CREATE (s)-[:FACT {
            type: $pred,
            valid_from: datetime(),
            valid_to: null,
            source: $source,
            confidence: 0.95
        }]->(o)
    """, subj=subject, pred=predicate, obj=object, source=source)

def invalidate_fact(neo4j, subject, predicate, object):
    neo4j.run("""
        MATCH (s {name: $subj})-[r:FACT {type: $pred}]->(o {name: $obj})
        WHERE r.valid_to IS NULL
        SET r.valid_to = datetime()
    """, subj=subject, pred=predicate, obj=object)`,
  },
]

export default function Techniques() {
  const [expanded, setExpanded] = useState(null)

  return (
    <div className="max-w-5xl mx-auto px-6 py-16">
      <motion.h1
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-4xl md:text-5xl font-bold mb-4 text-center"
      >
        <span className="bg-gradient-to-r from-[#F39C12] to-[#FF6B6B] bg-clip-text text-transparent">
          Graph RAG Techniques
        </span>
      </motion.h1>
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
        className="text-gray-400 text-center mb-16 max-w-2xl mx-auto"
      >
        Every major approach to combining knowledge graphs with retrieval-augmented generation —
        how each works, when to use it, and real code examples.
      </motion.p>

      <div className="space-y-6">
        {techniques.map((t, i) => {
          const Icon = t.icon
          const isOpen = expanded === t.id
          return (
            <motion.div
              key={t.id}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: '-30px' }}
              transition={{ delay: i * 0.05 }}
            >
              <div
                className="rounded-xl border overflow-hidden"
                style={{ borderColor: t.color + '30', backgroundColor: t.color + '05' }}
              >
                {/* Header — always visible */}
                <button
                  onClick={() => setExpanded(isOpen ? null : t.id)}
                  className="w-full p-6 flex items-start gap-4 text-left cursor-pointer hover:bg-white/[0.02] transition-colors"
                >
                  <div className="w-12 h-12 rounded-xl flex items-center justify-center shrink-0"
                    style={{ backgroundColor: t.color + '20' }}>
                    <Icon size={22} style={{ color: t.color }} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-3 mb-1">
                      <h2 className="text-xl font-bold text-white">{t.name}</h2>
                      <span className="text-xs px-2 py-0.5 rounded-full border"
                        style={{ color: t.color, borderColor: t.color + '40' }}>
                        {t.tagline}
                      </span>
                    </div>
                    <p className="text-sm text-gray-400 line-clamp-2">{t.what}</p>
                    <div className="flex items-center gap-4 mt-2 text-xs text-gray-600">
                      <span>Cost: {t.cost}</span>
                      <span>Quality: {t.quality}</span>
                    </div>
                  </div>
                  <motion.span
                    animate={{ rotate: isOpen ? 180 : 0 }}
                    className="text-gray-500 text-xl shrink-0 mt-1"
                  >
                    ▾
                  </motion.span>
                </button>

                {/* Expanded content */}
                {isOpen && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    transition={{ duration: 0.3 }}
                    className="border-t px-6 pb-6"
                    style={{ borderColor: t.color + '15' }}
                  >
                    {/* References */}
                    <div className="flex flex-wrap gap-4 py-4 text-xs text-gray-500">
                      <span>Paper: {t.paper}</span>
                      <span>Code: {t.github}</span>
                    </div>

                    {/* How it works */}
                    <div className="mb-6">
                      <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider mb-3">How It Works</h3>
                      <div className="flex flex-wrap gap-2">
                        {t.how.map((step, j) => (
                          <div key={j} className="flex items-center gap-2">
                            <span className="w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold"
                              style={{ backgroundColor: t.color + '20', color: t.color }}>
                              {j + 1}
                            </span>
                            <span className="text-sm text-gray-300">{step}</span>
                            {j < t.how.length - 1 && <span className="text-gray-700 mx-1">→</span>}
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Modes */}
                    <div className="mb-6">
                      <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider mb-3">Query Modes</h3>
                      <div className="grid md:grid-cols-2 gap-3">
                        {t.modes.map((mode) => (
                          <div key={mode.name} className="p-3 rounded-lg bg-white/5 border border-white/10">
                            <h4 className="text-sm font-bold text-white mb-1">{mode.name}</h4>
                            <p className="text-xs text-gray-400 mb-1">{mode.desc}</p>
                            <p className="text-xs" style={{ color: t.color }}>Best for: {mode.best}</p>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* What makes it different */}
                    <div className="mb-6">
                      <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider mb-3">What Makes It Different</h3>
                      <p className="text-sm text-gray-300 leading-relaxed">{t.why_different}</p>
                    </div>

                    {/* Incremental updates */}
                    <div className="mb-6">
                      <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider mb-3">Incremental Updates</h3>
                      <p className="text-sm text-gray-300">{t.incremental}</p>
                    </div>

                    {/* Code */}
                    <div>
                      <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider mb-3">Code Example</h3>
                      <CodeBlock language="python">{t.code}</CodeBlock>
                    </div>
                  </motion.div>
                )}
              </div>
            </motion.div>
          )
        })}
      </div>

      {/* Quick comparison table */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="mt-16"
      >
        <h2 className="text-2xl font-bold text-white mb-6 text-center">At a Glance</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/10">
                <th className="text-left py-3 px-4 text-gray-400 font-medium">Approach</th>
                <th className="text-left py-3 px-4 text-gray-400 font-medium">Cost</th>
                <th className="text-left py-3 px-4 text-gray-400 font-medium">Quality</th>
                <th className="text-left py-3 px-4 text-gray-400 font-medium">Incremental</th>
                <th className="text-left py-3 px-4 text-gray-400 font-medium">Best For</th>
              </tr>
            </thead>
            <tbody>
              {[
                ['Classic RAG', 'Lowest', 'Good for facts', 'Yes', 'Simple fact lookup'],
                ['LightRAG', 'Low ($0.50)', '70-90%', 'Yes (seamless)', 'Cost-effective Graph RAG'],
                ['GraphRAG', 'Very High ($33K+)', 'Highest', 'No (full rebuild)', 'Global corpus-wide queries'],
                ['Hybrid RAG', 'Moderate', 'Best overall', 'Yes', 'Production systems'],
                ['KG Embeddings', 'Low (compute)', 'Task-specific', 'Retrain needed', 'Link prediction, drug discovery'],
                ['Temporal KG', 'Low (metadata)', 'Time-aware', 'By design', 'Agent memory, evolving data'],
              ].map(([name, cost, quality, inc, best], i) => (
                <tr key={name} className="border-b border-white/5 hover:bg-white/[0.02]">
                  <td className="py-3 px-4 text-white font-medium">{name}</td>
                  <td className="py-3 px-4 text-gray-400">{cost}</td>
                  <td className="py-3 px-4 text-gray-400">{quality}</td>
                  <td className="py-3 px-4 text-gray-400">{inc}</td>
                  <td className="py-3 px-4 text-gray-400">{best}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.div>
    </div>
  )
}
