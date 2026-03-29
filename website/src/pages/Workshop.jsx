import { motion } from 'framer-motion'
import AnimatedCard from '../components/AnimatedCard'
import { BookOpen, Code, Zap, CheckCircle, Terminal, Database, GitBranch, Brain } from 'lucide-react'

const modules = [
  {
    title: 'Foundations & Cypher',
    color: '#4ECDC4',
    icon: BookOpen,
    topics: [
      'What are Knowledge Graphs — nodes, edges, triples',
      'Property graphs vs RDF — when to use which',
      'Neo4j setup with Docker',
      'Cypher queries — CREATE, MATCH, RETURN, DELETE',
      'Multi-hop traversal — the power of graphs',
      'Why traditional RAG fails on relationship questions',
    ],
    hands_on: 'Write Cypher queries in Neo4j Browser — create nodes, connect them, traverse paths',
  },
  {
    title: 'Building Knowledge Graphs from Text',
    color: '#6C5CE7',
    icon: Code,
    topics: [
      'LLM-based entity extraction with structured output',
      'Relationship extraction with Pydantic schemas',
      'Loading extracted data into Neo4j',
      'Graph visualization with pyvis',
      'Chunking large documents for batch extraction',
      'Entity resolution — merging duplicates across chunks',
    ],
    hands_on: 'Build a complete KG from a research paper — extract, load, visualize, query',
  },
  {
    title: 'Graph RAG & Hybrid RAG',
    color: '#F39C12',
    icon: Zap,
    topics: [
      'Classic RAG vs Graph RAG — live side-by-side comparison',
      'Text-to-Cypher — LLM generates graph queries dynamically',
      'Hybrid RAG — graph finds WHERE to look, text gives DETAILS',
      'LightRAG — automated Graph RAG in 5 lines of code',
      'When Graph RAG wins: multi-hop, aggregation, path tracing',
      'When Classic RAG wins: simple facts, exact numbers',
    ],
    hands_on: 'Run all three approaches on the same document, compare answers, see the difference',
  },
  {
    title: 'Structured Data & Real-World Patterns',
    color: '#FF6B6B',
    icon: Database,
    topics: [
      'CSV → Knowledge Graph (no LLM needed for structured data)',
      'SQLite → Knowledge Graph',
      'Incremental updates — adding new documents without rebuilding',
      'Schema design for different domains',
      'Production patterns: cost, latency, quality tradeoffs',
    ],
    hands_on: 'Transform a CSV dataset into a connected knowledge graph, run Graph RAG over it',
  },
]

const notebooks = [
  { name: 'company_kg_workshop.ipynb', desc: 'Build KG from text — people, companies, hobbies, careers', color: '#6C5CE7' },
  { name: 'kg_vs_rag_showdown.ipynb', desc: 'Classic RAG vs Graph RAG — honest practical comparison', color: '#F39C12' },
  { name: 'hybrid_rag.ipynb', desc: 'Hybrid RAG — the production answer combining both approaches', color: '#4ECDC4' },
  { name: 'lightrag_explained.ipynb', desc: 'LightRAG — automated Graph RAG with working example', color: '#FF6B6B' },
  { name: 'csv_sqlite_to_kg.ipynb', desc: 'Structured data → Knowledge Graph + Graph RAG queries', color: '#6C5CE7' },
  { name: 'large_document_chunking.ipynb', desc: 'Chunking big documents, batch extraction, entity merging', color: '#F39C12' },
]

const prerequisites = [
  { text: 'Python 3.11+ installed', icon: Code },
  { text: 'Docker (for Neo4j)', icon: Database },
  { text: 'OpenAI API key', icon: Brain },
  { text: 'VS Code with Jupyter extension', icon: Terminal },
  { text: 'Basic Python familiarity', icon: GitBranch },
]

export default function Workshop() {
  return (
    <div className="max-w-5xl mx-auto px-6 py-16">
      <motion.h1
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-4xl md:text-5xl font-bold mb-4 text-center"
      >
        <span className="bg-gradient-to-r from-[#4ECDC4] to-[#6C5CE7] bg-clip-text text-transparent">
          Knowledge Graph Workshop
        </span>
      </motion.h1>
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="text-gray-400 text-center mb-16 max-w-2xl mx-auto"
      >
        Hands-on workshop covering Knowledge Graphs, Graph RAG, and Hybrid RAG.
        Build real systems from scratch with Jupyter notebooks.
      </motion.p>

      {/* Modules */}
      <section className="mb-20">
        <div className="space-y-8">
          {modules.map((m, i) => {
            const Icon = m.icon
            return (
              <motion.div
                key={m.title}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: '-50px' }}
                transition={{ duration: 0.5, delay: i * 0.1 }}
              >
                <div
                  className="rounded-xl border p-6 md:p-8"
                  style={{
                    borderColor: m.color + '30',
                    backgroundColor: m.color + '08',
                  }}
                >
                  <div className="flex items-center gap-3 mb-5">
                    <div className="w-10 h-10 rounded-lg flex items-center justify-center"
                      style={{ backgroundColor: m.color + '20' }}>
                      <Icon size={20} style={{ color: m.color }} />
                    </div>
                    <h3 className="text-xl font-bold text-white">{m.title}</h3>
                  </div>

                  <div className="grid md:grid-cols-2 gap-6">
                    <div>
                      <h4 className="text-xs font-medium text-gray-500 mb-3 uppercase tracking-wider">Topics Covered</h4>
                      <ul className="space-y-2">
                        {m.topics.map((topic) => (
                          <li key={topic} className="text-sm text-gray-300 flex items-start gap-2">
                            <CheckCircle size={14} className="mt-0.5 shrink-0" style={{ color: m.color }} />
                            {topic}
                          </li>
                        ))}
                      </ul>
                    </div>
                    <div>
                      <h4 className="text-xs font-medium text-gray-500 mb-3 uppercase tracking-wider">Hands-On</h4>
                      <div className="p-4 rounded-lg bg-white/5 border border-white/10">
                        <p className="text-sm text-gray-300 leading-relaxed">{m.hands_on}</p>
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            )
          })}
        </div>
      </section>

      {/* Notebooks */}
      <section className="mb-20">
        <motion.h2
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="text-2xl font-bold text-white mb-8 text-center"
        >
          Jupyter Notebooks
        </motion.h2>
        <div className="grid md:grid-cols-2 gap-4">
          {notebooks.map((nb, i) => (
            <AnimatedCard key={nb.name} delay={i * 0.08}>
              <div className="flex items-start gap-3">
                <Terminal size={18} style={{ color: nb.color }} className="mt-0.5 shrink-0" />
                <div>
                  <h4 className="text-sm font-mono mb-1" style={{ color: nb.color }}>{nb.name}</h4>
                  <p className="text-sm text-gray-400">{nb.desc}</p>
                </div>
              </div>
            </AnimatedCard>
          ))}
        </div>
      </section>

      {/* Prerequisites */}
      <section className="mb-20">
        <motion.h2
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="text-2xl font-bold text-white mb-8 text-center"
        >
          Prerequisites
        </motion.h2>
        <div className="max-w-md mx-auto">
          <AnimatedCard>
            <ul className="space-y-3">
              {prerequisites.map((p) => {
                const Icon = p.icon
                return (
                  <li key={p.text} className="flex items-center gap-3 text-sm text-gray-300">
                    <Icon size={16} className="text-[#4ECDC4]" />
                    {p.text}
                  </li>
                )
              })}
            </ul>
            <div className="mt-6 p-3 rounded-lg bg-[#4ECDC4]/10 border border-[#4ECDC4]/20">
              <p className="text-xs text-[#4ECDC4]">
                Quick setup: <code className="bg-white/10 px-1.5 py-0.5 rounded">cd knowledgegraphs && uv sync && cd workshop && docker compose up -d</code>
              </p>
            </div>
          </AnimatedCard>
        </div>
      </section>

      {/* Quick Start */}
      <section>
        <motion.h2
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="text-2xl font-bold text-white mb-8 text-center"
        >
          Getting Started
        </motion.h2>
        <AnimatedCard>
          <div className="font-mono text-sm space-y-2">
            <p className="text-gray-500"># Clone and setup</p>
            <p className="text-[#4ECDC4]">git clone https://github.com/your-repo/knowledgegraphs</p>
            <p className="text-[#4ECDC4]">cd knowledgegraphs</p>
            <p className="text-[#4ECDC4]">cp .env.example .env <span className="text-gray-500"># add your OpenAI key</span></p>
            <p className="text-[#4ECDC4]">uv sync</p>
            <p></p>
            <p className="text-gray-500"># Start Neo4j</p>
            <p className="text-[#4ECDC4]">cd workshop && docker compose up -d</p>
            <p></p>
            <p className="text-gray-500"># Open notebooks in VS Code</p>
            <p className="text-[#4ECDC4]">code workshop/company_kg_workshop.ipynb</p>
          </div>
        </AnimatedCard>
      </section>
    </div>
  )
}
