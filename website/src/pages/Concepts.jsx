import { motion, useInView } from 'framer-motion'
import { useRef } from 'react'
import AnimatedCard from '../components/AnimatedCard'
import { Database, GitBranch, Zap, Brain, ArrowRight } from 'lucide-react'

function Section({ children, className = '' }) {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true, margin: '-60px' })
  return (
    <motion.section
      ref={ref}
      initial={{ opacity: 0, y: 40 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.7 }}
      className={`mb-20 ${className}`}
    >
      {children}
    </motion.section>
  )
}

const ragComparison = [
  {
    title: 'Traditional RAG',
    color: '#F39C12',
    items: [
      'Chunks documents into flat text segments',
      'Embeds chunks into a vector store',
      'Retrieves top-k similar chunks via cosine similarity',
      'Passes retrieved chunks to LLM as context',
      'Works well for simple, single-hop factual questions',
    ],
  },
  {
    title: 'Graph RAG',
    color: '#4ECDC4',
    items: [
      'Extracts entities and relationships from documents',
      'Builds a knowledge graph of interconnected concepts',
      'Traverses graph paths to gather multi-hop context',
      'Leverages community detection for global summarization',
      'Excels at complex, relationship-dependent queries',
    ],
  },
]

const pipelineSteps = [
  { label: 'Document', icon: '📄', color: '#6C5CE7' },
  { label: 'Extract Entities', icon: '🔍', color: '#4ECDC4' },
  { label: 'Build Graph', icon: '🕸️', color: '#F39C12' },
  { label: 'Query', icon: '❓', color: '#FF6B6B' },
  { label: 'Answer', icon: '✅', color: '#4ECDC4' },
]

const decisionMatrix = [
  { scenario: 'Simple factual Q&A', best: 'Traditional RAG', reason: 'Direct chunk retrieval is sufficient and cheaper' },
  { scenario: 'Multi-hop reasoning', best: 'Graph RAG', reason: 'Graph traversal connects related entities across documents' },
  { scenario: 'Aggregation queries', best: 'Graph RAG', reason: 'Community summaries provide holistic answers' },
  { scenario: 'Large document corpus', best: 'Hybrid RAG', reason: 'Combines vector speed with graph depth' },
  { scenario: 'Real-time applications', best: 'Traditional RAG', reason: 'Lower latency, no graph construction overhead' },
  { scenario: 'Domain-specific knowledge', best: 'Graph RAG', reason: 'Ontology-driven graphs capture domain structure' },
]

export default function Concepts() {
  return (
    <div className="max-w-5xl mx-auto px-6 py-16">
      <motion.h1
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="text-4xl md:text-5xl font-bold mb-4 text-center"
      >
        <span className="bg-gradient-to-r from-[#4ECDC4] to-[#6C5CE7] bg-clip-text text-transparent">
          Core Concepts
        </span>
      </motion.h1>
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="text-gray-400 text-center mb-16 max-w-2xl mx-auto"
      >
        Understanding the foundations of knowledge graphs and how they revolutionize
        information retrieval through structured, relationship-aware data.
      </motion.p>

      {/* What is a Knowledge Graph */}
      <Section>
        <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
          <Database size={24} className="text-[#4ECDC4]" />
          What is a Knowledge Graph?
        </h2>
        <div className="grid md:grid-cols-2 gap-8 items-center">
          <div className="text-gray-300 space-y-4">
            <p>
              A knowledge graph is a structured representation of real-world entities and
              the relationships between them. Unlike flat tables or document stores, knowledge
              graphs capture the <span className="text-[#4ECDC4] font-medium">semantic connections</span> that
              give data meaning.
            </p>
            <p>
              At its core, a knowledge graph consists of <span className="text-[#6C5CE7] font-medium">nodes</span> (entities
              like people, places, concepts) and <span className="text-[#F39C12] font-medium">edges</span> (relationships
              like "works at", "located in", "is a type of").
            </p>
            <p>
              This structure enables powerful reasoning: you can traverse paths to discover
              implicit connections, answer complex questions that span multiple hops, and
              understand context that flat text retrieval misses entirely.
            </p>
          </div>
          {/* CSS-drawn graph diagram */}
          <div className="relative h-64 md:h-80">
            <div className="absolute inset-0 flex items-center justify-center">
              {/* Edges */}
              <svg className="absolute inset-0 w-full h-full" viewBox="0 0 300 240">
                <line x1="150" y1="40" x2="60" y2="140" stroke="#6C5CE7" strokeWidth="2" opacity="0.6" />
                <line x1="150" y1="40" x2="240" y2="140" stroke="#4ECDC4" strokeWidth="2" opacity="0.6" />
                <line x1="60" y1="140" x2="150" y2="210" stroke="#F39C12" strokeWidth="2" opacity="0.6" />
                <line x1="240" y1="140" x2="150" y2="210" stroke="#FF6B6B" strokeWidth="2" opacity="0.6" />
                <line x1="60" y1="140" x2="240" y2="140" stroke="#4ECDC4" strokeWidth="2" opacity="0.3" />
              </svg>
              {/* Nodes */}
              {[
                { x: '50%', y: '10%', label: 'Person', color: '#6C5CE7' },
                { x: '15%', y: '55%', label: 'Company', color: '#4ECDC4' },
                { x: '85%', y: '55%', label: 'City', color: '#F39C12' },
                { x: '50%', y: '85%', label: 'Country', color: '#FF6B6B' },
              ].map((node) => (
                <div
                  key={node.label}
                  className="absolute flex flex-col items-center -translate-x-1/2 -translate-y-1/2"
                  style={{ left: node.x, top: node.y }}
                >
                  <div
                    className="w-12 h-12 rounded-full flex items-center justify-center text-xs font-bold text-white border-2"
                    style={{ backgroundColor: node.color + '30', borderColor: node.color }}
                  >
                    {node.label[0]}
                  </div>
                  <span className="text-xs text-gray-400 mt-1">{node.label}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </Section>

      {/* Traditional RAG vs Graph RAG */}
      <Section>
        <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
          <GitBranch size={24} className="text-[#6C5CE7]" />
          Traditional RAG vs Graph RAG
        </h2>
        <div className="grid md:grid-cols-2 gap-6">
          {ragComparison.map((side) => (
            <AnimatedCard key={side.title}>
              <h3 className="text-lg font-semibold mb-4" style={{ color: side.color }}>
                {side.title}
              </h3>
              <ul className="space-y-3">
                {side.items.map((item, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm text-gray-300">
                    <ArrowRight size={14} className="mt-1 shrink-0" style={{ color: side.color }} />
                    {item}
                  </li>
                ))}
              </ul>
            </AnimatedCard>
          ))}
        </div>
      </Section>

      {/* How Graph RAG Works */}
      <Section>
        <h2 className="text-2xl font-bold text-white mb-8 flex items-center gap-3">
          <Zap size={24} className="text-[#F39C12]" />
          How Graph RAG Works
        </h2>
        <div className="flex flex-wrap items-center justify-center gap-2 md:gap-4">
          {pipelineSteps.map((step, i) => (
            <div key={step.label} className="flex items-center gap-2 md:gap-4">
              <motion.div
                initial={{ opacity: 0, scale: 0.5 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.15, duration: 0.4 }}
                className="flex flex-col items-center"
              >
                <div
                  className="w-16 h-16 md:w-20 md:h-20 rounded-xl flex items-center justify-center text-2xl border"
                  style={{ borderColor: step.color, backgroundColor: step.color + '15' }}
                >
                  {step.icon}
                </div>
                <span className="text-xs text-gray-400 mt-2 text-center">{step.label}</span>
              </motion.div>
              {i < pipelineSteps.length - 1 && (
                <motion.div
                  initial={{ opacity: 0 }}
                  whileInView={{ opacity: 1 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.15 + 0.1 }}
                >
                  <ArrowRight size={20} className="text-gray-600" />
                </motion.div>
              )}
            </div>
          ))}
        </div>
        <div className="mt-10 grid md:grid-cols-3 gap-4">
          {[
            { title: 'Entity Extraction', desc: 'LLMs identify entities (people, organizations, concepts) and their relationships from source documents.', color: '#4ECDC4' },
            { title: 'Graph Construction', desc: 'Extracted triples (subject-predicate-object) are stored in a graph database like Neo4j, creating an interconnected web of knowledge.', color: '#6C5CE7' },
            { title: 'Graph-Enhanced Retrieval', desc: 'Queries trigger graph traversals that follow relationship paths, gathering richer, multi-hop context than vector similarity alone.', color: '#F39C12' },
          ].map((card, i) => (
            <AnimatedCard key={card.title} delay={i * 0.1}>
              <h4 className="font-semibold text-white mb-2" style={{ color: card.color }}>{card.title}</h4>
              <p className="text-sm text-gray-400">{card.desc}</p>
            </AnimatedCard>
          ))}
        </div>
      </Section>

      {/* When to Use What */}
      <Section>
        <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
          <Brain size={24} className="text-[#FF6B6B]" />
          When to Use What
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead>
              <tr className="border-b border-white/10">
                <th className="py-3 px-4 text-gray-400 font-medium">Scenario</th>
                <th className="py-3 px-4 text-gray-400 font-medium">Best Approach</th>
                <th className="py-3 px-4 text-gray-400 font-medium">Why</th>
              </tr>
            </thead>
            <tbody>
              {decisionMatrix.map((row, i) => (
                <motion.tr
                  key={row.scenario}
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.08 }}
                  className="border-b border-white/5 hover:bg-white/5 transition-colors"
                >
                  <td className="py-3 px-4 text-gray-300">{row.scenario}</td>
                  <td className="py-3 px-4">
                    <span
                      className="px-2 py-1 rounded text-xs font-medium"
                      style={{
                        color: row.best === 'Traditional RAG' ? '#F39C12' : row.best === 'Graph RAG' ? '#4ECDC4' : '#6C5CE7',
                        backgroundColor: (row.best === 'Traditional RAG' ? '#F39C12' : row.best === 'Graph RAG' ? '#4ECDC4' : '#6C5CE7') + '20',
                      }}
                    >
                      {row.best}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-gray-400">{row.reason}</td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
    </div>
  )
}
