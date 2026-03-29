import { motion, useInView } from 'framer-motion'
import { useRef } from 'react'
import AnimatedCard from '../components/AnimatedCard'
import { BarChart3, Target, TrendingUp, Award } from 'lucide-react'

const approaches = ['Classic RAG', 'Graph RAG', 'Hybrid RAG']
const approachColors = { 'Classic RAG': '#F39C12', 'Graph RAG': '#4ECDC4', 'Hybrid RAG': '#6C5CE7' }

const questionTypes = [
  {
    type: 'Simple Factual',
    example: '"What is the capital of France?"',
    scores: { 'Classic RAG': 92, 'Graph RAG': 88, 'Hybrid RAG': 93 },
    winner: 'Hybrid RAG',
    insight: 'Vector retrieval excels here. Graph overhead provides minimal benefit for direct lookups.',
  },
  {
    type: 'Multi-hop Reasoning',
    example: '"Which researchers at MIT collaborated with Google on transformer architectures?"',
    scores: { 'Classic RAG': 45, 'Graph RAG': 89, 'Hybrid RAG': 91 },
    winner: 'Hybrid RAG',
    insight: 'Graph traversal connects entities across documents. Classic RAG misses the relationship chain.',
  },
  {
    type: 'Aggregation',
    example: '"Summarize the main themes across all research papers on Graph RAG."',
    scores: { 'Classic RAG': 38, 'Graph RAG': 87, 'Hybrid RAG': 82 },
    winner: 'Graph RAG',
    insight: 'Community detection in GraphRAG produces holistic summaries that vector retrieval cannot match.',
  },
  {
    type: 'Path Tracing',
    example: '"How is entity A connected to entity D through intermediate entities?"',
    scores: { 'Classic RAG': 22, 'Graph RAG': 94, 'Hybrid RAG': 88 },
    winner: 'Graph RAG',
    insight: 'Graph databases natively support path queries. Vector stores have no concept of paths.',
  },
  {
    type: 'Temporal Queries',
    example: '"How has the approach to knowledge representation evolved from 2020 to 2024?"',
    scores: { 'Classic RAG': 52, 'Graph RAG': 78, 'Hybrid RAG': 85 },
    winner: 'Hybrid RAG',
    insight: 'Hybrid approach uses vectors for temporal context and graph for evolution tracking.',
  },
  {
    type: 'Comparative Analysis',
    example: '"Compare Neo4j and ArangoDB for knowledge graph storage."',
    scores: { 'Classic RAG': 68, 'Graph RAG': 75, 'Hybrid RAG': 88 },
    winner: 'Hybrid RAG',
    insight: 'Vectors find relevant comparison docs; graph identifies shared and different properties.',
  },
]

const overallScores = [
  { label: 'Accuracy (avg)', classic: 53, graph: 85, hybrid: 88 },
  { label: 'Latency (lower=better)', classic: 95, graph: 45, hybrid: 65 },
  { label: 'Cost Efficiency', classic: 90, graph: 35, hybrid: 55 },
  { label: 'Complex Query Handling', classic: 30, graph: 90, hybrid: 88 },
  { label: 'Setup Simplicity', classic: 95, graph: 30, hybrid: 45 },
]

const takeaways = [
  {
    title: 'Classic RAG is not dead',
    desc: 'For simple factual queries, classic RAG remains the most cost-effective and fastest approach. Do not over-engineer with graphs when vectors suffice.',
    color: '#F39C12',
    icon: Target,
  },
  {
    title: 'Graph RAG shines on complexity',
    desc: 'When queries require multi-hop reasoning, aggregation, or path tracing, Graph RAG dramatically outperforms vector-only retrieval. The accuracy gap is 40-70% on these query types.',
    color: '#4ECDC4',
    icon: TrendingUp,
  },
  {
    title: 'Hybrid is the production answer',
    desc: 'For real-world systems handling diverse query types, Hybrid RAG provides the best overall results by routing queries to the appropriate retrieval method.',
    color: '#6C5CE7',
    icon: Award,
  },
]

function ScoreBar({ score, color, label, delay = 0 }) {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true })

  return (
    <div ref={ref} className="flex items-center gap-3">
      <span className="text-xs text-gray-500 w-20 text-right shrink-0">{label}</span>
      <div className="flex-1 h-6 bg-white/5 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={inView ? { width: `${score}%` } : {}}
          transition={{ duration: 0.8, delay, ease: 'easeOut' }}
          className="h-full rounded-full flex items-center justify-end pr-2"
          style={{ backgroundColor: color }}
        >
          <span className="text-xs font-bold text-white">{score}</span>
        </motion.div>
      </div>
    </div>
  )
}

export default function Comparison() {
  const tableRef = useRef(null)
  const tableInView = useInView(tableRef, { once: true, margin: '-50px' })

  return (
    <div className="max-w-6xl mx-auto px-6 py-16">
      <motion.h1
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-4xl md:text-5xl font-bold mb-4 text-center"
      >
        <span className="bg-gradient-to-r from-[#F39C12] via-[#4ECDC4] to-[#6C5CE7] bg-clip-text text-transparent">
          RAG Comparison
        </span>
      </motion.h1>
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="text-gray-400 text-center mb-16 max-w-2xl mx-auto"
      >
        Head-to-head comparison of Classic RAG, Graph RAG, and Hybrid RAG across
        different query types, based on benchmark evaluations.
      </motion.p>

      {/* Comparison table */}
      <section ref={tableRef} className="mb-20 overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-white/10">
              <th className="py-3 px-4 text-left text-gray-400 font-medium">Query Type</th>
              {approaches.map((a) => (
                <th key={a} className="py-3 px-4 text-center font-medium" style={{ color: approachColors[a] }}>
                  {a}
                </th>
              ))}
              <th className="py-3 px-4 text-center text-gray-400 font-medium">Winner</th>
            </tr>
          </thead>
          <tbody>
            {questionTypes.map((q, i) => (
              <motion.tr
                key={q.type}
                initial={{ opacity: 0, y: 10 }}
                animate={tableInView ? { opacity: 1, y: 0 } : {}}
                transition={{ delay: i * 0.1 }}
                className="border-b border-white/5 hover:bg-white/[0.03] transition-colors"
              >
                <td className="py-4 px-4">
                  <div className="text-white font-medium">{q.type}</div>
                  <div className="text-xs text-gray-500 mt-0.5 italic">{q.example}</div>
                </td>
                {approaches.map((a) => {
                  const score = q.scores[a]
                  const isWinner = a === q.winner
                  return (
                    <td key={a} className="py-4 px-4 text-center">
                      <span
                        className={`inline-block px-3 py-1 rounded-full text-sm font-mono font-bold ${
                          isWinner ? 'ring-2' : ''
                        }`}
                        style={{
                          color: approachColors[a],
                          backgroundColor: approachColors[a] + '15',
                          ringColor: isWinner ? approachColors[a] : 'transparent',
                        }}
                      >
                        {score}%
                      </span>
                    </td>
                  )
                })}
                <td className="py-4 px-4 text-center">
                  <span
                    className="text-xs font-medium px-2 py-1 rounded"
                    style={{
                      color: approachColors[q.winner],
                      backgroundColor: approachColors[q.winner] + '20',
                    }}
                  >
                    {q.winner}
                  </span>
                </td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </section>

      {/* Insights per query type */}
      <section className="mb-20">
        <h2 className="text-2xl font-bold text-white mb-8 text-center flex items-center justify-center gap-3">
          <BarChart3 size={24} className="text-[#4ECDC4]" />
          Query-Level Insights
        </h2>
        <div className="grid md:grid-cols-2 gap-4">
          {questionTypes.map((q, i) => (
            <AnimatedCard key={q.type} delay={i * 0.08}>
              <h4 className="text-sm font-semibold text-white mb-1">{q.type}</h4>
              <p className="text-xs text-gray-400 mb-3">{q.insight}</p>
              <div className="space-y-2">
                {approaches.map((a, j) => (
                  <ScoreBar
                    key={a}
                    score={q.scores[a]}
                    color={approachColors[a]}
                    label={a.replace(' RAG', '')}
                    delay={j * 0.1}
                  />
                ))}
              </div>
            </AnimatedCard>
          ))}
        </div>
      </section>

      {/* Overall scorecard */}
      <section className="mb-20">
        <h2 className="text-2xl font-bold text-white mb-8 text-center">Overall Scorecard</h2>
        <AnimatedCard className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/10">
                <th className="py-2 px-4 text-left text-gray-400 font-medium">Metric</th>
                <th className="py-2 px-4 text-center font-medium" style={{ color: '#F39C12' }}>Classic</th>
                <th className="py-2 px-4 text-center font-medium" style={{ color: '#4ECDC4' }}>Graph</th>
                <th className="py-2 px-4 text-center font-medium" style={{ color: '#6C5CE7' }}>Hybrid</th>
              </tr>
            </thead>
            <tbody>
              {overallScores.map((row) => (
                <tr key={row.label} className="border-b border-white/5">
                  <td className="py-3 px-4 text-gray-300">{row.label}</td>
                  <td className="py-3 px-4 text-center">
                    <span className="font-mono text-[#F39C12]">{row.classic}</span>
                  </td>
                  <td className="py-3 px-4 text-center">
                    <span className="font-mono text-[#4ECDC4]">{row.graph}</span>
                  </td>
                  <td className="py-3 px-4 text-center">
                    <span className="font-mono text-[#6C5CE7]">{row.hybrid}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </AnimatedCard>
      </section>

      {/* Key takeaways */}
      <section>
        <h2 className="text-2xl font-bold text-white mb-8 text-center">Key Takeaways</h2>
        <div className="grid md:grid-cols-3 gap-6">
          {takeaways.map((t, i) => {
            const Icon = t.icon
            return (
              <motion.div
                key={t.title}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.15, duration: 0.5 }}
                className="rounded-xl border p-6"
                style={{
                  borderColor: t.color + '30',
                  backgroundColor: t.color + '08',
                }}
              >
                <Icon size={28} style={{ color: t.color }} className="mb-3" />
                <h3 className="text-lg font-bold text-white mb-2">{t.title}</h3>
                <p className="text-sm text-gray-400 leading-relaxed">{t.desc}</p>
              </motion.div>
            )
          })}
        </div>
      </section>
    </div>
  )
}
