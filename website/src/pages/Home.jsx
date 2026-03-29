import { motion, useInView } from 'framer-motion'
import { useRef } from 'react'
import { Link } from 'react-router-dom'
import { Database, GitBranch, BookOpen, Code, Zap, Brain } from 'lucide-react'
import AnimatedCard from '../components/AnimatedCard'

const features = [
  {
    icon: Database,
    title: 'Knowledge Base',
    desc: '16 comprehensive sections covering graph databases, ontologies, RDF, property graphs, and knowledge representation from fundamentals to advanced topics.',
    color: '#4ECDC4',
    link: '/concepts',
  },
  {
    icon: Code,
    title: '17 Projects',
    desc: 'Hands-on projects from building your first knowledge graph in Neo4j to production-grade Graph RAG pipelines with LightRAG and Microsoft GraphRAG.',
    color: '#6C5CE7',
    link: '/projects',
  },
  {
    icon: Zap,
    title: 'RAG Techniques',
    desc: 'Deep dive into LightRAG, Microsoft GraphRAG, Hybrid RAG, community detection, embeddings, and when each approach beats the others.',
    color: '#F39C12',
    link: '/techniques',
  },
]

const stats = [
  { value: '16', label: 'Sections' },
  { value: '17', label: 'Projects' },
  { value: '6', label: 'Workshops' },
  { value: '5', label: 'Graph Databases' },
]

const techStack = [
  { name: 'Neo4j', icon: Database, color: '#4ECDC4' },
  { name: 'LangChain', icon: GitBranch, color: '#6C5CE7' },
  { name: 'LightRAG', icon: Zap, color: '#F39C12' },
  { name: 'GraphRAG', icon: Brain, color: '#FF6B6B' },
]

export default function Home() {
  const statsRef = useRef(null)
  const statsInView = useInView(statsRef, { once: true, margin: '-50px' })
  const techRef = useRef(null)
  const techInView = useInView(techRef, { once: true, margin: '-50px' })

  return (
    <div className="min-h-screen">
      {/* Hero */}
      <section className="relative overflow-hidden px-6 py-24 md:py-36">
        {/* Background glow effects */}
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-[#4ECDC4]/10 rounded-full blur-[128px] pointer-events-none" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-[#6C5CE7]/10 rounded-full blur-[128px] pointer-events-none" />

        <div className="max-w-4xl mx-auto text-center relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight">
              <span className="bg-gradient-to-r from-[#4ECDC4] via-[#6C5CE7] to-[#F39C12] bg-clip-text text-transparent">
                Knowledge Graphs
              </span>
              <br />
              <span className="text-white">&amp; Graph RAG</span>
            </h1>
          </motion.div>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3 }}
            className="text-lg md:text-xl text-gray-400 max-w-2xl mx-auto mb-10"
          >
            A comprehensive learning repository for understanding knowledge graphs,
            graph databases, and how Graph RAG transforms retrieval-augmented generation
            with structured, relationship-aware reasoning.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.5 }}
            className="flex flex-wrap justify-center gap-4"
          >
            <Link
              to="/concepts"
              className="px-6 py-3 rounded-lg bg-gradient-to-r from-[#4ECDC4] to-[#6C5CE7] text-white font-medium hover:opacity-90 transition-opacity"
            >
              Start Learning
            </Link>
            <Link
              to="/projects"
              className="px-6 py-3 rounded-lg border border-white/20 text-white font-medium hover:bg-white/5 transition-colors"
            >
              View Projects
            </Link>
          </motion.div>
        </div>
      </section>

      {/* Feature cards */}
      <section className="px-6 py-16 max-w-6xl mx-auto">
        <div className="grid md:grid-cols-3 gap-6">
          {features.map((f, i) => (
            <Link to={f.link} key={f.title} className="block no-underline">
              <AnimatedCard delay={i * 0.15} className="h-full hover:border-white/20 transition-colors">
                <f.icon size={32} style={{ color: f.color }} className="mb-4" />
                <h3 className="text-xl font-semibold text-white mb-2">{f.title}</h3>
                <p className="text-gray-400 text-sm leading-relaxed">{f.desc}</p>
              </AnimatedCard>
            </Link>
          ))}
        </div>
      </section>

      {/* Stats */}
      <section ref={statsRef} className="px-6 py-16 border-y border-white/10">
        <div className="max-w-4xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-8">
          {stats.map((s, i) => (
            <motion.div
              key={s.label}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={statsInView ? { opacity: 1, scale: 1 } : {}}
              transition={{ duration: 0.5, delay: i * 0.1 }}
              className="text-center"
            >
              <div className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-[#4ECDC4] to-[#6C5CE7] bg-clip-text text-transparent">
                {s.value}
              </div>
              <div className="text-gray-400 text-sm mt-1">{s.label}</div>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Tech stack */}
      <section ref={techRef} className="px-6 py-16 max-w-4xl mx-auto">
        <motion.h2
          initial={{ opacity: 0 }}
          animate={techInView ? { opacity: 1 } : {}}
          transition={{ duration: 0.6 }}
          className="text-2xl font-bold text-center mb-10 text-white"
        >
          Technologies Covered
        </motion.h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {techStack.map((t, i) => (
            <motion.div
              key={t.name}
              initial={{ opacity: 0, y: 20 }}
              animate={techInView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.5, delay: i * 0.1 }}
              className="flex flex-col items-center gap-2 p-4 rounded-lg border border-white/10 bg-white/5"
            >
              <t.icon size={28} style={{ color: t.color }} />
              <span className="text-sm text-gray-300">{t.name}</span>
            </motion.div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="px-6 py-20 text-center">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <h2 className="text-3xl md:text-4xl font-bold mb-4 text-white">Ready to explore?</h2>
          <p className="text-gray-400 mb-8 max-w-lg mx-auto">
            Dive into concepts, build projects, and master Graph RAG from scratch.
          </p>
          <Link
            to="/graph-rag"
            className="inline-block px-8 py-4 rounded-lg bg-gradient-to-r from-[#6C5CE7] to-[#4ECDC4] text-white font-semibold hover:opacity-90 transition-opacity"
          >
            Explore Graph RAG
          </Link>
        </motion.div>
      </section>
    </div>
  )
}
