import { motion } from 'framer-motion'
import { useState } from 'react'
import AnimatedCard from '../components/AnimatedCard'
import { Database, GitBranch, Code, Zap, Brain, BookOpen } from 'lucide-react'

const projects = [
  { num: 1, title: 'Your First Knowledge Graph', level: 'Beginner', tech: ['Neo4j', 'Python'], desc: 'Build a simple knowledge graph from scratch using Neo4j and learn Cypher query language fundamentals.' },
  { num: 2, title: 'RDF & SPARQL Basics', level: 'Beginner', tech: ['RDFLib', 'SPARQL'], desc: 'Create RDF triples, model ontologies, and query linked data with SPARQL endpoints.' },
  { num: 3, title: 'Property Graph Modeling', level: 'Beginner', tech: ['Neo4j', 'Cypher'], desc: 'Design a property graph schema for a movie database with nodes, relationships, and properties.' },
  { num: 4, title: 'NetworkX Graph Analytics', level: 'Beginner', tech: ['NetworkX', 'Python'], desc: 'Analyze graph structures with centrality measures, community detection, and path algorithms.' },
  { num: 5, title: 'Classic RAG Pipeline', level: 'Beginner', tech: ['LangChain', 'ChromaDB'], desc: 'Build a standard RAG pipeline with document chunking, embeddings, and vector retrieval.' },
  { num: 6, title: 'Neo4j + LangChain Integration', level: 'Intermediate', tech: ['Neo4j', 'LangChain'], desc: 'Connect LangChain to Neo4j for graph-based question answering with Cypher generation.' },
  { num: 7, title: 'Entity Extraction Pipeline', level: 'Intermediate', tech: ['SpaCy', 'GPT-4'], desc: 'Extract entities and relationships from unstructured text using NLP and LLMs.' },
  { num: 8, title: 'Knowledge Graph from Documents', level: 'Intermediate', tech: ['LangChain', 'Neo4j'], desc: 'Automatically construct a knowledge graph from PDF documents using LLM-based extraction.' },
  { num: 9, title: 'Graph RAG with Neo4j', level: 'Intermediate', tech: ['Neo4j', 'LangChain'], desc: 'Implement Graph RAG by combining vector retrieval with graph traversal for enhanced context.' },
  { num: 10, title: 'LightRAG Implementation', level: 'Intermediate', tech: ['LightRAG', 'Python'], desc: 'Deploy LightRAG for lightweight graph-enhanced retrieval with dual-level search.' },
  { num: 11, title: 'Microsoft GraphRAG', level: 'Intermediate', tech: ['GraphRAG', 'Python'], desc: 'Use Microsoft GraphRAG framework with community detection and global summarization.' },
  { num: 12, title: 'Multi-Database Graph System', level: 'Advanced', tech: ['Neo4j', 'ArangoDB'], desc: 'Compare and integrate multiple graph databases for different use cases in a single pipeline.' },
  { num: 13, title: 'Hybrid RAG Architecture', level: 'Advanced', tech: ['LangChain', 'Neo4j', 'Chroma'], desc: 'Build a production hybrid RAG system combining vector and graph retrieval with intelligent routing.' },
  { num: 14, title: 'Graph Neural Networks for KG', level: 'Advanced', tech: ['PyTorch', 'DGL'], desc: 'Apply GNNs to knowledge graph embeddings for link prediction and node classification.' },
  { num: 15, title: 'Temporal Knowledge Graphs', level: 'Advanced', tech: ['Neo4j', 'Python'], desc: 'Model time-evolving knowledge with temporal relationships and historical graph queries.' },
  { num: 16, title: 'KG-Powered Agent System', level: 'Advanced', tech: ['LangGraph', 'Neo4j'], desc: 'Build an AI agent that uses a knowledge graph as its memory and reasoning backbone.' },
  { num: 17, title: 'Production Graph RAG Pipeline', level: 'Advanced', tech: ['FastAPI', 'Neo4j', 'Redis'], desc: 'Deploy a scalable Graph RAG system with caching, monitoring, and API endpoints.' },
]

const levelColors = {
  Beginner: '#4ECDC4',
  Intermediate: '#F39C12',
  Advanced: '#FF6B6B',
}

const levelIcons = {
  Beginner: BookOpen,
  Intermediate: Zap,
  Advanced: Brain,
}

export default function Projects() {
  const [filter, setFilter] = useState('All')

  const filtered = filter === 'All' ? projects : projects.filter((p) => p.level === filter)

  return (
    <div className="max-w-6xl mx-auto px-6 py-16">
      <motion.h1
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-4xl md:text-5xl font-bold mb-4 text-center"
      >
        <span className="bg-gradient-to-r from-[#4ECDC4] to-[#6C5CE7] bg-clip-text text-transparent">
          Projects
        </span>
      </motion.h1>
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="text-gray-400 text-center mb-10 max-w-2xl mx-auto"
      >
        17 hands-on projects taking you from your first knowledge graph to production-grade
        Graph RAG systems. Each project builds on the last.
      </motion.p>

      {/* Filter buttons */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="flex flex-wrap justify-center gap-3 mb-12"
      >
        {['All', 'Beginner', 'Intermediate', 'Advanced'].map((level) => (
          <button
            key={level}
            onClick={() => setFilter(level)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all cursor-pointer ${
              filter === level
                ? 'text-white shadow-lg'
                : 'text-gray-400 bg-white/5 border border-white/10 hover:bg-white/10'
            }`}
            style={
              filter === level
                ? {
                    backgroundColor:
                      level === 'All'
                        ? '#6C5CE7'
                        : levelColors[level],
                  }
                : {}
            }
          >
            {level} {level !== 'All' && `(${projects.filter((p) => p.level === level).length})`}
            {level === 'All' && ` (${projects.length})`}
          </button>
        ))}
      </motion.div>

      {/* Project grid */}
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-5">
        {filtered.map((project, i) => {
          const Icon = levelIcons[project.level]
          return (
            <AnimatedCard key={project.num} delay={i * 0.05} className="relative overflow-hidden">
              {/* Number badge */}
              <div
                className="absolute top-4 right-4 w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold"
                style={{
                  backgroundColor: levelColors[project.level] + '20',
                  color: levelColors[project.level],
                }}
              >
                {project.num}
              </div>

              <div className="pr-10">
                <div className="flex items-center gap-2 mb-3">
                  <Icon size={16} style={{ color: levelColors[project.level] }} />
                  <span
                    className="text-xs font-medium px-2 py-0.5 rounded-full"
                    style={{
                      color: levelColors[project.level],
                      backgroundColor: levelColors[project.level] + '15',
                    }}
                  >
                    {project.level}
                  </span>
                </div>
                <h3 className="text-base font-semibold text-white mb-2">{project.title}</h3>
                <p className="text-sm text-gray-400 mb-4 leading-relaxed">{project.desc}</p>
                <div className="flex flex-wrap gap-2">
                  {project.tech.map((t) => (
                    <span
                      key={t}
                      className="text-xs px-2 py-1 rounded bg-white/5 text-gray-400 border border-white/10"
                    >
                      {t}
                    </span>
                  ))}
                </div>
              </div>
            </AnimatedCard>
          )
        })}
      </div>
    </div>
  )
}
