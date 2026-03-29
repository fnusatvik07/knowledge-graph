import { motion, useInView } from 'framer-motion'
import { useRef, useState } from 'react'
import { ArrowRight, RotateCcw } from 'lucide-react'
import AnimatedCard from '../components/AnimatedCard'
import CodeBlock from '../components/CodeBlock'

const pipelineSteps = [
  { label: 'Chunk', desc: 'Split documents into manageable text segments', color: '#6C5CE7' },
  { label: 'Extract Entities', desc: 'Use LLMs to identify entities and relationships', color: '#4ECDC4' },
  { label: 'Build Graph', desc: 'Store triples in a knowledge graph (Neo4j, NetworkX)', color: '#F39C12' },
  { label: 'Community Detection', desc: 'Identify clusters via Leiden algorithm for global context', color: '#FF6B6B' },
  { label: 'Query', desc: 'Traverse graph paths and retrieve structured context', color: '#4ECDC4' },
]

const approaches = [
  {
    title: 'NaiveRAG',
    color: '#F39C12',
    front: {
      how: 'Embeds document chunks into vectors, retrieves top-k by cosine similarity, feeds to LLM.',
      cost: 'Low',
      quality: 'Good for simple queries',
      when: 'Single-hop factual questions, FAQ systems, document search',
    },
    back: {
      pros: ['Fast retrieval', 'Simple to implement', 'Low compute cost', 'Works with any document type'],
      cons: ['Misses relationships between entities', 'Cannot do multi-hop reasoning', 'No global context understanding', 'Struggles with aggregation queries'],
    },
  },
  {
    title: 'GraphRAG',
    color: '#4ECDC4',
    front: {
      how: 'Extracts entities/relationships, builds knowledge graph, uses community detection and graph traversal for retrieval.',
      cost: 'High (indexing)',
      quality: 'Excellent for complex queries',
      when: 'Multi-hop reasoning, relationship queries, holistic summarization',
    },
    back: {
      pros: ['Multi-hop reasoning', 'Global summarization via communities', 'Relationship-aware', 'Handles complex queries'],
      cons: ['Expensive graph construction (many LLM calls)', 'Slower indexing time', 'Requires graph database infrastructure', 'Overkill for simple queries'],
    },
  },
  {
    title: 'LightRAG',
    color: '#6C5CE7',
    front: {
      how: 'Lightweight graph construction with dual-level retrieval: entity-level for specifics, relationship-level for connections.',
      cost: 'Medium',
      quality: 'Very good balance',
      when: 'When you need graph benefits without full GraphRAG cost',
    },
    back: {
      pros: ['Cheaper than full GraphRAG', 'Dual-level retrieval', 'Incremental graph updates', 'Good performance-cost trade-off'],
      cons: ['Less comprehensive than GraphRAG', 'Fewer community-level features', 'Newer, less battle-tested', 'Limited global summarization'],
    },
  },
  {
    title: 'HybridRAG',
    color: '#FF6B6B',
    front: {
      how: 'Combines vector retrieval with graph traversal. Uses vectors for initial retrieval, then expands context via graph neighbors.',
      cost: 'Medium-High',
      quality: 'Best overall',
      when: 'Production systems needing both speed and depth',
    },
    back: {
      pros: ['Best of both worlds', 'Handles all query types', 'Graceful fallback', 'Production-ready pattern'],
      cons: ['More complex architecture', 'Two retrieval systems to maintain', 'Needs careful query routing', 'Higher operational overhead'],
    },
  },
]

const codeSnippets = [
  {
    language: 'python',
    title: 'Classic RAG',
    code: `from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# Load and split documents
docs = text_splitter.split_documents(raw_docs)

# Create vector store
vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())

# Build retrieval chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4"),
    retriever=vectorstore.as_retriever(k=5),
    return_source_documents=True,
)

result = qa_chain.invoke({"query": "What is knowledge graph?"})`,
  },
  {
    language: 'python',
    title: 'Graph RAG (Neo4j + LangChain)',
    code: `from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI

# Connect to Neo4j knowledge graph
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="password"
)

# Build graph QA chain
chain = GraphCypherQAChain.from_llm(
    llm=ChatOpenAI(model="gpt-4", temperature=0),
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True,
)

# Query with graph-aware retrieval
result = chain.invoke({
    "query": "Which researchers collaborated on graph neural networks?"
})`,
  },
  {
    language: 'python',
    title: 'Hybrid RAG',
    code: `from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import Chroma
from langchain_community.graphs import Neo4jGraph

# Vector retriever for semantic search
vector_retriever = vectorstore.as_retriever(k=3)

# Graph retriever for relationship-aware context
graph_retriever = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    graph=graph,
    node_label="Entity",
    embedding_node_property="embedding",
).as_retriever(k=3)

# Combine both retrievers
hybrid = EnsembleRetriever(
    retrievers=[vector_retriever, graph_retriever],
    weights=[0.5, 0.5],
)

docs = hybrid.invoke("How does GraphRAG handle multi-hop queries?")`,
  },
]

function FlipCard({ approach }) {
  const [flipped, setFlipped] = useState(false)

  return (
    <div
      className="cursor-pointer perspective-[1000px] min-h-[320px]"
      onClick={() => setFlipped(!flipped)}
    >
      <motion.div
        animate={{ rotateY: flipped ? 180 : 0 }}
        transition={{ duration: 0.5 }}
        style={{ transformStyle: 'preserve-3d' }}
        className="relative w-full h-full"
      >
        {/* Front */}
        <div
          className="absolute inset-0 rounded-xl border p-6 backface-hidden"
          style={{
            borderColor: approach.color + '40',
            backgroundColor: approach.color + '08',
            backfaceVisibility: 'hidden',
          }}
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-bold" style={{ color: approach.color }}>{approach.title}</h3>
            <RotateCcw size={16} className="text-gray-500" />
          </div>
          <div className="space-y-3 text-sm">
            <div>
              <span className="text-gray-500 text-xs uppercase tracking-wide">How it works</span>
              <p className="text-gray-300 mt-1">{approach.front.how}</p>
            </div>
            <div className="flex gap-4">
              <div>
                <span className="text-gray-500 text-xs uppercase tracking-wide">Cost</span>
                <p className="text-gray-300 mt-1">{approach.front.cost}</p>
              </div>
              <div>
                <span className="text-gray-500 text-xs uppercase tracking-wide">Quality</span>
                <p className="text-gray-300 mt-1">{approach.front.quality}</p>
              </div>
            </div>
            <div>
              <span className="text-gray-500 text-xs uppercase tracking-wide">When to use</span>
              <p className="text-gray-300 mt-1">{approach.front.when}</p>
            </div>
          </div>
          <p className="text-xs text-gray-600 mt-4 text-center">Click to flip</p>
        </div>

        {/* Back */}
        <div
          className="absolute inset-0 rounded-xl border p-6"
          style={{
            borderColor: approach.color + '40',
            backgroundColor: approach.color + '08',
            backfaceVisibility: 'hidden',
            transform: 'rotateY(180deg)',
          }}
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-bold" style={{ color: approach.color }}>{approach.title}</h3>
            <RotateCcw size={16} className="text-gray-500" />
          </div>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-green-400 text-xs uppercase tracking-wide font-medium">Pros</span>
              <ul className="mt-2 space-y-1">
                {approach.back.pros.map((p) => (
                  <li key={p} className="text-gray-300 flex items-start gap-1">
                    <span className="text-green-400 mt-0.5">+</span> {p}
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <span className="text-red-400 text-xs uppercase tracking-wide font-medium">Cons</span>
              <ul className="mt-2 space-y-1">
                {approach.back.cons.map((c) => (
                  <li key={c} className="text-gray-300 flex items-start gap-1">
                    <span className="text-red-400 mt-0.5">-</span> {c}
                  </li>
                ))}
              </ul>
            </div>
          </div>
          <p className="text-xs text-gray-600 mt-4 text-center">Click to flip back</p>
        </div>
      </motion.div>
    </div>
  )
}

export default function GraphRag() {
  const pipelineRef = useRef(null)
  const pipelineInView = useInView(pipelineRef, { once: true, margin: '-50px' })

  return (
    <div className="max-w-6xl mx-auto px-6 py-16">
      <motion.h1
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-4xl md:text-5xl font-bold mb-4 text-center"
      >
        <span className="bg-gradient-to-r from-[#4ECDC4] to-[#6C5CE7] bg-clip-text text-transparent">
          Graph RAG Deep Dive
        </span>
      </motion.h1>
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="text-gray-400 text-center mb-16 max-w-2xl mx-auto"
      >
        How knowledge graphs transform retrieval-augmented generation from simple
        vector lookup into relationship-aware, multi-hop reasoning.
      </motion.p>

      {/* Pipeline diagram */}
      <section ref={pipelineRef} className="mb-20">
        <h2 className="text-2xl font-bold text-white mb-8 text-center">The Graph RAG Pipeline</h2>
        <div className="flex flex-col md:flex-row items-center justify-center gap-4">
          {pipelineSteps.map((step, i) => (
            <div key={step.label} className="flex items-center gap-4">
              <motion.div
                initial={{ opacity: 0, scale: 0.5 }}
                animate={pipelineInView ? { opacity: 1, scale: 1 } : {}}
                transition={{ delay: i * 0.2, duration: 0.4 }}
                className="flex flex-col items-center text-center w-36"
              >
                <div
                  className="w-16 h-16 rounded-xl flex items-center justify-center text-xl font-bold border-2 mb-2"
                  style={{ borderColor: step.color, backgroundColor: step.color + '15', color: step.color }}
                >
                  {i + 1}
                </div>
                <span className="text-sm font-medium text-white">{step.label}</span>
                <span className="text-xs text-gray-500 mt-1">{step.desc}</span>
              </motion.div>
              {i < pipelineSteps.length - 1 && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={pipelineInView ? { opacity: 1 } : {}}
                  transition={{ delay: i * 0.2 + 0.1 }}
                  className="hidden md:block"
                >
                  <ArrowRight size={20} className="text-gray-600" />
                </motion.div>
              )}
            </div>
          ))}
        </div>
      </section>

      {/* Comparison cards */}
      <section className="mb-20">
        <h2 className="text-2xl font-bold text-white mb-8 text-center">Compare Approaches</h2>
        <div className="grid md:grid-cols-2 gap-6">
          {approaches.map((a, i) => (
            <motion.div
              key={a.title}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1 }}
            >
              <FlipCard approach={a} />
            </motion.div>
          ))}
        </div>
      </section>

      {/* Code snippets */}
      <section>
        <h2 className="text-2xl font-bold text-white mb-8 text-center">Code Examples</h2>
        <div className="space-y-8">
          {codeSnippets.map((snippet, i) => (
            <AnimatedCard key={snippet.title} delay={i * 0.1} className="p-0 overflow-hidden">
              <div className="px-6 pt-5 pb-3">
                <h3 className="text-lg font-semibold text-white">{snippet.title}</h3>
              </div>
              <div className="px-4 pb-4">
                <CodeBlock code={snippet.code} language={snippet.language} />
              </div>
            </AnimatedCard>
          ))}
        </div>
      </section>
    </div>
  )
}
