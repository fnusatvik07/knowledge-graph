import { Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Footer from './components/Footer'
import Home from './pages/Home'
import Concepts from './pages/Concepts'
import GraphRag from './pages/GraphRag'
import Projects from './pages/Projects'
import Comparison from './pages/Comparison'
import Techniques from './pages/Techniques'

function App() {
  return (
    <div className="min-h-screen bg-[#0d1117] text-white">
      <Navbar />
      <main>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/concepts" element={<Concepts />} />
          <Route path="/graph-rag" element={<GraphRag />} />
          <Route path="/techniques" element={<Techniques />} />
          <Route path="/projects" element={<Projects />} />
          <Route path="/comparison" element={<Comparison />} />
        </Routes>
      </main>
      <Footer />
    </div>
  )
}

export default App
