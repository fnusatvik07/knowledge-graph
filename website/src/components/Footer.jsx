import { ExternalLink, Heart } from 'lucide-react'

export default function Footer() {
  return (
    <footer className="border-t border-white/10 bg-[#0d1117]">
      <div className="max-w-6xl mx-auto px-6 py-8 flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-2 text-gray-400 text-sm">
          <span>Built with</span>
          <Heart size={14} className="text-[#FF6B6B]" />
          <span>for the Knowledge Graph community</span>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-xs text-gray-500">React + Vite + Tailwind + Framer Motion</span>
          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-400 hover:text-white transition-colors"
          >
            <ExternalLink size={20} />
          </a>
        </div>
      </div>
    </footer>
  )
}
