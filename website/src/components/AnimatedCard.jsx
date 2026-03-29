import { motion, useInView } from 'framer-motion'
import { useRef } from 'react'

export default function AnimatedCard({ children, className = '', delay = 0 }) {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: true, margin: '-50px' })

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 40 }}
      animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 40 }}
      transition={{ duration: 0.6, delay, ease: 'easeOut' }}
      whileHover={{ scale: 1.02, transition: { duration: 0.2 } }}
      className={`rounded-xl border border-white/10 bg-white/5 backdrop-blur-sm p-6 ${className}`}
    >
      {children}
    </motion.div>
  )
}
