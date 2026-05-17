/* ─────────────────────────────────────────────────────────────
   Landing Page — Vibrant Dark Showcase UI
   ───────────────────────────────────────────────────────────── */

import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  ScanLine, BrainCircuit, BarChart3, Receipt, 
  Smartphone, ShieldAlert, Sparkles, CheckCircle2, ChevronRight 
} from 'lucide-react';

export default function Landing() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { staggerChildren: 0.15 } }
  };

  const itemVariants = {
    hidden: { y: 30, opacity: 0 },
    visible: { y: 0, opacity: 1, transition: { duration: 0.6, ease: 'easeOut' } }
  };

  return (
    <div className="relative min-h-screen bg-[#030712] overflow-hidden flex flex-col justify-between">
      {/* ── Glowing Mesh Backgrounds ────────────────────────────── */}
      <div className="glowing-mesh w-[500px] h-[500px] bg-emerald-500/10 top-[-10%] left-[-10%] pulse-glow" />
      <div className="glowing-mesh w-[600px] h-[600px] bg-teal-500/10 bottom-[-20%] right-[-10%] pulse-glow" style={{ animationDelay: '3s' }} />

      {/* Header bar */}
      <header className="relative z-10 max-w-7xl mx-auto w-full px-6 md:px-12 h-24 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-2xl">🥬</span>
          <span className="font-extrabold text-xl bg-gradient-to-r from-emerald-400 to-teal-300 bg-clip-text text-transparent">
            GrocerySmart
          </span>
        </div>
        <div className="flex items-center gap-4">
          <Link 
            to="/login" 
            className="px-4 py-2 text-sm font-semibold text-gray-300 hover:text-white transition"
          >
            Sign In
          </Link>
          <Link 
            to="/register" 
            className="px-5 py-2 text-sm font-semibold bg-emerald-500 hover:bg-emerald-400 text-black rounded-xl transition duration-200 shadow-lg shadow-emerald-500/15"
          >
            Get Started
          </Link>
        </div>
      </header>

      {/* Hero section */}
      <main className="relative z-10 max-w-7xl mx-auto w-full px-6 md:px-12 py-12 md:py-24 grid md:grid-cols-12 gap-12 items-center flex-1">
        <motion.div 
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="md:col-span-7 space-y-6 text-center md:text-left"
        >
          <motion.div 
            variants={itemVariants}
            className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-bold uppercase tracking-wider"
          >
            <Sparkles className="h-3.5 w-3.5 animate-pulse" />
            AI-Driven Receipt Recognition
          </motion.div>

          <motion.h2 
            variants={itemVariants}
            className="text-4xl md:text-6xl font-extrabold tracking-tight leading-tight"
          >
            Track Grocery Spending <br />
            <span className="bg-gradient-to-r from-emerald-400 via-teal-300 to-blue-500 bg-clip-text text-transparent">
              With Pure Intelligence
            </span>
          </motion.h2>

          <motion.p 
            variants={itemVariants}
            className="text-gray-400 text-base md:text-lg max-w-xl leading-relaxed"
          >
            Upload grocery receipts using your phone camera. Our machine intelligence automatically detects items, extracts prices, scans barcodes, and categorizes your shopping trends instantly.
          </motion.p>

          <motion.div 
            variants={itemVariants}
            className="flex flex-col sm:flex-row items-center justify-center md:justify-start gap-4 pt-2"
          >
            <Link
              to="/register"
              className="w-full sm:w-auto px-8 py-4 rounded-xl font-bold bg-gradient-to-r from-emerald-500 to-teal-400 text-black hover:opacity-90 flex items-center justify-center gap-2 shadow-xl shadow-emerald-500/20 group transition"
            >
              Start Scanning Free
              <ChevronRight className="h-5 w-5 group-hover:translate-x-1 transition-transform" />
            </Link>
            <a
              href="#features"
              className="w-full sm:w-auto px-8 py-4 rounded-xl font-bold glass-panel hover:bg-gray-900 border border-gray-800 text-white text-center transition"
            >
              Explore Features
            </a>
          </motion.div>

          <motion.div 
            variants={itemVariants}
            className="flex flex-wrap justify-center md:justify-start gap-x-8 gap-y-3 pt-6 text-gray-500 text-xs font-semibold uppercase tracking-wider"
          >
            <span className="flex items-center gap-2"><CheckCircle2 className="h-4 w-4 text-emerald-500" /> 100% Secure Sessions</span>
            <span className="flex items-center gap-2"><CheckCircle2 className="h-4 w-4 text-emerald-500" /> Offline Barcode Engine</span>
            <span className="flex items-center gap-2"><CheckCircle2 className="h-4 w-4 text-emerald-500" /> Rich PDF Reports</span>
          </motion.div>
        </motion.div>

        {/* Hero mockup graphics panel */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9, x: 50 }}
          animate={{ opacity: 1, scale: 1, x: 0 }}
          transition={{ duration: 0.8, delay: 0.3, ease: 'easeOut' }}
          className="md:col-span-5 relative"
        >
          <div className="relative glass-panel rounded-3xl p-6 shadow-2xl border-emerald-500/10 overflow-hidden bg-gray-950/40">
            {/* Visual Glass Header */}
            <div className="flex items-center justify-between border-b border-gray-900 pb-4 mb-5">
              <div className="flex items-center gap-2">
                <span className="h-3 w-3 rounded-full bg-rose-500" />
                <span className="h-3 w-3 rounded-full bg-amber-500" />
                <span className="h-3 w-3 rounded-full bg-emerald-500" />
              </div>
              <span className="text-xs text-gray-500 font-semibold tracking-wider uppercase">GrocerySmart OCR</span>
            </div>

            {/* Simulated OCR scanning feedback */}
            <div className="space-y-4 relative">
              <div className="absolute top-0 bottom-0 left-[18px] w-[2px] bg-gradient-to-b from-emerald-500 to-transparent opacity-20" />

              <div className="flex gap-4">
                <div className="h-9 w-9 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400 font-bold text-sm">
                  1
                </div>
                <div className="space-y-1">
                  <h4 className="text-sm font-semibold text-white">Capture Receipt</h4>
                  <p className="text-xs text-gray-400">Snap a picture of your grocery bill using your mobile camera.</p>
                </div>
              </div>

              <div className="flex gap-4">
                <div className="h-9 w-9 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400 font-bold text-sm">
                  2
                </div>
                <div className="space-y-1">
                  <h4 className="text-sm font-semibold text-white">OCR Parsing</h4>
                  <p className="text-xs text-gray-400">Extracts store name, total prices, products & quantity via AI.</p>
                </div>
              </div>

              <div className="flex gap-4">
                <div className="h-9 w-9 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400 font-bold text-sm">
                  3
                </div>
                <div className="space-y-1">
                  <h4 className="text-sm font-semibold text-white">Smart Category Routing</h4>
                  <p className="text-xs text-gray-400">Identifies items into Vegetables, Snacks, Dairy, or Household automatically.</p>
                </div>
              </div>
            </div>

            {/* Glowing Scan Bar overlay animation */}
            <div className="absolute top-16 left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-emerald-500 to-transparent animate-bounce opacity-40" />
          </div>
        </motion.div>
      </main>

      {/* Feature Section */}
      <section id="features" className="relative z-10 max-w-7xl mx-auto w-full px-6 md:px-12 py-16 border-t border-gray-900">
        <h3 className="text-center font-bold text-2xl md:text-3xl mb-12 bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
          Comprehensive Tooling Stack
        </h3>
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6">
          <div className="glass-panel p-6 rounded-2xl space-y-3">
            <ScanLine className="h-8 w-8 text-emerald-400" />
            <h4 className="text-lg font-bold text-white">Smart Bill Camera</h4>
            <p className="text-sm text-gray-400">Use any mobile browser to take high-fidelity pictures of store invoices with auto contrast.</p>
          </div>
          <div className="glass-panel p-6 rounded-2xl space-y-3">
            <BrainCircuit className="h-8 w-8 text-teal-400" />
            <h4 className="text-lg font-bold text-white">AI Classification</h4>
            <p className="text-sm text-gray-400">Powerful AI categorizer instantly files items under respective dairy, greens, and health budgets.</p>
          </div>
          <div className="glass-panel p-6 rounded-2xl space-y-3">
            <BarChart3 className="h-8 w-8 text-indigo-400" />
            <h4 className="text-lg font-bold text-white">Aesthetic Analytics</h4>
            <p className="text-sm text-gray-400">Beautiful dashboard views, category breakdowns, savings logs, and weekly trending graphs.</p>
          </div>
        </div>
      </section>

      {/* Footer copyright */}
      <footer className="relative z-10 py-8 border-t border-gray-900/60 bg-gray-950/20 text-center text-xs text-gray-500">
        <p>&copy; {new Date().getFullYear()} GrocerySmart. Built for modern financial clarity.</p>
      </footer>
    </div>
  );
}
