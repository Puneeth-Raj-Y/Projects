/* ─────────────────────────────────────────────────────────────
   Auth Page — Login / Register Combo Glassmorphism portal
   ───────────────────────────────────────────────────────────── */

import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { motion } from 'framer-motion';
import { ShieldCheck, Mail, Lock, User, AlertCircle, ArrowRight } from 'lucide-react';

export default function Auth({ defaultTab = 'login' }) {
  const navigate = useNavigate();
  const { login, register } = useAuth();
  
  const [tab, setTab] = useState(defaultTab);
  const [formData, setFormData] = useState({ name: '', email: '', password: '' });
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const handleInputChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!formData.email || !formData.password || (tab === 'register' && !formData.name)) {
      setError('Please fill in all required fields');
      return;
    }

    setSubmitting(true);
    setError('');

    try {
      if (tab === 'login') {
        await login(formData.email, formData.password);
      } else {
        await register(formData.name, formData.email, formData.password);
      }
      navigate('/dashboard');
    } catch (err) {
      setError(err.response?.data?.error || 'Authentication failed. Please verify credentials.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="relative min-h-screen bg-[#030712] overflow-hidden flex items-center justify-center p-6">
      {/* ── Glowing neon grids ────────────────────────────────── */}
      <div className="glowing-mesh w-[400px] h-[400px] bg-emerald-500/10 top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 pulse-glow" />

      {/* Main Glass Box */}
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-md glass-panel rounded-3xl p-8 relative z-10 bg-gray-950/40"
      >
        {/* Brand header */}
        <div className="text-center space-y-2 mb-8">
          <Link to="/" className="inline-flex items-center gap-2">
            <span className="text-3xl">🥬</span>
            <span className="font-extrabold text-2xl bg-gradient-to-r from-emerald-400 to-teal-300 bg-clip-text text-transparent">
              GrocerySmart
            </span>
          </Link>
          <p className="text-xs text-gray-500 font-semibold uppercase tracking-wider">
            {tab === 'login' ? 'Welcome back to spend tracker' : 'Register your smart wallet'}
          </p>
        </div>

        {/* Tab triggers */}
        <div className="flex bg-gray-950/80 p-1.5 rounded-2xl border border-gray-900 mb-6">
          <button
            onClick={() => { setTab('login'); setError(''); }}
            className={`flex-1 py-3 text-sm font-bold rounded-xl transition duration-200 ${tab === 'login' ? 'bg-emerald-500 text-black shadow-lg shadow-emerald-500/10' : 'text-gray-400 hover:text-white'}`}
          >
            Sign In
          </button>
          <button
            onClick={() => { setTab('register'); setError(''); }}
            className={`flex-1 py-3 text-sm font-bold rounded-xl transition duration-200 ${tab === 'register' ? 'bg-emerald-500 text-black shadow-lg shadow-emerald-500/10' : 'text-gray-400 hover:text-white'}`}
          >
            Sign Up
          </button>
        </div>

        {/* Error Alert Box */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-center gap-3 p-4 bg-rose-500/10 border border-rose-500/20 text-rose-400 rounded-2xl mb-6 text-sm"
          >
            <AlertCircle className="h-5 w-5 shrink-0" />
            <p className="font-medium">{error}</p>
          </motion.div>
        )}

        {/* Auth form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          {tab === 'register' && (
            <div className="space-y-1.5">
              <label className="text-xs font-bold text-gray-400 uppercase tracking-wider px-1">Full Name</label>
              <div className="relative">
                <User className="absolute left-4 top-3.5 h-5 w-5 text-gray-500" />
                <input
                  type="text"
                  name="name"
                  value={formData.name}
                  onChange={handleInputChange}
                  placeholder="John Doe"
                  className="w-full pl-12 pr-4 glass-input"
                  required
                />
              </div>
            </div>
          )}

          <div className="space-y-1.5">
            <label className="text-xs font-bold text-gray-400 uppercase tracking-wider px-1">Email Address</label>
            <div className="relative">
              <Mail className="absolute left-4 top-3.5 h-5 w-5 text-gray-500" />
              <input
                type="email"
                name="email"
                value={formData.email}
                onChange={handleInputChange}
                placeholder="yourname@gmail.com"
                className="w-full pl-12 pr-4 glass-input"
                required
              />
            </div>
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-bold text-gray-400 uppercase tracking-wider px-1">Password</label>
            <div className="relative">
              <Lock className="absolute left-4 top-3.5 h-5 w-5 text-gray-500" />
              <input
                type="password"
                name="password"
                value={formData.password}
                onChange={handleInputChange}
                placeholder="••••••••"
                className="w-full pl-12 pr-4 glass-input"
                required
              />
            </div>
          </div>

          {/* Form Submit Button */}
          <button
            type="submit"
            disabled={submitting}
            className="w-full mt-6 py-4 rounded-2xl bg-gradient-to-r from-emerald-500 to-teal-400 hover:opacity-90 text-black font-extrabold flex items-center justify-center gap-2.5 shadow-xl shadow-emerald-500/10 transition disabled:opacity-50"
          >
            {submitting ? 'Authenticating...' : tab === 'login' ? 'Access Account' : 'Register Securely'}
            {!submitting && <ArrowRight className="h-5 w-5" />}
          </button>
        </form>

        {/* Demo Hint Footer */}
        <div className="mt-8 border-t border-gray-900 pt-5 text-center">
          <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-gray-950 border border-gray-900 text-xs text-gray-500">
            <ShieldCheck className="h-4 w-4 text-emerald-500" />
            <span>Try demo: <strong className="text-emerald-400 font-bold">demo@grocerysmart.com</strong> / <strong className="text-emerald-400 font-bold">demo123</strong></span>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
