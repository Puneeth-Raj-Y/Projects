/* ─────────────────────────────────────────────────────────────
   Budget Planner — Category thresholds & Spent margins
   ───────────────────────────────────────────────────────────── */

import { useState, useEffect } from 'react';
import apiClient from '../services/api';
import { useToast } from '../hooks/useToast';
import { Wallet, Target, AlertTriangle, CheckCircle2, X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export default function BudgetPlanner() {
  const { success, error, warning } = useToast();
  const [budgets, setBudgets] = useState([]);
  const [categories, setCategories] = useState([]);
  const [loading, setLoading] = useState(true);

  // Edit Drawer control
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [formData, setFormData] = useState({
    category_id: '',
    amount: ''
  });

  const fetchBudgets = async () => {
    try {
      const response = await apiClient.get('/api/budgets');
      setBudgets(response.data);
    } catch (err) {
      console.error('Failed to query budgets:', err.message);
    }
  };

  useEffect(() => {
    async function init() {
      setLoading(true);
      try {
        await fetchBudgets();
        const catRes = await apiClient.get('/api/categories');
        setCategories(catRes.data);
        if (catRes.data.length) {
          setFormData(prev => ({ ...prev, category_id: catRes.data[0].id }));
        }
      } catch (err) {
        console.error('Initialization failed:', err.message);
      } finally {
        setLoading(false);
      }
    }
    init();
  }, []);

  const handleOpenDrawer = () => {
    setFormData({
      category_id: categories[0]?.id || '',
      amount: ''
    });
    setDrawerOpen(true);
  };

  const handleFormSubmit = async (e) => {
    e.preventDefault();
    try {
      await apiClient.post('/api/budgets', formData);
      setDrawerOpen(false);
      success('Budget boundary successfully configured!');
      fetchBudgets();
    } catch (err) {
      error(err.message || 'Failed to set budget boundary.');
    }
  };

  const handleDeleteBudget = async (id) => {
    if (!confirm('Are you sure you want to delete this budget target?')) return;
    try {
      await apiClient.delete(`/api/budgets/${id}`);
      success('Budget plan removed successfully.');
      fetchBudgets();
    } catch (err) {
      error('Failed to delete budget plan.');
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#030712]">
        <div className="flex flex-col items-center gap-4">
          <div className="h-10 w-10 border-4 border-emerald-500 border-t-transparent rounded-full animate-spin" />
          <p className="text-sm font-semibold text-gray-400">Loading budget matrix...</p>
        </div>
      </div>
    );
  }

  // Aggregate totals
  const totalBudgeted = budgets.reduce((sum, b) => sum + b.amount, 0);
  const totalSpent = budgets.reduce((sum, b) => sum + b.spent, 0);

  return (
    <div className="space-y-8 pb-12 w-full">
      <div className="flex flex-col sm:flex-row justify-between sm:items-center gap-4">
        <div>
          <h2 className="text-2xl md:text-3xl font-extrabold text-white">Budget Planner</h2>
          <p className="text-sm text-gray-500 font-medium">Set monthly thresholds for grocery categories and avoid overspending.</p>
        </div>
        <button
          onClick={handleOpenDrawer}
          className="px-5 py-3 rounded-xl font-bold bg-emerald-500 text-black hover:bg-emerald-400 flex items-center justify-center gap-2 shadow-lg shadow-emerald-500/10 transition"
        >
          <Target className="h-5 w-5" />
          Set Budget Limit
        </button>
      </div>

      {/* ── Overview cards ────────────────────────────────────── */}
      <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Card 1 */}
        <div className="glass-panel p-6 rounded-2xl relative overflow-hidden flex flex-col justify-between min-h-[120px]">
          <div className="flex justify-between items-start">
            <span className="text-xs font-bold text-gray-500 uppercase tracking-wider">Total Allocated</span>
            <Wallet className="h-5 w-5 text-emerald-400" />
          </div>
          <div>
            <h3 className="text-3xl font-black text-white mt-2">
              ₹{totalBudgeted.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
            </h3>
          </div>
        </div>

        {/* Card 2 */}
        <div className="glass-panel p-6 rounded-2xl relative overflow-hidden flex flex-col justify-between min-h-[120px]">
          <div className="flex justify-between items-start">
            <span className="text-xs font-bold text-gray-500 uppercase tracking-wider">Total Budget Spent</span>
            <Target className="h-5 w-5 text-teal-400" />
          </div>
          <div>
            <h3 className="text-3xl font-black text-white mt-2">
              ₹{totalSpent.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
            </h3>
          </div>
        </div>

        {/* Card 3: Status Summary */}
        <div className="glass-panel p-6 rounded-2xl relative overflow-hidden flex flex-col justify-between min-h-[120px] sm:col-span-2 lg:col-span-1">
          <div className="flex justify-between items-start">
            <span className="text-xs font-bold text-gray-500 uppercase tracking-wider">Wallet Balance</span>
            <CheckCircle2 className="h-5 w-5 text-emerald-400" />
          </div>
          <div>
            <h3 className="text-3xl font-black text-white mt-2">
              ₹{Math.max(0, totalBudgeted - totalSpent).toLocaleString('en-IN', { minimumFractionDigits: 2 })}
            </h3>
          </div>
        </div>
      </div>

      {/* ── Category progress listings ─────────────────────────── */}
      <div className="grid md:grid-cols-2 gap-6">
        {budgets.map((b) => {
          const ratio = b.amount > 0 ? (b.spent / b.amount) * 100 : 0;
          const isOver = b.spent > b.amount;

          return (
            <div key={b.id} className="glass-panel p-6 rounded-2xl space-y-4 relative overflow-hidden flex flex-col justify-between">
              
              <div className="flex justify-between items-start">
                <div className="flex items-center gap-3">
                  <div className="h-10 w-10 rounded-xl bg-gray-950/40 border border-gray-900 flex items-center justify-center text-lg">
                    {b.category_icon || '📦'}
                  </div>
                  <div>
                    <h4 className="font-extrabold text-sm text-white">{b.category_name}</h4>
                    <p className="text-[10px] text-gray-500 font-semibold uppercase tracking-wider">Monthly Limit</p>
                  </div>
                </div>
                <button 
                  onClick={() => handleDeleteBudget(b.id)}
                  className="text-gray-500 hover:text-rose-400 transition text-xs font-semibold"
                >
                  Delete
                </button>
              </div>

              {/* Progress bar */}
              <div className="space-y-1">
                <div className="h-2.5 w-full bg-gray-900 rounded-full overflow-hidden">
                  <div 
                    className={`h-full rounded-full transition-all duration-300 ${isOver ? 'bg-rose-500' : ratio > 80 ? 'bg-amber-500' : 'bg-emerald-500'}`} 
                    style={{ width: `${Math.min(100, ratio)}%` }}
                  />
                </div>
                <div className="flex justify-between items-center text-[10px] font-bold">
                  <span className="text-gray-400">Spent: ₹{b.spent.toFixed(0)}</span>
                  <span className="text-white">Limit: ₹{b.amount.toFixed(0)} ({ratio.toFixed(0)}%)</span>
                </div>
              </div>

              {/* Alert indicator */}
              {isOver && (
                <div className="flex items-center gap-2 p-3 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs font-semibold">
                  <AlertTriangle className="h-4 w-4 shrink-0" />
                  <span>Threshold breached! Reduce item purchasing.</span>
                </div>
              )}
            </div>
          );
        })}

        {!budgets.length && (
          <div className="col-span-2 border-2 border-dashed border-gray-900 rounded-3xl p-16 text-center text-gray-500">
            No budget limits allocated yet. Start by setting your category parameters.
          </div>
        )}
      </div>

      {/* ── Drawer Modal form ─────────────────────────────────── */}
      <AnimatePresence>
        {drawerOpen && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-6 bg-black/60 backdrop-blur-sm">
            <motion.div 
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="w-full max-w-md glass-panel rounded-3xl p-6 bg-gray-950/90 relative"
            >
              <button 
                onClick={() => setDrawerOpen(false)}
                className="absolute right-5 top-5 p-1 bg-gray-900 border border-gray-800 text-gray-400 hover:text-white rounded-lg transition"
              >
                <X className="h-4.5 w-4.5" />
              </button>

              <h3 className="font-extrabold text-lg text-white mb-6">Allocate Budget boundary</h3>

              <form onSubmit={handleFormSubmit} className="space-y-4 text-xs font-semibold">
                <div className="space-y-1.5">
                  <label className="text-gray-400">Select Category</label>
                  <select 
                    value={formData.category_id}
                    onChange={(e) => setFormData({ ...formData, category_id: e.target.value })}
                    className="w-full px-4 glass-input appearance-none"
                    required
                  >
                    {categories.map((c) => (
                      <option key={c.id} value={c.id}>{c.name}</option>
                    ))}
                  </select>
                </div>

                <div className="space-y-1.5">
                  <label className="text-gray-400">Monthly Target (INR)</label>
                  <input 
                    type="number" 
                    value={formData.amount}
                    onChange={(e) => setFormData({ ...formData, amount: e.target.value })}
                    placeholder="e.g. 3000" 
                    className="w-full px-4 glass-input"
                    required 
                  />
                </div>

                <button 
                  type="submit"
                  className="w-full py-4 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-black font-extrabold shadow-lg shadow-emerald-500/10 transition mt-6"
                >
                  Set budget boundary
                </button>
              </form>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}
