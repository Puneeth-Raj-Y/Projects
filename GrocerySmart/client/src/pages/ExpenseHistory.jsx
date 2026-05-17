/* ─────────────────────────────────────────────────────────────
   Expense History — Filtering, Editing & Manual Expense Add
   ───────────────────────────────────────────────────────────── */

import { useState, useEffect } from 'react';
import apiClient from '../services/api';
import { useToast } from '../hooks/useToast';
import { 
  Search, Calendar, Filter, Plus, FileSpreadsheet, 
  FileText, Trash2, Edit2, X, AlertCircle 
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export default function ExpenseHistory() {
  const { success, error } = useToast();

  // Expenses and pagination
  const [expenses, setExpenses] = useState([]);
  const [categories, setCategories] = useState([]);
  const [pagination, setPagination] = useState({ page: 1, totalPages: 1 });
  const [loading, setLoading] = useState(true);

  // Filters
  const [filters, setFilters] = useState({
    search: '',
    category: '',
    start_date: '',
    end_date: '',
    page: 1
  });

  // Modal Control
  const [modalOpen, setModalOpen] = useState(false);
  const [editingExpense, setEditingExpense] = useState(null);
  const [formData, setFormData] = useState({
    description: '',
    amount: '',
    category_id: '',
    expense_date: new Date().toISOString().split('T')[0]
  });

  const fetchExpenses = async () => {
    setLoading(true);
    try {
      const params = {
        page: filters.page,
        limit: 10,
        ...(filters.search && { search: filters.search }),
        ...(filters.category && { category: filters.category }),
        ...(filters.start_date && { start_date: filters.start_date }),
        ...(filters.end_date && { end_date: filters.end_date })
      };
      
      const response = await apiClient.get('/api/expenses', { params });
      setExpenses(response.data.expenses);
      setPagination(response.data.pagination);
    } catch (err) {
      console.error('Failed to load expenses list:', err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    async function init() {
      try {
        const catRes = await apiClient.get('/api/categories');
        setCategories(catRes.data);
      } catch (err) {
        console.error('Failed to load categories:', err.message);
      }
    }
    init();
  }, []);

  useEffect(() => {
    fetchExpenses();
  }, [filters]);

  const handleFilterChange = (e) => {
    setFilters({ ...filters, [e.target.name]: e.target.value, page: 1 });
  };

  const handleOpenAddModal = () => {
    setEditingExpense(null);
    setFormData({
      description: '',
      amount: '',
      category_id: categories[0]?.id || '',
      expense_date: new Date().toISOString().split('T')[0]
    });
    setModalOpen(true);
  };

  const handleOpenEditModal = (exp) => {
    setEditingExpense(exp);
    setFormData({
      description: exp.description.replace(/\s\(\d+x\)$/, ''), // Clean unit trace
      amount: exp.amount,
      category_id: exp.category_id,
      expense_date: exp.expense_date
    });
    setModalOpen(true);
  };

  const handleFormSubmit = async (e) => {
    e.preventDefault();
    try {
      if (editingExpense) {
        await apiClient.put(`/api/expenses/${editingExpense.id}`, formData);
        success('Expense details updated!');
      } else {
        await apiClient.post('/api/expenses', formData);
        success('Manual expense successfully logged!');
      }
      setModalOpen(false);
      fetchExpenses();
    } catch (err) {
      error(err.message || 'Failed to save expense details.');
    }
  };

  const handleDelete = async (id) => {
    if (!confirm('Are you sure you want to delete this expense entry?')) return;
    try {
      await apiClient.delete(`/api/expenses/${id}`);
      success('Expense record deleted.');
      fetchExpenses();
    } catch (err) {
      error('Failed to delete item.');
    }
  };

  const handleExport = (type) => {
    // Correctly resolve the export path for backend deployment (Render/Local)
    const baseApi = import.meta.env.VITE_API_URL || '';
    const token = localStorage.getItem('gs_token');
    
    // In production, we need to pass the authorization token in the URL or load via window.open
    const exportUrl = `${baseApi}/api/expenses/export/${type}?token=${token}`;
    window.open(exportUrl, '_blank');
  };

  return (
    <div className="space-y-8 pb-12 w-full">
      <div className="flex flex-col sm:flex-row justify-between sm:items-center gap-4">
        <div>
          <h2 className="text-2xl md:text-3xl font-extrabold text-white">Expense Ledger</h2>
          <p className="text-sm text-gray-500 font-medium">Add, edit, filter, and export detailed grocery statements.</p>
        </div>
        <button
          onClick={handleOpenAddModal}
          className="px-5 py-3 rounded-xl font-bold bg-emerald-500 text-black hover:bg-emerald-400 flex items-center justify-center gap-2 shadow-lg shadow-emerald-500/10 transition"
        >
          <Plus className="h-5 w-5" />
          Add Expense
        </button>
      </div>

      {/* ── Filter Bar ────────────────────────────────────────── */}
      <div className="glass-panel p-5 rounded-2xl grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4 items-center">
        {/* Search */}
        <div className="relative col-span-1 lg:col-span-2">
          <Search className="absolute left-3.5 top-3.5 h-4.5 w-4.5 text-gray-500" />
          <input 
            type="text" 
            name="search"
            value={filters.search}
            onChange={handleFilterChange}
            placeholder="Search items..." 
            className="w-full pl-11 pr-4 glass-input py-2.5 text-xs" 
          />
        </div>

        {/* Category */}
        <div className="relative">
          <Filter className="absolute left-3.5 top-3.5 h-4.5 w-4.5 text-gray-500" />
          <select 
            name="category"
            value={filters.category}
            onChange={handleFilterChange}
            className="w-full pl-11 pr-4 glass-input py-2.5 text-xs appearance-none"
          >
            <option value="">All Categories</option>
            {categories.map((c) => (
              <option key={c.id} value={c.id}>{c.name}</option>
            ))}
          </select>
        </div>

        {/* Start Date */}
        <div className="relative">
          <Calendar className="absolute left-3.5 top-3.5 h-4.5 w-4.5 text-gray-500" />
          <input 
            type="date" 
            name="start_date"
            value={filters.start_date}
            onChange={handleFilterChange}
            className="w-full pl-11 pr-4 glass-input py-2 text-xs" 
          />
        </div>

        {/* End Date */}
        <div className="relative">
          <Calendar className="absolute left-3.5 top-3.5 h-4.5 w-4.5 text-gray-500" />
          <input 
            type="date" 
            name="end_date"
            value={filters.end_date}
            onChange={handleFilterChange}
            className="w-full pl-11 pr-4 glass-input py-2 text-xs" 
          />
        </div>
      </div>

      {/* Export operations */}
      <div className="flex gap-3 justify-end text-xs font-semibold">
        <button 
          onClick={() => handleExport('excel')}
          className="px-4 py-2.5 bg-gray-950 border border-gray-900 text-gray-300 hover:text-white hover:border-emerald-500/20 rounded-xl flex items-center gap-2 transition"
        >
          <FileSpreadsheet className="h-4.5 w-4.5 text-emerald-400" />
          Export Excel
        </button>
        <button 
          onClick={() => handleExport('pdf')}
          className="px-4 py-2.5 bg-gray-950 border border-gray-900 text-gray-300 hover:text-white hover:border-rose-500/20 rounded-xl flex items-center gap-2 transition"
        >
          <FileText className="h-4.5 w-4.5 text-rose-400" />
          Export PDF
        </button>
      </div>

      {/* ── Expenses List ────────────────────────────────────── */}
      <div className="glass-panel rounded-2xl overflow-hidden">
        {loading ? (
          <div className="p-16 text-center text-gray-500">Querying ledger records...</div>
        ) : expenses.length ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-gray-900 text-gray-500 font-semibold bg-gray-950/20">
                  <th className="p-4">Item details</th>
                  <th className="p-4">Category</th>
                  <th className="p-4">Store Location</th>
                  <th className="p-4">Date</th>
                  <th className="p-4 text-right">Price</th>
                  <th className="p-4 text-center">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-900/60">
                {expenses.map((e) => (
                  <tr key={e.id} className="text-gray-300 hover:bg-gray-950/30 transition">
                    <td className="p-4 font-bold text-white max-w-[200px] truncate">{e.description}</td>
                    <td className="p-4">
                      <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded-lg bg-gray-900 border border-gray-800 text-gray-300 font-semibold">
                        <span>{e.category_icon || '📦'}</span>
                        {e.category_name || 'Others'}
                      </span>
                    </td>
                    <td className="p-4 font-semibold text-gray-400">{e.store_name || 'Manual Log'}</td>
                    <td className="p-4 font-semibold text-gray-400">{e.expense_date}</td>
                    <td className="p-4 text-right font-black text-white">₹{e.amount.toFixed(2)}</td>
                    <td className="p-4 text-center">
                      <div className="flex items-center justify-center gap-2.5">
                        <button 
                          onClick={() => handleOpenEditModal(e)}
                          className="text-gray-400 hover:text-emerald-400 transition"
                        >
                          <Edit2 className="h-4 w-4" />
                        </button>
                        <button 
                          onClick={() => handleDelete(e.id)}
                          className="text-gray-400 hover:text-rose-400 transition"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="p-16 text-center text-gray-500 space-y-2">
            <AlertCircle className="h-10 w-10 text-gray-600 mx-auto" />
            <p className="text-sm font-semibold">No expenses located.</p>
          </div>
        )}
      </div>

      {/* Pagination indicators */}
      {pagination.totalPages > 1 && (
        <div className="flex gap-2 justify-center pt-2">
          {Array.from({ length: pagination.totalPages }, (_, i) => i + 1).map((p) => (
            <button
              key={p}
              onClick={() => setFilters({ ...filters, page: p })}
              className={`h-8 w-8 rounded-lg font-bold text-xs border ${p === pagination.page ? 'bg-emerald-500 text-black border-emerald-500' : 'bg-gray-950 text-gray-400 border-gray-900 hover:text-white'}`}
            >
              {p}
            </button>
          ))}
        </div>
      )}

      {/* ── Add / Edit Modal Drawer ───────────────────────────── */}
      <AnimatePresence>
        {modalOpen && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-6 bg-black/60 backdrop-blur-sm">
            <motion.div 
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="w-full max-w-md glass-panel rounded-3xl p-6 bg-gray-950/90 relative"
            >
              <button 
                onClick={() => setModalOpen(false)}
                className="absolute right-5 top-5 p-1 bg-gray-900 border border-gray-800 text-gray-400 hover:text-white rounded-lg transition"
              >
                <X className="h-4.5 w-4.5" />
              </button>

              <h3 className="font-extrabold text-lg text-white mb-6">
                {editingExpense ? 'Edit Expense Record' : 'Manual Expense Creation'}
              </h3>

              <form onSubmit={handleFormSubmit} className="space-y-4 text-xs font-semibold">
                <div className="space-y-1.5">
                  <label className="text-gray-400">Description</label>
                  <input 
                    type="text" 
                    value={formData.description}
                    onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                    placeholder="e.g. Organic Tomatoes" 
                    className="w-full px-4 glass-input"
                    required 
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1.5">
                    <label className="text-gray-400">Amount (INR)</label>
                    <input 
                      type="number" 
                      value={formData.amount}
                      onChange={(e) => setFormData({ ...formData, amount: e.target.value })}
                      placeholder="e.g. 150.00" 
                      className="w-full px-4 glass-input"
                      required 
                    />
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-gray-400">Expense Date</label>
                    <input 
                      type="date" 
                      value={formData.expense_date}
                      onChange={(e) => setFormData({ ...formData, expense_date: e.target.value })}
                      className="w-full px-4 glass-input py-2.5" 
                      required 
                    />
                  </div>
                </div>

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

                <button 
                  type="submit"
                  className="w-full py-4 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-black font-extrabold shadow-lg shadow-emerald-500/10 transition mt-6"
                >
                  {editingExpense ? 'Update Log' : 'Save Log'}
                </button>
              </form>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}
