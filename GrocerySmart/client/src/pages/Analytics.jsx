/* ─────────────────────────────────────────────────────────────
   Analytics Page — Detailed Graphic Spending Reviews
   ───────────────────────────────────────────────────────────── */

import { useState, useEffect } from 'react';
import apiClient from '../services/api';
import { 
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, 
  Tooltip, PieChart as RechartsPieChart, Pie, Cell 
} from 'recharts';
import { PieChart, BarChart3, Sparkles } from 'lucide-react';

export default function Analytics() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchAnalytics() {
      try {
        const response = await apiClient.get('/api/analytics/dashboard');
        setData(response.data);
      } catch (err) {
        console.error('Failed to query analytics:', err.message);
      } finally {
        setLoading(false);
      }
    }
    fetchAnalytics();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#030712]">
        <div className="flex flex-col items-center gap-4">
          <div className="h-10 w-10 border-4 border-emerald-500 border-t-transparent rounded-full animate-spin" />
          <p className="text-sm font-semibold text-gray-400">Loading graphical parameters...</p>
        </div>
      </div>
    );
  }

  const categoryExpenses = data?.categoryExpenses || [];
  const weeklyTrends = data?.weeklyTrends || [];
  
  // Custom tooltips
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      return (
        <div className="glass-panel p-3.5 rounded-xl border border-gray-800 bg-[#0f172a]/95 text-xs text-white">
          <p className="font-bold">{payload[0].name}</p>
          <p className="font-semibold text-emerald-400 mt-1">₹{payload[0].value.toFixed(2)}</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-8 pb-12 w-full">
      <div>
        <h2 className="text-2xl md:text-3xl font-extrabold text-white">Analytics Hub</h2>
        <p className="text-sm text-gray-500 font-medium">Deep-dive graphics showcasing monthly spending vectors.</p>
      </div>

      {/* Charts Grid */}
      <div className="grid lg:grid-cols-2 gap-8">
        
        {/* Graph 1: Bar chart shares */}
        <div className="glass-panel p-6 rounded-2xl space-y-6">
          <div className="flex justify-between items-center">
            <div>
              <h4 className="font-extrabold text-lg text-white">Share Distribution</h4>
              <p className="text-xs text-gray-500 font-medium">Category spending comparison</p>
            </div>
            <BarChart3 className="h-5 w-5 text-gray-500" />
          </div>

          <div className="h-80 w-full">
            {categoryExpenses.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={categoryExpenses} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <XAxis dataKey="category" tick={{ fill: '#6b7280', fontSize: 10 }} tickLine={false} axisLine={false} />
                  <YAxis tick={{ fill: '#6b7280', fontSize: 10 }} tickLine={false} axisLine={false} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="value" name="Amount" radius={[8, 8, 0, 0]}>
                    {categoryExpenses.map((entry, idx) => (
                      <Cell key={`cell-${idx}`} fill={entry.color || '#10b981'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center border border-dashed border-gray-800 rounded-xl">
                <span className="text-xs text-gray-500">Scan bills to construct charts.</span>
              </div>
            )}
          </div>
        </div>

        {/* Graph 2: Pie chart splits */}
        <div className="glass-panel p-6 rounded-2xl space-y-6">
          <div className="flex justify-between items-center">
            <div>
              <h4 className="font-extrabold text-lg text-white">Expense Share Allocation</h4>
              <p className="text-xs text-gray-500 font-medium">Visual proportion split</p>
            </div>
            <PieChart className="h-5 w-5 text-gray-500" />
          </div>

          <div className="h-80 w-full flex flex-col md:flex-row items-center justify-center gap-6">
            {categoryExpenses.length ? (
              <>
                <div className="h-60 w-60 shrink-0">
                  <ResponsiveContainer width="100%" height="100%">
                    <RechartsPieChart>
                      <Tooltip content={<CustomTooltip />} />
                      <Pie
                        data={categoryExpenses}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={80}
                        paddingAngle={3}
                        dataKey="value"
                        nameKey="category"
                      >
                        {categoryExpenses.map((entry, idx) => (
                          <Cell key={`cell-${idx}`} fill={entry.color || '#10b981'} />
                        ))}
                      </Pie>
                    </RechartsPieChart>
                  </ResponsiveContainer>
                </div>
                
                {/* Visual Legend */}
                <div className="flex-1 space-y-2.5 max-h-[220px] overflow-y-auto w-full pr-1">
                  {categoryExpenses.map((c, idx) => (
                    <div key={idx} className="flex justify-between items-center text-xs font-semibold">
                      <span className="flex items-center gap-2 text-white">
                        <span className="h-2.5 w-2.5 rounded-full shrink-0" style={{ backgroundColor: c.color }} />
                        {c.category}
                      </span>
                      <span className="text-gray-400">₹{c.value.toFixed(0)}</span>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div className="h-full w-full flex items-center justify-center border border-dashed border-gray-800 rounded-xl">
                <span className="text-xs text-gray-500">Scan bills to populate chart.</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* AI Premium Highlights block */}
      <div className="glass-panel p-6 rounded-2xl relative overflow-hidden flex flex-col md:flex-row md:items-center justify-between gap-6 bg-gradient-to-r from-emerald-500/10 to-teal-500/5 border-emerald-500/20">
        <div className="flex gap-4">
          <div className="p-3 bg-emerald-500/20 border border-emerald-500/30 text-emerald-400 rounded-xl h-12 w-12 flex items-center justify-center shrink-0">
            <Sparkles className="h-6 w-6 animate-pulse" />
          </div>
          <div>
            <h4 className="font-extrabold text-white text-base">Intelligent Pattern Analysis</h4>
            <p className="text-xs text-gray-400 leading-relaxed mt-1 max-w-xl">
              GrocerySmart continuously parses grocery products using custom AI classifiers to discover optimization vectors.
            </p>
          </div>
        </div>
        <div className="flex gap-4 items-center">
          <div className="text-right">
            <span className="text-[10px] font-bold text-gray-500 uppercase tracking-wider block">Average Week Outlay</span>
            <span className="text-lg font-black text-white">₹{
              weeklyTrends.length 
                ? (weeklyTrends.reduce((s,w) => s + w.total, 0) / weeklyTrends.length).toFixed(2)
                : '0.00'
            }</span>
          </div>
        </div>
      </div>
    </div>
  );
}
