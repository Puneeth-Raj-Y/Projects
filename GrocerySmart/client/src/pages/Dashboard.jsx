/* ─────────────────────────────────────────────────────────────
   Aesthetic Spend Dashboard — Visual Card Hub
   ───────────────────────────────────────────────────────────── */

import { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import apiClient from '../services/api';
import { 
  TrendingUp, PiggyBank, Sparkles, AlertTriangle, 
  ChevronRight, Calendar, ShoppingBag, PlusCircle 
} from 'lucide-react';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip } from 'recharts';
import { Link } from 'react-router-dom';

export default function Dashboard() {
  const { user } = useAuth();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchDashboard() {
      try {
        const response = await apiClient.get('/api/analytics/dashboard');
        setData(response.data);
      } catch (err) {
        console.error('Failed to load dashboard metrics:', err.message);
      } finally {
        setLoading(false);
      }
    }
    fetchDashboard();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#030712]">
        <div className="flex flex-col items-center gap-4">
          <div className="h-10 w-10 border-4 border-emerald-500 border-t-transparent rounded-full animate-spin" />
          <p className="text-sm font-semibold text-gray-400">Loading your AI Dashboard...</p>
        </div>
      </div>
    );
  }

  // Calculate quick metrics
  const currency = user?.currency === 'INR' ? '₹' : '$';
  const monthlySpent = data?.summary?.monthlySpent || 0;
  const prevMonthlySpent = data?.summary?.previousMonthlySpent || 0;
  const totalSavings = data?.summary?.totalSavings || 0;

  // Percentage change
  let changePercent = 0;
  if (prevMonthlySpent > 0) {
    changePercent = ((monthlySpent - prevMonthlySpent) / prevMonthlySpent) * 100;
  }

  return (
    <div className="space-y-8 pb-12 w-full">
      {/* Welcome Header bar */}
      <div className="flex flex-col sm:flex-row justify-between sm:items-center gap-4">
        <div>
          <h2 className="text-2xl md:text-3xl font-extrabold text-white">
            Hello, {user?.name || 'Puneeth'} 👋
          </h2>
          <p className="text-sm text-gray-500 font-medium">
            Here is your smart grocery expense digest. Check scanned trends below.
          </p>
        </div>
        <div className="flex gap-3">
          <Link
            to="/scanner"
            className="px-5 py-3 rounded-xl font-bold bg-emerald-500 text-black hover:bg-emerald-400 flex items-center gap-2 shadow-lg shadow-emerald-500/10 transition"
          >
            <PlusCircle className="h-5 w-5" />
            Scan Bill
          </Link>
        </div>
      </div>

      {/* ── Key Metrics Cards ─────────────────────────────────── */}
      <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Card 1: Spent this Month */}
        <div className="glass-panel p-6 rounded-2xl relative overflow-hidden flex flex-col justify-between min-h-[140px]">
          <div className="flex justify-between items-start">
            <span className="text-xs font-bold text-gray-500 uppercase tracking-wider">Month Spent</span>
            <TrendingUp className="h-5 w-5 text-emerald-400" />
          </div>
          <div>
            <h3 className="text-3xl font-black text-white mt-2">
              {currency}{monthlySpent.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
            </h3>
            <p className="text-xs font-semibold text-gray-400 mt-1.5">
              {changePercent > 0 ? (
                <span className="text-rose-400 font-bold">+{changePercent.toFixed(1)}%</span>
              ) : (
                <span className="text-emerald-400 font-bold">{changePercent.toFixed(1)}%</span>
              )}{' '}
              compared to last month ({currency}{prevMonthlySpent.toFixed(0)})
            </p>
          </div>
        </div>

        {/* Card 2: Total Savings */}
        <div className="glass-panel p-6 rounded-2xl relative overflow-hidden flex flex-col justify-between min-h-[140px]">
          <div className="flex justify-between items-start">
            <span className="text-xs font-bold text-gray-500 uppercase tracking-wider">Invoice Savings</span>
            <PiggyBank className="h-5 w-5 text-teal-400" />
          </div>
          <div>
            <h3 className="text-3xl font-black text-white mt-2">
              {currency}{totalSavings.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
            </h3>
            <p className="text-xs text-gray-400 mt-1.5 font-medium">
              Accumulated discounts extracted automatically from bill logs.
            </p>
          </div>
        </div>

        {/* Card 3: Total Expenses Tracker */}
        <div className="glass-panel p-6 rounded-2xl relative overflow-hidden flex flex-col justify-between min-h-[140px] sm:col-span-2 lg:col-span-1">
          <div className="flex justify-between items-start">
            <span className="text-xs font-bold text-gray-500 uppercase tracking-wider">All-Time Expenditure</span>
            <ShoppingBag className="h-5 w-5 text-blue-400" />
          </div>
          <div>
            <h3 className="text-3xl font-black text-white mt-2">
              {currency}{(data?.summary?.totalSpent || 0).toLocaleString('en-IN', { minimumFractionDigits: 2 })}
            </h3>
            <p className="text-xs text-gray-400 mt-1.5 font-medium">
              Aggregated summation of all automated & manual expense rows.
            </p>
          </div>
        </div>
      </div>

      {/* ── Main Layout: Analytics + Side Dials ────────────────── */}
      <div className="grid lg:grid-cols-12 gap-8">
        
        {/* Weekly Trend Graph Chart */}
        <div className="lg:col-span-8 glass-panel p-6 rounded-2xl space-y-6">
          <div className="flex justify-between items-center">
            <div>
              <h4 className="font-extrabold text-lg text-white">Expenditure Trends</h4>
              <p className="text-xs text-gray-500 font-medium">Weekly grocery purchasing behavior</p>
            </div>
            <Calendar className="h-5 w-5 text-gray-500" />
          </div>

          <div className="h-72 w-full">
            {data?.weeklyTrends?.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={data.weeklyTrends} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorSpent" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#10b981" stopOpacity={0.2}/>
                      <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="week_start" tick={{ fill: '#6b7280', fontSize: 10 }} tickLine={false} axisLine={false} />
                  <YAxis tick={{ fill: '#6b7280', fontSize: 10 }} tickLine={false} axisLine={false} />
                  <Tooltip 
                    contentStyle={{ background: '#0f172a', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '12px' }}
                    labelStyle={{ color: '#fff', fontSize: 11, fontWeight: 'bold' }}
                    itemStyle={{ color: '#10b981', fontSize: 12 }}
                  />
                  <Area type="monotone" dataKey="total" name="Spent" stroke="#10b981" strokeWidth={2.5} fillOpacity={1} fill="url(#colorSpent)" />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center border border-dashed border-gray-800 rounded-xl">
                <span className="text-xs text-gray-500">Scan bills to construct spending history curves.</span>
              </div>
            )}
          </div>
        </div>

        {/* Categories breakdown dials */}
        <div className="lg:col-span-4 glass-panel p-6 rounded-2xl flex flex-col justify-between">
          <div className="space-y-1 mb-6">
            <h4 className="font-extrabold text-lg text-white">Expense Share</h4>
            <p className="text-xs text-gray-500 font-medium">Category-wise spend concentration</p>
          </div>

          <div className="space-y-4 flex-1">
            {data?.categoryExpenses?.slice(0, 5).map((cat) => {
              const share = ((cat.value / (data.summary.totalSpent || 1)) * 100).toFixed(0);
              return (
                <div key={cat.category} className="space-y-1">
                  <div className="flex justify-between text-xs font-semibold">
                    <span className="flex items-center gap-2 text-white">
                      <span>{cat.icon || '📦'}</span>
                      {cat.category}
                    </span>
                    <span className="text-gray-400">{share}% ({currency}{cat.value.toFixed(0)})</span>
                  </div>
                  <div className="h-2 w-full bg-gray-900 rounded-full overflow-hidden">
                    <div 
                      className="h-full rounded-full" 
                      style={{ width: `${share}%`, backgroundColor: cat.color || '#10b981' }} 
                    />
                  </div>
                </div>
              );
            })}

            {!data?.categoryExpenses?.length && (
              <div className="h-48 flex items-center justify-center border border-dashed border-gray-800 rounded-xl">
                <span className="text-xs text-gray-500">No categories found.</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ── Down-grid: AI Insights + Top Purchased Products ───── */}
      <div className="grid lg:grid-cols-12 gap-8">
        
        {/* AI Insight Box */}
        <div className="lg:col-span-6 glass-panel p-6 rounded-2xl relative overflow-hidden flex flex-col justify-between">
          <div className="flex items-center gap-2 mb-4">
            <Sparkles className="h-5 w-5 text-amber-400 animate-pulse" />
            <h4 className="font-extrabold text-lg text-white">AI Grocery Insights</h4>
          </div>

          <div className="space-y-4 flex-1">
            {data?.aiInsights?.map((insight, idx) => (
              <div key={idx} className="flex gap-3.5 p-4 rounded-xl bg-gray-950/40 border border-gray-900">
                <div className="h-5 w-5 shrink-0 rounded-full bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-[10px] text-emerald-400 font-bold mt-0.5">
                  {idx + 1}
                </div>
                <p className="text-xs font-semibold text-gray-300 leading-relaxed" dangerouslySetInnerHTML={{ __html: insight }} />
              </div>
            ))}

            {!data?.aiInsights?.length && (
              <p className="text-xs text-gray-500">Record a few transactions to activate smart AI recommendations.</p>
            )}
          </div>
        </div>

        {/* Top items table widget */}
        <div className="lg:col-span-6 glass-panel p-6 rounded-2xl flex flex-col justify-between">
          <div className="flex justify-between items-center mb-4">
            <div>
              <h4 className="font-extrabold text-lg text-white">Top Purchased Products</h4>
              <p className="text-xs text-gray-500 font-medium">Most frequently bought grocery items</p>
            </div>
          </div>

          <div className="flex-1 overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-gray-900 text-gray-500 font-semibold">
                  <th className="pb-3">Product</th>
                  <th className="pb-3 text-center">Frequency</th>
                  <th className="pb-3 text-right">Sum Value</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-900/60">
                {data?.topProducts?.map((p, idx) => (
                  <tr key={idx} className="text-gray-300 hover:text-white transition">
                    <td className="py-3 font-semibold truncate max-w-[200px]">{p.name}</td>
                    <td className="py-3 text-center font-bold text-emerald-400">{p.purchase_count}x</td>
                    <td className="py-3 text-right font-semibold">{currency}{p.total_spent.toFixed(2)}</td>
                  </tr>
                ))}

                {!data?.topProducts?.length && (
                  <tr>
                    <td colSpan="3" className="py-12 text-center text-gray-500">Scan bills to track individual products.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
