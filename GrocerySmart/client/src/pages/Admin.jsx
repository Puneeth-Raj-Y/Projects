/* ─────────────────────────────────────────────────────────────
   Admin Panel — User Telemetry & System Auditing
   ───────────────────────────────────────────────────────────── */

import { useState, useEffect } from 'react';
import apiClient from '../services/api';
import { ShieldAlert, Users, Database, FileClock, Activity } from 'lucide-react';

export default function Admin() {
  const [users, setUsers] = useState([]);
  const [logs, setLogs] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchAdminData() {
      try {
        const [userRes, logRes, metricRes] = await Promise.all([
          apiClient.get('/api/admin/users'),
          apiClient.get('/api/admin/logs'),
          apiClient.get('/api/admin/metrics')
        ]);
        setUsers(userRes.data);
        setLogs(logRes.data);
        setMetrics(metricRes.data);
      } catch (err) {
        console.error('Failed to query admin credentials:', err.message);
      } finally {
        setLoading(false);
      }
    }
    fetchAdminData();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#030712]">
        <div className="flex flex-col items-center gap-4">
          <div className="h-10 w-10 border-4 border-emerald-500 border-t-transparent rounded-full animate-spin" />
          <p className="text-sm font-semibold text-gray-400">Querying System Logs...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8 pb-12 w-full">
      <div className="flex items-center gap-3">
        <ShieldAlert className="h-8 w-8 text-rose-500 animate-pulse" />
        <div>
          <h2 className="text-2xl md:text-3xl font-extrabold text-white">Admin Telemetry Control</h2>
          <p className="text-sm text-gray-500 font-medium font-semibold uppercase tracking-wider">Security & System Performance logs</p>
        </div>
      </div>

      {/* Metrics Row */}
      <div className="grid sm:grid-cols-3 gap-6">
        <div className="glass-panel p-6 rounded-2xl flex items-center gap-4">
          <div className="h-12 w-12 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
            <Users className="h-6 w-6" />
          </div>
          <div>
            <span className="text-[10px] font-bold text-gray-500 uppercase tracking-wider block">Total Accounts</span>
            <span className="text-2xl font-black text-white">{metrics?.totalUsers || 0}</span>
          </div>
        </div>

        <div className="glass-panel p-6 rounded-2xl flex items-center gap-4">
          <div className="h-12 w-12 rounded-xl bg-teal-500/10 border border-teal-500/20 flex items-center justify-center text-teal-400">
            <Database className="h-6 w-6" />
          </div>
          <div>
            <span className="text-[10px] font-bold text-gray-500 uppercase tracking-wider block">Invoices Scanned</span>
            <span className="text-2xl font-black text-white">{metrics?.totalBills || 0}</span>
          </div>
        </div>

        <div className="glass-panel p-6 rounded-2xl flex items-center gap-4">
          <div className="h-12 w-12 rounded-xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400">
            <Activity className="h-6 w-6" />
          </div>
          <div>
            <span className="text-[10px] font-bold text-gray-500 uppercase tracking-wider block">System Outlay</span>
            <span className="text-2xl font-black text-white">₹{(metrics?.totalExpenses || 0).toFixed(0)}</span>
          </div>
        </div>
      </div>

      <div className="grid lg:grid-cols-12 gap-8">
        {/* User Account Registry */}
        <div className="lg:col-span-7 glass-panel p-6 rounded-2xl space-y-4">
          <h3 className="font-extrabold text-white text-base">Account Registry</h3>
          <div className="overflow-x-auto text-xs">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-gray-900 text-gray-500 font-semibold">
                  <th className="pb-3">Name</th>
                  <th className="pb-3">Email</th>
                  <th className="pb-3 text-center">Bills</th>
                  <th className="pb-3 text-right">Spend</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-900/60">
                {users.map((u) => (
                  <tr key={u.id} className="text-gray-300 hover:text-white transition">
                    <td className="py-3 font-bold">{u.name}</td>
                    <td className="py-3 text-gray-400">{u.email}</td>
                    <td className="py-3 text-center text-emerald-400 font-extrabold">{u.bills_uploaded}</td>
                    <td className="py-3 text-right font-black text-white">₹{u.total_spent.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Security Logs auditing */}
        <div className="lg:col-span-5 glass-panel p-6 rounded-2xl space-y-4">
          <div className="flex justify-between items-center">
            <h3 className="font-extrabold text-white text-base">Audit Trace Logs</h3>
            <FileClock className="h-4.5 w-4.5 text-gray-500" />
          </div>

          <div className="space-y-3 max-h-[300px] overflow-y-auto pr-1">
            {logs.map((log) => (
              <div key={log.id} className="p-3.5 rounded-xl bg-gray-950 border border-gray-900 space-y-1 text-[10px]">
                <div className="flex justify-between font-bold">
                  <span className="text-emerald-400 uppercase tracking-wider">{log.action.replace('_', ' ')}</span>
                  <span className="text-gray-600">{new Date(log.created_at).toLocaleString()}</span>
                </div>
                <p className="text-gray-400 leading-relaxed font-semibold">{log.details}</p>
                <div className="flex justify-between text-gray-600 pt-1 border-t border-gray-900/40">
                  <span>User: {log.user_name || 'System Guest'}</span>
                  <span>IP: {log.ip_address || '127.0.0.1'}</span>
                </div>
              </div>
            ))}

            {!logs.length && (
              <div className="py-12 text-center text-gray-500">No logs located.</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
