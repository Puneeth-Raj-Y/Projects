/* ─────────────────────────────────────────────────────────────
   App Layout — Main Layout Wrapper
   ───────────────────────────────────────────────────────────── */

import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import Navigation from '../components/Navigation';

// Pages
import Dashboard from '../pages/Dashboard';
import BillScanner from '../pages/BillScanner';
import ExpenseHistory from '../pages/ExpenseHistory';
import Analytics from '../pages/Analytics';
import BudgetPlanner from '../pages/BudgetPlanner';
import Admin from '../pages/Admin';
import ProfileSettings from '../pages/Settings';

import { useAuth } from '../context/AuthContext';

/**
 * Route guard restricting pages to logged in users
 */
function ProtectedRoute({ children }) {
  const { isAuthenticated, loading } = useAuth();
  
  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#030712]">
        <div className="h-10 w-10 border-4 border-emerald-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return isAuthenticated ? children : <Navigate to="/login" replace />;
}

/**
 * Route guard restricting pages to admin role only
 */
function AdminRoute({ children }) {
  const { user, isAuthenticated, loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#030712]">
        <div className="h-10 w-10 border-4 border-emerald-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return isAuthenticated && user?.role === 'admin' ? children : <Navigate to="/dashboard" replace />;
}

export default function AppLayout() {
  return (
    <div className="min-h-screen bg-[#030712] relative overflow-hidden flex w-full">
      {/* Dynamic background mesh grids */}
      <div className="glowing-mesh w-[500px] h-[500px] bg-emerald-500/5 top-[-10%] left-[-10%] pulse-glow" />
      <div className="glowing-mesh w-[600px] h-[600px] bg-teal-500/5 bottom-[-20%] right-[-10%] pulse-glow" style={{ animationDelay: '3s' }} />

      <Navigation />

      <main className="flex-1 lg:pl-64 pt-16 lg:pt-0 min-w-0 transition-all duration-200 w-full">
        <div className="max-w-7xl mx-auto px-6 md:px-12 py-8 relative z-10">
          <Routes>
            <Route path="/dashboard" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
            <Route path="/scanner" element={<ProtectedRoute><BillScanner /></ProtectedRoute>} />
            <Route path="/history" element={<ProtectedRoute><ExpenseHistory /></ProtectedRoute>} />
            <Route path="/analytics" element={<ProtectedRoute><Analytics /></ProtectedRoute>} />
            <Route path="/budget" element={<ProtectedRoute><BudgetPlanner /></ProtectedRoute>} />
            <Route path="/admin" element={<AdminRoute><Admin /></AdminRoute>} />
            <Route path="/settings" element={<ProtectedRoute><ProfileSettings /></ProtectedRoute>} />
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
          </Routes>
        </div>
      </main>
    </div>
  );
}
