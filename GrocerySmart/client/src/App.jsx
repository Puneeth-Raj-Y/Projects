/* ─────────────────────────────────────────────────────────────
   Main App — Client React router and layout framework
   ───────────────────────────────────────────────────────────── */

import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import { ToastProvider } from './context/ToastContext';
import ErrorBoundary from './components/ErrorBoundary';

// Pages
import Landing from './pages/Landing';
import Auth from './pages/Auth';

// Layout
import AppLayout from './layouts/AppLayout';

export default function App() {
  return (
    <ErrorBoundary>
      <Router>
        <ToastProvider>
          <AuthProvider>
            <Routes>
              <Route path="/" element={<Landing />} />
              <Route path="/login" element={<Auth defaultTab="login" />} />
              <Route path="/register" element={<Auth defaultTab="register" />} />
              <Route path="/*" element={<AppLayout />} />
            </Routes>
          </AuthProvider>
        </ToastProvider>
      </Router>
    </ErrorBoundary>
  );
}
