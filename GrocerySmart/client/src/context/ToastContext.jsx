/* ─────────────────────────────────────────────────────────────
   Toast Context — High-end glassmorphic notification engine
   ───────────────────────────────────────────────────────────── */

import React, { createContext, useContext, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { CheckCircle2, AlertCircle, Info, AlertTriangle, X } from 'lucide-react';

const ToastContext = createContext(null);

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([]);

  const addToast = useCallback((message, type = 'info', duration = 4000) => {
    const id = Math.random().toString(36).substring(2, 9);
    setToasts((prev) => [...prev, { id, message, type, duration }]);

    setTimeout(() => {
      removeToast(id);
    }, duration);
  }, []);

  const removeToast = useCallback((id) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const getIcon = (type) => {
    switch (type) {
      case 'success':
        return <CheckCircle2 className="h-5 w-5 text-emerald-400" />;
      case 'error':
        return <AlertCircle className="h-5 w-5 text-rose-400" />;
      case 'warning':
        return <AlertTriangle className="h-5 w-5 text-amber-400" />;
      case 'info':
      default:
        return <Info className="h-5 w-5 text-sky-400" />;
    }
  };

  const getTypeStyles = (type) => {
    switch (type) {
      case 'success':
        return 'border-emerald-500/30 bg-emerald-950/40 text-emerald-100 shadow-emerald-950/20';
      case 'error':
        return 'border-rose-500/30 bg-rose-950/40 text-rose-100 shadow-rose-950/20';
      case 'warning':
        return 'border-amber-500/30 bg-amber-950/40 text-amber-100 shadow-amber-950/20';
      case 'info':
      default:
        return 'border-sky-500/30 bg-sky-950/40 text-sky-100 shadow-sky-950/20';
    }
  };

  return (
    <ToastContext.Provider value={{ addToast, removeToast }}>
      {children}
      
      {/* Toast Portal/Tray Container */}
      <div className="fixed bottom-6 right-6 z-[9999] flex flex-col gap-3 w-full max-w-sm pointer-events-none">
        <AnimatePresence>
          {toasts.map((t) => (
            <motion.div
              key={t.id}
              layout
              initial={{ opacity: 0, y: 50, scale: 0.9 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, scale: 0.85, transition: { duration: 0.2 } }}
              className={`pointer-events-auto flex items-start gap-3 p-4 rounded-2xl border backdrop-blur-md shadow-2xl transition-all duration-200 ${getTypeStyles(t.type)}`}
            >
              <div className="flex-shrink-0 mt-0.5">{getIcon(t.type)}</div>
              <div className="flex-1 text-sm font-medium pr-2 leading-relaxed">{t.message}</div>
              <button
                onClick={() => removeToast(t.id)}
                className="flex-shrink-0 text-gray-400 hover:text-white transition-colors duration-150 p-0.5 rounded-lg hover:bg-white/5"
              >
                <X className="h-4 w-4" />
              </button>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </ToastContext.Provider>
  );
}

export function useToast() {
  const context = useContext(ToastContext);
  if (!context) throw new Error('useToast must be wrapped in ToastProvider');
  return context;
}
