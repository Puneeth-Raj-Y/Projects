/* ─────────────────────────────────────────────────────────────
   Error Boundary Component — Exception Catching Framework
   ───────────────────────────────────────────────────────────── */

import React, { Component } from 'react';
import { AlertOctagon, RotateCcw, Home } from 'lucide-react';

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('[ErrorBoundary caught an error]', error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-[#030712] flex items-center justify-center p-6 relative overflow-hidden">
          {/* Dynamic background mesh grids */}
          <div className="glowing-mesh w-[400px] h-[400px] bg-rose-500/5 top-[-10%] left-[-10%] pulse-glow" />
          <div className="glowing-mesh w-[400px] h-[400px] bg-rose-600/5 bottom-[-10%] right-[-10%] pulse-glow" style={{ animationDelay: '2s' }} />

          <div className="max-w-md w-full glass-panel border border-rose-500/20 p-8 rounded-3xl text-center shadow-2xl relative z-10">
            <div className="mx-auto w-16 h-16 rounded-2xl bg-rose-500/10 border border-rose-500/20 flex items-center justify-center text-rose-500 mb-6 animate-pulse">
              <AlertOctagon className="h-8 w-8" />
            </div>

            <h1 className="text-2xl font-extrabold text-white mb-2 tracking-tight">Something went wrong</h1>
            <p className="text-gray-400 text-sm mb-6 leading-relaxed">
              The application encountered an unexpected error. Don't worry, your grocery data is secure.
            </p>

            {this.state.error && (
              <div className="text-left bg-black/40 border border-gray-900 rounded-xl p-4 mb-6 max-h-36 overflow-auto font-mono text-[10px] text-rose-300 leading-normal select-all">
                {this.state.error.toString()}
              </div>
            )}

            <div className="flex gap-4">
              <button
                onClick={this.handleReset}
                className="flex-1 flex items-center justify-center gap-2 bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-400 hover:to-teal-500 text-white font-semibold py-3 px-4 rounded-xl shadow-lg shadow-emerald-950/20 hover:shadow-emerald-950/30 transition-all duration-200"
              >
                <RotateCcw className="h-4 w-4" />
                Retry App
              </button>
              <a
                href="/"
                className="flex-1 flex items-center justify-center gap-2 bg-gray-900 hover:bg-gray-800 border border-gray-800 text-gray-300 hover:text-white font-semibold py-3 px-4 rounded-xl transition-all duration-200"
              >
                <Home className="h-4 w-4" />
                Go Home
              </a>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
