/* ─────────────────────────────────────────────────────────────
   Navigation System — Sidebar + Topbar Responsive Combo
   ───────────────────────────────────────────────────────────── */

import { useState } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { 
  LayoutDashboard, ScanLine, History, PieChart, 
  Wallet, Settings, LogOut, Menu, X, ShieldAlert 
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export default function Navigation() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [isOpen, setIsOpen] = useState(false);

  const navLinks = [
    { name: 'Dashboard', path: '/dashboard', icon: LayoutDashboard },
    { name: 'Bill Scanner', path: '/scanner', icon: ScanLine },
    { name: 'History', path: '/history', icon: History },
    { name: 'Analytics', path: '/analytics', icon: PieChart },
    { name: 'Budget Planner', path: '/budget', icon: Wallet },
    { name: 'Profile Settings', path: '/settings', icon: Settings },
  ];

  if (user?.role === 'admin') {
    navLinks.push({ name: 'Admin Control', path: '/admin', icon: ShieldAlert });
  }

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  return (
    <>
      {/* ── Desktop Sidebar ───────────────────────────────────── */}
      <aside className="hidden lg:flex flex-col w-64 glass-panel border-r border-gray-800/80 fixed top-0 bottom-0 left-0 z-20">
        {/* Brand Logo */}
        <div className="flex items-center gap-3 px-6 h-20 border-b border-gray-900">
          <div className="bg-emerald-500/10 p-2 rounded-xl border border-emerald-500/20">
            <span className="text-2xl">🥬</span>
          </div>
          <div>
            <h1 className="font-extrabold text-lg bg-gradient-to-r from-emerald-400 to-teal-300 bg-clip-text text-transparent">
              GrocerySmart
            </h1>
            <span className="text-xs text-gray-500 font-semibold tracking-wider uppercase">AI expense v1</span>
          </div>
        </div>

        {/* Links Navigation */}
        <nav className="flex-1 px-4 py-6 space-y-1.5 overflow-y-auto">
          {navLinks.map((link) => (
            <NavLink
              key={link.name}
              to={link.path}
              className={({ isActive }) =>
                `flex items-center gap-3.5 px-4 py-3 rounded-xl transition-all duration-200 group text-sm font-medium ${
                  isActive
                    ? 'bg-gradient-to-r from-emerald-500/20 to-teal-500/10 text-emerald-400 border border-emerald-500/20'
                    : 'text-gray-400 hover:text-white hover:bg-gray-900/60 border border-transparent'
                }`
              }
            >
              {({ isActive }) => (
                <>
                  <link.icon className={`h-5 w-5 transition-transform duration-200 group-hover:scale-105 ${isActive ? 'text-emerald-400' : 'text-gray-400 group-hover:text-white'}`} />
                  {link.name}
                </>
              )}
            </NavLink>
          ))}
        </nav>

        {/* User Footplate */}
        <div className="p-4 border-t border-gray-900/80">
          <div className="flex items-center justify-between p-3 rounded-xl bg-gray-950/40 border border-gray-900 mb-2">
            <div className="flex items-center gap-3 min-w-0">
              <div className="h-9 w-9 rounded-full bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center font-bold text-emerald-400 uppercase">
                {user?.name?.[0] || 'U'}
              </div>
              <div className="min-w-0">
                <p className="text-xs font-semibold text-white truncate">{user?.name}</p>
                <p className="text-[10px] text-gray-500 truncate">{user?.email}</p>
              </div>
            </div>
          </div>
          <button
            onClick={handleLogout}
            className="flex items-center gap-3 w-full px-4 py-2.5 rounded-xl text-sm font-semibold text-rose-400 hover:bg-rose-500/10 border border-transparent hover:border-rose-500/20 transition-all duration-200"
          >
            <LogOut className="h-4 w-4" />
            Sign Out
          </button>
        </div>
      </aside>

      {/* ── Mobile Topbar ────────────────────────────────────── */}
      <header className="lg:hidden flex items-center justify-between px-6 h-16 glass-panel border-b border-gray-900 fixed top-0 left-0 right-0 z-30">
        <div className="flex items-center gap-2">
          <span className="text-xl">🥬</span>
          <span className="font-extrabold text-base bg-gradient-to-r from-emerald-400 to-teal-300 bg-clip-text text-transparent">
            GrocerySmart
          </span>
        </div>
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="p-1.5 rounded-lg bg-gray-900 border border-gray-800 text-gray-400 hover:text-white"
        >
          {isOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </header>

      {/* Mobile Drawer Menu */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.2 }}
            className="lg:hidden fixed top-16 left-0 right-0 z-20 glass-panel border-b border-gray-900 flex flex-col p-6 space-y-4 max-h-[85vh] overflow-y-auto"
          >
            <nav className="flex flex-col space-y-2">
              {navLinks.map((link) => (
                <NavLink
                  key={link.name}
                  to={link.path}
                  onClick={() => setIsOpen(false)}
                  className={({ isActive }) =>
                    `flex items-center gap-3.5 px-4 py-3 rounded-xl transition-all duration-200 text-sm font-semibold ${
                      isActive
                        ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/10'
                        : 'text-gray-400 hover:text-white hover:bg-gray-900/60 border border-transparent'
                    }`
                  }
                >
                  <link.icon className="h-5 w-5" />
                  {link.name}
                </NavLink>
              ))}
            </nav>

            <div className="border-t border-gray-800 pt-4">
              <div className="flex items-center gap-3 mb-4">
                <div className="h-10 w-10 rounded-full bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center font-bold text-emerald-400 uppercase">
                  {user?.name?.[0] || 'U'}
                </div>
                <div>
                  <p className="text-sm font-semibold text-white">{user?.name}</p>
                  <p className="text-xs text-gray-500">{user?.email}</p>
                </div>
              </div>
              <button
                onClick={handleLogout}
                className="flex items-center gap-3 w-full px-4 py-3 rounded-xl text-sm font-semibold text-rose-400 bg-rose-500/5 hover:bg-rose-500/10 border border-rose-500/20"
              >
                <LogOut className="h-4 w-4" />
                Sign Out
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
