/* ─────────────────────────────────────────────────────────────
   Profile Settings — User details adjustments & Currency preferences
   ───────────────────────────────────────────────────────────── */

import { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { Settings, Save, KeyRound, CheckCircle2 } from 'lucide-react';
import { motion } from 'framer-motion';

export default function ProfileSettings() {
  const { user, updateProfile } = useAuth();
  const [formData, setFormData] = useState({
    name: user?.name || '',
    phone: user?.phone || '',
    currency: user?.currency || 'INR',
    theme: user?.theme || 'dark',
    password: '',
    confirmPassword: ''
  });
  const [success, setSuccess] = useState('');
  const [error, setError] = useState('');

  const handleInputChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
    setSuccess('');
    setError('');
  };

  const handleProfileSubmit = async (e) => {
    e.preventDefault();
    if (formData.password && formData.password !== formData.confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    try {
      await updateProfile({
        name: formData.name,
        phone: formData.phone,
        currency: formData.currency,
        theme: formData.theme,
        ...(formData.password && { password: formData.password })
      });
      setSuccess('Profile specifications saved securely!');
      setFormData(prev => ({ ...prev, password: '', confirmPassword: '' }));
    } catch (err) {
      setError('Could not update profile coordinates.');
    }
  };

  return (
    <div className="space-y-8 pb-12 max-w-2xl mx-auto">
      <div className="flex items-center gap-3">
        <Settings className="h-8 w-8 text-emerald-400" />
        <div>
          <h2 className="text-2xl md:text-3xl font-extrabold text-white">Profile Settings</h2>
          <p className="text-sm text-gray-500 font-medium">Update account information, currency standards, and secure credentials.</p>
        </div>
      </div>

      {/* Save Success Box */}
      {success && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center gap-3 p-4 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 rounded-2xl text-xs font-semibold"
        >
          <CheckCircle2 className="h-5 w-5 shrink-0" />
          <span>{success}</span>
        </motion.div>
      )}

      {error && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center gap-3 p-4 bg-rose-500/10 border border-rose-500/20 text-rose-400 rounded-2xl text-xs font-semibold"
        >
          <CheckCircle2 className="h-5 w-5 shrink-0 text-rose-500" />
          <span>{error}</span>
        </motion.div>
      )}

      {/* Main glass frame */}
      <div className="glass-panel p-6 rounded-3xl bg-gray-950/40">
        <form onSubmit={handleProfileSubmit} className="space-y-6 text-xs font-bold text-gray-400 uppercase tracking-wider">
          
          <h3 className="text-white text-sm font-extrabold border-b border-gray-900 pb-3 mb-4">Account coordinates</h3>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-1.5">
              <label>Full Name</label>
              <input 
                type="text" 
                name="name"
                value={formData.name}
                onChange={handleInputChange}
                className="w-full px-4 glass-input" 
                required
              />
            </div>
            <div className="space-y-1.5">
              <label>Email Address (Disabled)</label>
              <input 
                type="email" 
                value={user?.email} 
                disabled
                className="w-full px-4 glass-input opacity-40 cursor-not-allowed bg-gray-950" 
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-1.5">
              <label>Phone Number</label>
              <input 
                type="text" 
                name="phone"
                value={formData.phone}
                onChange={handleInputChange}
                placeholder="e.g. 9876543210"
                className="w-full px-4 glass-input" 
              />
            </div>
            <div className="space-y-1.5">
              <label>Standard Currency</label>
              <select
                name="currency"
                value={formData.currency}
                onChange={handleInputChange}
                className="w-full px-4 glass-input appearance-none"
              >
                <option value="INR">INR (₹)</option>
                <option value="USD">USD ($)</option>
              </select>
            </div>
          </div>

          <h3 className="text-white text-sm font-extrabold border-b border-gray-900 pb-3 pt-4 mb-4 flex items-center gap-2">
            <KeyRound className="h-4.5 w-4.5 text-gray-500" />
            Security Passwords
          </h3>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-1.5">
              <label>New Password</label>
              <input 
                type="password" 
                name="password"
                value={formData.password}
                onChange={handleInputChange}
                placeholder="••••••••"
                className="w-full px-4 glass-input" 
              />
            </div>
            <div className="space-y-1.5">
              <label>Verify Password</label>
              <input 
                type="password" 
                name="confirmPassword"
                value={formData.confirmPassword}
                onChange={handleInputChange}
                placeholder="••••••••"
                className="w-full px-4 glass-input" 
              />
            </div>
          </div>

          <button
            type="submit"
            className="w-full py-4 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-black font-extrabold flex items-center justify-center gap-2 shadow-lg shadow-emerald-500/10 transition mt-6"
          >
            <Save className="h-5 w-5" />
            Save Profile coordinates
          </button>
        </form>
      </div>
    </div>
  );
}
