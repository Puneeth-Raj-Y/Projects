/* ─────────────────────────────────────────────────────────────
   Auth Context — Centralized authentication state management
   ───────────────────────────────────────────────────────────── */

import { createContext, useContext, useState, useEffect } from 'react';
import apiClient from '../services/api';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function checkUser() {
      const token = localStorage.getItem('gs_token');
      if (!token) {
        setLoading(false);
        return;
      }
      try {
        const response = await apiClient.get('/api/auth/me');
        setUser(response.data.user);
      } catch (err) {
        console.error('Session restore failed:', err.message);
        logout();
      } finally {
        setLoading(false);
      }
    }
    checkUser();
  }, []);

  const login = async (email, password) => {
    try {
      const response = await apiClient.post('/api/auth/login', { email, password });
      const { token: newToken, user: newUser } = response.data;
      localStorage.setItem('gs_token', newToken);
      setUser(newUser);
      return newUser;
    } catch (err) {
      throw err;
    }
  };

  const register = async (name, email, password) => {
    try {
      const response = await apiClient.post('/api/auth/register', { name, email, password });
      const { token: newToken, user: newUser } = response.data;
      localStorage.setItem('gs_token', newToken);
      setUser(newUser);
      return newUser;
    } catch (err) {
      throw err;
    }
  };

  const updateProfile = async (profileData) => {
    try {
      const response = await apiClient.put('/api/auth/profile', profileData);
      setUser(response.data.user);
      return response.data.user;
    } catch (err) {
      throw err;
    }
  };

  const logout = () => {
    localStorage.removeItem('gs_token');
    setUser(null);
  };

  const val = {
    user,
    loading,
    login,
    register,
    updateProfile,
    logout,
    isAuthenticated: !!user
  };

  return <AuthContext.Provider value={val}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) throw new Error('useAuth must be wrapped in AuthProvider');
  return context;
}
