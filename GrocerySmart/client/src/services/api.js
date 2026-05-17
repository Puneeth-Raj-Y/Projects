/* ─────────────────────────────────────────────────────────────
   Centralized API Client Service
   ───────────────────────────────────────────────────────────── */

import axios from 'axios';

// Resolve backend API base URL for development and production environments
const API_URL = import.meta.env.VITE_API_URL || '';

const apiClient = axios.create({
  baseURL: API_URL,
  timeout: 60000, // 60 seconds (generous for OCR file processing)
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request Interceptor: Attach JWT Token dynamically from local storage
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('gs_token');
    if (token) {
      config.headers['Authorization'] = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response Interceptor: Format error responses uniformly
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    const customError = {
      message: error.response?.data?.error || error.message || 'An unexpected error occurred',
      status: error.response?.status || 500,
      data: error.response?.data || null,
    };
    return Promise.reject(customError);
  }
);

export default apiClient;
