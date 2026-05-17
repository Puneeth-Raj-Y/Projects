/* ─────────────────────────────────────────────────────────────
   GrocerySmart — Express Server Entry Point
   ───────────────────────────────────────────────────────────── */

import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import rateLimit from 'express-rate-limit';
import path from 'path';
import { fileURLToPath } from 'url';

import { initializeDatabase } from './config/database.js';
import { errorHandler, notFound } from './middleware/errorHandler.js';

// Routes
import authRoutes from './routes/auth.js';
import billRoutes from './routes/bills.js';
import expenseRoutes from './routes/expenses.js';
import categoryRoutes from './routes/categories.js';
import analyticsRoutes from './routes/analytics.js';
import budgetRoutes from './routes/budgets.js';
import adminRoutes from './routes/admin.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 5000;

/* ── Global Middleware ──────────────────────────────────────── */

app.use(helmet({ crossOriginResourcePolicy: { policy: 'cross-origin' } }));
app.use(cors({
  origin: process.env.CLIENT_URL || 'http://localhost:5173',
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH'],
}));
app.use(morgan('dev'));
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Rate limiting
const limiter = rateLimit({
  windowMs: parseInt(process.env.RATE_LIMIT_WINDOW_MS) || 15 * 60 * 1000,
  max: parseInt(process.env.RATE_LIMIT_MAX) || 100,
  message: { error: 'Too many requests, please try again later.' },
  standardHeaders: true,
  legacyHeaders: false,
});
app.use('/api/', limiter);

// Static files — uploaded bill images
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

/* ── API Routes ─────────────────────────────────────────────── */

app.use('/api/auth', authRoutes);
app.use('/api/bills', billRoutes);
app.use('/api/expenses', expenseRoutes);
app.use('/api/categories', categoryRoutes);
app.use('/api/analytics', analyticsRoutes);
app.use('/api/budgets', budgetRoutes);
app.use('/api/admin', adminRoutes);

// Health check
app.get('/api/health', (_req, res) => {
  res.json({ status: 'ok', uptime: process.uptime(), timestamp: new Date().toISOString() });
});

/* ── Serve React build in production ────────────────────────── */

if (process.env.NODE_ENV === 'production') {
  app.use(express.static(path.join(__dirname, '../client/dist')));
  app.get('*', (_req, res) => {
    res.sendFile(path.join(__dirname, '../client/dist/index.html'));
  });
}

/* ── Error Handling ─────────────────────────────────────────── */

app.use(notFound);
app.use(errorHandler);

/* ── Boot ───────────────────────────────────────────────────── */

async function start() {
  try {
    initializeDatabase();
    console.log('✅  Database initialized');

    app.listen(PORT, () => {
      console.log(`🚀  GrocerySmart API running on http://localhost:${PORT}`);
      console.log(`📦  Environment: ${process.env.NODE_ENV || 'development'}`);
    });
  } catch (err) {
    console.error('❌  Failed to start server:', err);
    process.exit(1);
  }
}

start();

export default app;
