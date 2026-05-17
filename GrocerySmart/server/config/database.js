/* ─────────────────────────────────────────────────────────────
   SQLite Database — Schema & Connection
   ───────────────────────────────────────────────────────────── */

import Database from 'better-sqlite3';
import path from 'path';
import { fileURLToPath } from 'url';
import { v4 as uuid } from 'uuid';
import bcrypt from 'bcryptjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const DB_PATH = path.join(__dirname, '..', 'grocerysmart.db');
let db;

export function getDb() {
  if (!db) {
    db = new Database(DB_PATH);
    db.pragma('journal_mode = WAL');
    db.pragma('foreign_keys = ON');
  }
  return db;
}

/* ── Schema ─────────────────────────────────────────────────── */

export function initializeDatabase() {
  const conn = getDb();

  conn.exec(`
    /* ── Users ──────────────────────────────────────────────── */
    CREATE TABLE IF NOT EXISTS users (
      id            TEXT PRIMARY KEY,
      name          TEXT NOT NULL,
      email         TEXT NOT NULL UNIQUE,
      password      TEXT NOT NULL,
      role          TEXT NOT NULL DEFAULT 'user',
      avatar        TEXT,
      phone         TEXT,
      currency      TEXT DEFAULT 'INR',
      theme         TEXT DEFAULT 'dark',
      created_at    DATETIME DEFAULT (datetime('now')),
      updated_at    DATETIME DEFAULT (datetime('now'))
    );

    /* ── Categories ─────────────────────────────────────────── */
    CREATE TABLE IF NOT EXISTS categories (
      id            TEXT PRIMARY KEY,
      name          TEXT NOT NULL UNIQUE,
      icon          TEXT,
      color         TEXT,
      created_at    DATETIME DEFAULT (datetime('now'))
    );

    /* ── Bills ──────────────────────────────────────────────── */
    CREATE TABLE IF NOT EXISTS bills (
      id              TEXT PRIMARY KEY,
      user_id         TEXT NOT NULL,
      store_name      TEXT,
      bill_date       DATE,
      total_amount    REAL DEFAULT 0,
      tax_amount      REAL DEFAULT 0,
      discount_amount REAL DEFAULT 0,
      image_path      TEXT,
      raw_text        TEXT,
      barcode         TEXT,
      status          TEXT DEFAULT 'processed',
      duplicate_hash  TEXT,
      language        TEXT DEFAULT 'en',
      created_at      DATETIME DEFAULT (datetime('now')),
      updated_at      DATETIME DEFAULT (datetime('now')),
      FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );

    /* ── Bill Items ─────────────────────────────────────────── */
    CREATE TABLE IF NOT EXISTS bill_items (
      id            TEXT PRIMARY KEY,
      bill_id       TEXT NOT NULL,
      category_id   TEXT,
      name          TEXT NOT NULL,
      quantity      REAL DEFAULT 1,
      unit          TEXT,
      price         REAL NOT NULL,
      total_price   REAL NOT NULL,
      created_at    DATETIME DEFAULT (datetime('now')),
      FOREIGN KEY (bill_id) REFERENCES bills(id) ON DELETE CASCADE,
      FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE SET NULL
    );

    /* ── Expenses (manual entries) ──────────────────────────── */
    CREATE TABLE IF NOT EXISTS expenses (
      id            TEXT PRIMARY KEY,
      user_id       TEXT NOT NULL,
      bill_id       TEXT,
      category_id   TEXT,
      description   TEXT,
      amount        REAL NOT NULL,
      expense_date  DATE NOT NULL,
      created_at    DATETIME DEFAULT (datetime('now')),
      updated_at    DATETIME DEFAULT (datetime('now')),
      FOREIGN KEY (user_id)     REFERENCES users(id)      ON DELETE CASCADE,
      FOREIGN KEY (bill_id)     REFERENCES bills(id)       ON DELETE SET NULL,
      FOREIGN KEY (category_id) REFERENCES categories(id)  ON DELETE SET NULL
    );

    /* ── Budgets ────────────────────────────────────────────── */
    CREATE TABLE IF NOT EXISTS budgets (
      id            TEXT PRIMARY KEY,
      user_id       TEXT NOT NULL,
      category_id   TEXT,
      amount        REAL NOT NULL,
      spent         REAL DEFAULT 0,
      period        TEXT DEFAULT 'monthly',
      month         INTEGER,
      year          INTEGER,
      created_at    DATETIME DEFAULT (datetime('now')),
      updated_at    DATETIME DEFAULT (datetime('now')),
      FOREIGN KEY (user_id)     REFERENCES users(id)      ON DELETE CASCADE,
      FOREIGN KEY (category_id) REFERENCES categories(id)  ON DELETE SET NULL
    );

    /* ── Analytics Logs ─────────────────────────────────────── */
    CREATE TABLE IF NOT EXISTS analytics_logs (
      id            TEXT PRIMARY KEY,
      user_id       TEXT,
      action        TEXT NOT NULL,
      details       TEXT,
      ip_address    TEXT,
      created_at    DATETIME DEFAULT (datetime('now'))
    );

    /* ── Performance Indexes ────────────────────────────────── */
    CREATE INDEX IF NOT EXISTS idx_bills_user        ON bills(user_id);
    CREATE INDEX IF NOT EXISTS idx_bills_date        ON bills(bill_date);
    CREATE INDEX IF NOT EXISTS idx_bills_store       ON bills(store_name);
    CREATE INDEX IF NOT EXISTS idx_bills_hash        ON bills(duplicate_hash);
    CREATE INDEX IF NOT EXISTS idx_bill_items_bill   ON bill_items(bill_id);
    CREATE INDEX IF NOT EXISTS idx_bill_items_cat    ON bill_items(category_id);
    CREATE INDEX IF NOT EXISTS idx_expenses_user     ON expenses(user_id);
    CREATE INDEX IF NOT EXISTS idx_expenses_date     ON expenses(expense_date);
    CREATE INDEX IF NOT EXISTS idx_expenses_cat      ON expenses(category_id);
    CREATE INDEX IF NOT EXISTS idx_budgets_user      ON budgets(user_id);
    CREATE INDEX IF NOT EXISTS idx_budgets_period    ON budgets(user_id, month, year);
    CREATE INDEX IF NOT EXISTS idx_analytics_user    ON analytics_logs(user_id);
    CREATE INDEX IF NOT EXISTS idx_analytics_action  ON analytics_logs(action);
  `);

  // Seed default categories
  seedCategories(conn);
  // Seed demo admin account
  seedAdmin(conn);
}

/* ── Seed helpers ───────────────────────────────────────────── */

function seedCategories(conn) {
  const existing = conn.prepare('SELECT COUNT(*) as count FROM categories').get();
  if (existing.count > 0) return;

  const categories = [
    { name: 'Vegetables',     icon: '🥬', color: '#22c55e' },
    { name: 'Fruits',         icon: '🍎', color: '#ef4444' },
    { name: 'Dairy',          icon: '🥛', color: '#3b82f6' },
    { name: 'Snacks',         icon: '🍿', color: '#f59e0b' },
    { name: 'Beverages',      icon: '🥤', color: '#8b5cf6' },
    { name: 'Household',      icon: '🏠', color: '#06b6d4' },
    { name: 'Personal Care',  icon: '🧴', color: '#ec4899' },
    { name: 'Medicines',      icon: '💊', color: '#14b8a6' },
    { name: 'Grains & Pulses', icon: '🌾', color: '#d97706' },
    { name: 'Meat & Seafood', icon: '🥩', color: '#dc2626' },
    { name: 'Bakery',         icon: '🍞', color: '#a16207' },
    { name: 'Frozen Foods',   icon: '🧊', color: '#0ea5e9' },
    { name: 'Others',         icon: '📦', color: '#6b7280' },
  ];

  const insert = conn.prepare(
    'INSERT INTO categories (id, name, icon, color) VALUES (?, ?, ?, ?)'
  );

  const tx = conn.transaction(() => {
    for (const c of categories) {
      insert.run(uuid(), c.name, c.icon, c.color);
    }
  });
  tx();
}

function seedAdmin(conn) {
  const existing = conn.prepare("SELECT COUNT(*) as count FROM users WHERE role = 'admin'").get();
  if (existing.count > 0) return;

  const hash = bcrypt.hashSync('admin123', 12);
  conn.prepare(
    'INSERT INTO users (id, name, email, password, role) VALUES (?, ?, ?, ?, ?)'
  ).run(uuid(), 'Admin', 'admin@grocerysmart.com', hash, 'admin');
}

export default getDb;
