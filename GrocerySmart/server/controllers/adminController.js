/* ─────────────────────────────────────────────────────────────
   Admin Controller
   ───────────────────────────────────────────────────────────── */

import { getDb } from '../config/database.js';

/**
 * Fetch lists of all registered users
 */
export function getUsers(req, res, next) {
  const db = getDb();
  try {
    const users = db.prepare(`
      SELECT id, name, email, role, phone, currency, created_at,
             (SELECT COUNT(*) FROM bills WHERE user_id = users.id) as bills_uploaded,
             (SELECT COALESCE(SUM(amount), 0) FROM expenses WHERE user_id = users.id) as total_spent
      FROM users
      ORDER BY created_at DESC
    `).all();

    res.json(users);
  } catch (err) {
    next(err);
  }
}

/**
 * System interaction and security audit log
 */
export function getLogs(req, res, next) {
  const db = getDb();
  try {
    const logs = db.prepare(`
      SELECT al.*, u.name as user_name, u.email as user_email
      FROM analytics_logs al
      LEFT JOIN users u ON al.user_id = u.id
      ORDER BY al.created_at DESC
      LIMIT 100
    `).all();

    res.json(logs);
  } catch (err) {
    next(err);
  }
}

/**
 * System metrics dashboard overview
 */
export function getMetrics(req, res, next) {
  const db = getDb();
  try {
    const totalUsers = db.prepare('SELECT COUNT(*) as count FROM users').get().count;
    const totalBills = db.prepare('SELECT COUNT(*) as count FROM bills').get().count;
    const totalExpenses = db.prepare('SELECT COALESCE(SUM(amount), 0) as total FROM expenses').get().total;

    res.json({
      totalUsers,
      totalBills,
      totalExpenses
    });
  } catch (err) {
    next(err);
  }
}
