/* ─────────────────────────────────────────────────────────────
   Budgets Controller
   ───────────────────────────────────────────────────────────── */

import { v4 as uuid } from 'uuid';
import { getDb } from '../config/database.js';

/**
 * Get user budgets with current spending calculated in real-time
 */
export function getBudgets(req, res, next) {
  const db = getDb();
  const currentMonth = new Date().getMonth() + 1;
  const currentYear = new Date().getFullYear();

  try {
    const budgets = db.prepare(`
      SELECT b.*, c.name as category_name, c.color as category_color, c.icon as category_icon,
             COALESCE((
               SELECT SUM(e.amount)
               FROM expenses e
               WHERE e.user_id = b.user_id 
                 AND e.category_id = b.category_id 
                 AND strftime('%m', e.expense_date) = ?
                 AND strftime('%Y', e.expense_date) = ?
             ), 0) as spent
      FROM budgets b
      LEFT JOIN categories c ON b.category_id = c.id
      WHERE b.user_id = ? AND b.month = ? AND b.year = ?
    `).all(
      String(currentMonth).padStart(2, '0'),
      String(currentYear),
      req.user.id,
      currentMonth,
      currentYear
    );

    res.json(budgets);
  } catch (err) {
    next(err);
  }
}

/**
 * Set or update budget for a category
 */
export function setBudget(req, res, next) {
  const { category_id, amount } = req.body;
  const db = getDb();
  const currentMonth = new Date().getMonth() + 1;
  const currentYear = new Date().getFullYear();

  try {
    // Check if budget is already defined for this category & month
    const existing = db.prepare(`
      SELECT id FROM budgets 
      WHERE user_id = ? AND category_id = ? AND month = ? AND year = ?
    `).get(req.user.id, category_id, currentMonth, currentYear);

    if (existing) {
      db.prepare(`
        UPDATE budgets 
        SET amount = ?, updated_at = datetime('now')
        WHERE id = ?
      `).run(parseFloat(amount), existing.id);

      return res.json({ message: 'Budget updated successfully' });
    }

    const id = uuid();
    db.prepare(`
      INSERT INTO budgets (id, user_id, category_id, amount, month, year)
      VALUES (?, ?, ?, ?, ?, ?)
    `).run(id, req.user.id, category_id, parseFloat(amount), currentMonth, currentYear);

    res.status(201).json({ message: 'Budget set successfully', id });
  } catch (err) {
    next(err);
  }
}

/**
 * Delete custom budget plan
 */
export function deleteBudget(req, res, next) {
  const db = getDb();

  try {
    const budget = db.prepare('SELECT id FROM budgets WHERE id = ? AND user_id = ?').get(req.params.id, req.user.id);
    if (!budget) {
      return res.status(404).json({ error: 'Budget plan not found' });
    }

    db.prepare('DELETE FROM budgets WHERE id = ?').run(req.params.id);
    res.json({ message: 'Budget plan removed' });
  } catch (err) {
    next(err);
  }
}
