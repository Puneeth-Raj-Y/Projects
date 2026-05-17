/* ─────────────────────────────────────────────────────────────
   Categories Controller
   ───────────────────────────────────────────────────────────── */

import { v4 as uuid } from 'uuid';
import { getDb } from '../config/database.js';

/**
 * Fetch all categories
 */
export function getCategories(req, res, next) {
  const db = getDb();
  try {
    const categories = db.prepare('SELECT * FROM categories ORDER BY name ASC').all();
    res.json(categories);
  } catch (err) {
    next(err);
  }
}

/**
 * Create a custom category
 */
export function createCategory(req, res, next) {
  const { name, icon, color } = req.body;
  const db = getDb();

  try {
    const existing = db.prepare('SELECT id FROM categories WHERE name = ?').get(name);
    if (existing) {
      return res.status(400).json({ error: 'Category name already exists' });
    }

    const id = uuid();
    db.prepare(`
      INSERT INTO categories (id, name, icon, color)
      VALUES (?, ?, ?, ?)
    `).run(id, name, icon || '📦', color || '#6b7280');

    res.status(201).json({ message: 'Category created successfully', id });
  } catch (err) {
    next(err);
  }
}
