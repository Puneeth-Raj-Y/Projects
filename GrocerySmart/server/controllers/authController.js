/* ─────────────────────────────────────────────────────────────
   Auth Controller
   ───────────────────────────────────────────────────────────── */

import bcrypt from 'bcryptjs';
import { v4 as uuid } from 'uuid';
import { getDb } from '../config/database.js';
import { generateToken } from '../middleware/auth.js';

/**
 * Register a new user
 */
export async function register(req, res, next) {
  const { name, email, password } = req.body;
  const db = getDb();

  try {
    // Check if user already exists
    const existing = db.prepare('SELECT id FROM users WHERE email = ?').get(email);
    if (existing) {
      return res.status(400).json({ error: 'Email already registered' });
    }

    const id = uuid();
    const hash = await bcrypt.hash(password, 12);

    db.prepare(
      'INSERT INTO users (id, name, email, password) VALUES (?, ?, ?, ?)'
    ).run(id, name, email.toLowerCase(), hash);

    // Write action into system log
    db.prepare(
      'INSERT INTO analytics_logs (id, user_id, action, details) VALUES (?, ?, ?, ?)'
    ).run(uuid(), id, 'user_registration', `User registered: ${email}`);

    const user = { id, name, email: email.toLowerCase(), role: 'user', avatar: null, currency: 'INR', theme: 'dark' };
    const token = generateToken(id);

    res.status(201).json({ user, token });
  } catch (err) {
    next(err);
  }
}

/**
 * Authenticate user & get token
 */
export async function login(req, res, next) {
  const { email, password } = req.body;
  const db = getDb();

  try {
    const user = db.prepare('SELECT * FROM users WHERE email = ?').get(email.toLowerCase());
    if (!user) {
      return res.status(400).json({ error: 'Invalid email or password' });
    }

    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
      return res.status(400).json({ error: 'Invalid email or password' });
    }

    const token = generateToken(user.id);

    // Track active login sessions
    db.prepare(
      'INSERT INTO analytics_logs (id, user_id, action, details, ip_address) VALUES (?, ?, ?, ?, ?)'
    ).run(uuid(), user.id, 'user_login', `Logged in successfully`, req.ip);

    const safeUser = {
      id: user.id,
      name: user.name,
      email: user.email,
      role: user.role,
      avatar: user.avatar,
      currency: user.currency,
      theme: user.theme
    };

    res.json({ user: safeUser, token });
  } catch (err) {
    next(err);
  }
}

/**
 * Get current user profile
 */
export function getMe(req, res) {
  res.json({ user: req.user });
}

/**
 * Update profile settings
 */
export async function updateProfile(req, res, next) {
  const { name, phone, currency, theme, password } = req.body;
  const db = getDb();

  try {
    let updateQuery = 'UPDATE users SET name = ?, phone = ?, currency = ?, theme = ?';
    const params = [name || req.user.name, phone || null, currency || 'INR', theme || 'dark'];

    if (password) {
      const hash = await bcrypt.hash(password, 12);
      updateQuery += ', password = ?';
      params.push(hash);
    }

    updateQuery += ', updated_at = datetime(\'now\') WHERE id = ?';
    params.push(req.user.id);

    db.prepare(updateQuery).run(...params);

    const updatedUser = db.prepare(
      'SELECT id, name, email, role, avatar, phone, currency, theme FROM users WHERE id = ?'
    ).get(req.user.id);

    res.json({ message: 'Profile updated successfully', user: updatedUser });
  } catch (err) {
    next(err);
  }
}
