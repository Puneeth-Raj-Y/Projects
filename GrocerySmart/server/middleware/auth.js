/* ─────────────────────────────────────────────────────────────
   JWT Authentication Middleware
   ───────────────────────────────────────────────────────────── */

import jwt from 'jsonwebtoken';
import { getDb } from '../config/database.js';

const JWT_SECRET = process.env.JWT_SECRET || 'dev_secret_change_me';

/** Generate a signed JWT for a user id */
export function generateToken(userId) {
  return jwt.sign({ id: userId }, JWT_SECRET, {
    expiresIn: process.env.JWT_EXPIRES_IN || '7d',
  });
}

/** Verify token and attach req.user */
export function authenticate(req, res, next) {
  try {
    const header = req.headers.authorization;
    if (!header?.startsWith('Bearer ')) {
      return res.status(401).json({ error: 'Authentication required' });
    }

    const token = header.split(' ')[1];
    const decoded = jwt.verify(token, JWT_SECRET);

    const db = getDb();
    const user = db.prepare('SELECT id, name, email, role, avatar, currency, theme FROM users WHERE id = ?').get(decoded.id);

    if (!user) {
      return res.status(401).json({ error: 'User no longer exists' });
    }

    req.user = user;
    next();
  } catch (err) {
    if (err.name === 'TokenExpiredError') {
      return res.status(401).json({ error: 'Token expired, please login again' });
    }
    return res.status(401).json({ error: 'Invalid token' });
  }
}

/** Restrict to specific roles */
export function authorize(...roles) {
  return (req, res, next) => {
    if (!req.user || !roles.includes(req.user.role)) {
      return res.status(403).json({ error: 'Insufficient permissions' });
    }
    next();
  };
}

export default authenticate;
