/* ─────────────────────────────────────────────────────────────
   Request Validation Helpers
   ───────────────────────────────────────────────────────────── */

/** Validate required fields exist on req.body */
export function requireFields(...fields) {
  return (req, res, next) => {
    const missing = fields.filter((f) => {
      const val = req.body[f];
      return val === undefined || val === null || val === '';
    });
    if (missing.length) {
      return res.status(400).json({
        error: `Missing required fields: ${missing.join(', ')}`,
      });
    }
    next();
  };
}

/** Validate email format */
export function validateEmail(req, res, next) {
  const { email } = req.body;
  if (email && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    return res.status(400).json({ error: 'Invalid email format' });
  }
  next();
}

/** Validate numeric fields */
export function validateNumeric(...fields) {
  return (req, res, next) => {
    for (const f of fields) {
      if (req.body[f] !== undefined && isNaN(Number(req.body[f]))) {
        return res.status(400).json({ error: `${f} must be a number` });
      }
    }
    next();
  };
}

/** Sanitize pagination params */
export function paginate(req, _res, next) {
  req.pagination = {
    page: Math.max(1, parseInt(req.query.page) || 1),
    limit: Math.min(100, Math.max(1, parseInt(req.query.limit) || 20)),
  };
  req.pagination.offset = (req.pagination.page - 1) * req.pagination.limit;
  next();
}
