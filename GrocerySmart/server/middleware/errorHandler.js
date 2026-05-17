/* ─────────────────────────────────────────────────────────────
   Global Error Handler Middleware
   ───────────────────────────────────────────────────────────── */

/** 404 — catch-all for unmatched routes */
export function notFound(req, res, next) {
  const error = new Error(`Not found — ${req.originalUrl}`);
  error.status = 404;
  next(error);
}

/** Central error handler */
export function errorHandler(err, _req, res, _next) {
  const status = err.status || 500;
  const message = err.message || 'Internal server error';

  console.error(`[ERROR] ${status} — ${message}`);
  if (process.env.NODE_ENV !== 'production') {
    console.error(err.stack);
  }

  res.status(status).json({
    error: message,
    ...(process.env.NODE_ENV !== 'production' && { stack: err.stack }),
  });
}
