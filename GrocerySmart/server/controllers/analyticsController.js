/* ─────────────────────────────────────────────────────────────
   Analytics Controller
   ───────────────────────────────────────────────────────────── */

import * as analyticsService from '../services/analyticsService.js';

/**
 * Comprehensive dashboard analytics payload
 */
export async function getDashboard(req, res, next) {
  try {
    const analytics = await analyticsService.getDashboardAnalytics(req.user.id);
    res.json(analytics);
  } catch (err) {
    next(err);
  }
}
