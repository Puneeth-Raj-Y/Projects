/* ─────────────────────────────────────────────────────────────
   Analytics Router
   ───────────────────────────────────────────────────────────── */

import express from 'express';
import * as analyticsController from '../controllers/analyticsController.js';
import { authenticate } from '../middleware/auth.js';

const router = express.Router();

/**
 * @route   GET /api/analytics/dashboard
 * @desc    Comprehensive dashboard analytics payload
 */
router.get('/dashboard', authenticate, analyticsController.getDashboard);

export default router;
