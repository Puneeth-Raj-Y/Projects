/* ─────────────────────────────────────────────────────────────
   Admin Router
   ───────────────────────────────────────────────────────────── */

import express from 'express';
import * as adminController from '../controllers/adminController.js';
import { authenticate, authorize } from '../middleware/auth.js';

const router = express.Router();

// Restrict all routes in this router to admin users
router.use(authenticate, authorize('admin'));

/**
 * @route   GET /api/admin/users
 * @desc    Fetch lists of all registered users
 */
router.get('/users', adminController.getUsers);

/**
 * @route   GET /api/admin/logs
 * @desc    System interaction and security audit log
 */
router.get('/logs', adminController.getLogs);

/**
 * @route   GET /api/admin/metrics
 * @desc    System metrics dashboard overview
 */
router.get('/metrics', adminController.getMetrics);

export default router;
