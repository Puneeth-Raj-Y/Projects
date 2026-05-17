/* ─────────────────────────────────────────────────────────────
   Budgets Router
   ───────────────────────────────────────────────────────────── */

import express from 'express';
import * as budgetsController from '../controllers/budgetsController.js';
import { authenticate } from '../middleware/auth.js';
import { requireFields, validateNumeric } from '../middleware/validation.js';

const router = express.Router();

/**
 * @route   GET /api/budgets
 * @desc    Get user budgets with current spending calculated in real-time
 */
router.get('/', authenticate, budgetsController.getBudgets);

/**
 * @route   POST /api/budgets
 * @desc    Set or update budget for a category
 */
router.post('/', authenticate, requireFields('category_id', 'amount'), validateNumeric('amount'), budgetsController.setBudget);

/**
 * @route   DELETE /api/budgets/:id
 * @desc    Delete custom budget plan
 */
router.delete('/:id', authenticate, budgetsController.deleteBudget);

export default router;
