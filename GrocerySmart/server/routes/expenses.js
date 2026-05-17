/* ─────────────────────────────────────────────────────────────
   Expenses Router
   ───────────────────────────────────────────────────────────── */

import express from 'express';
import * as expensesController from '../controllers/expensesController.js';
import { authenticate } from '../middleware/auth.js';
import { requireFields, validateNumeric, paginate } from '../middleware/validation.js';

const router = express.Router();

/**
 * @route   POST /api/expenses
 * @desc    Add manual grocery expense
 */
router.post('/', authenticate, requireFields('description', 'amount', 'category_id', 'expense_date'), validateNumeric('amount'), expensesController.addExpense);

/**
 * @route   GET /api/expenses
 * @desc    Fetch expenses list with dynamic category / search filters and optimized indexes
 */
router.get('/', authenticate, paginate, expensesController.getExpenses);

/**
 * @route   GET /api/expenses/export/excel
 * @desc    Export grocery expenses report as Excel (.xlsx) sheet
 */
router.get('/export/excel', authenticate, expensesController.exportExcel);

/**
 * @route   GET /api/expenses/export/pdf
 * @desc    Export professional PDF report of expenses
 */
router.get('/export/pdf', authenticate, expensesController.exportPdf);

/**
 * @route   PUT /api/expenses/:id
 * @desc    Edit manual or parsed expense
 */
router.put('/:id', authenticate, requireFields('description', 'amount', 'category_id', 'expense_date'), validateNumeric('amount'), expensesController.updateExpense);

/**
 * @route   DELETE /api/expenses/:id
 * @desc    Delete custom or bill linked expense
 */
router.delete('/:id', authenticate, expensesController.deleteExpense);

export default router;
