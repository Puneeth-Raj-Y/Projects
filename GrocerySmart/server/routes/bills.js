/* ─────────────────────────────────────────────────────────────
   Bills Router
   ───────────────────────────────────────────────────────────── */

import express from 'express';
import * as billsController from '../controllers/billsController.js';
import { authenticate } from '../middleware/auth.js';
import { upload } from '../middleware/upload.js';
import { paginate } from '../middleware/validation.js';

const router = express.Router();

/**
 * @route   POST /api/bills/scan
 * @desc    Upload image, perform OCR, detect barcode, and auto-parse contents
 */
router.post('/scan', authenticate, upload.single('bill'), billsController.scanBill);

/**
 * @route   POST /api/bills/save
 * @desc    Confirm scanned details and save into SQL Database
 */
router.post('/save', authenticate, billsController.saveBill);

/**
 * @route   GET /api/bills
 * @desc    Fetch scanned bills list with pagination
 */
router.get('/', authenticate, paginate, billsController.getBills);

/**
 * @route   GET /api/bills/:id
 * @desc    Get bill details with items
 */
router.get('/:id', authenticate, billsController.getBillById);

/**
 * @route   DELETE /api/bills/:id
 * @desc    Delete scanned bill & its items/linked expenses
 */
router.delete('/:id', authenticate, billsController.deleteBill);

export default router;
