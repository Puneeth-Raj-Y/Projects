/* ─────────────────────────────────────────────────────────────
   Categories Router
   ───────────────────────────────────────────────────────────── */

import express from 'express';
import * as categoriesController from '../controllers/categoriesController.js';
import { authenticate } from '../middleware/auth.js';
import { requireFields } from '../middleware/validation.js';

const router = express.Router();

/**
 * @route   GET /api/categories
 * @desc    Fetch all categories
 */
router.get('/', authenticate, categoriesController.getCategories);

/**
 * @route   POST /api/categories
 * @desc    Create a custom category
 */
router.post('/', authenticate, requireFields('name'), categoriesController.createCategory);

export default router;
