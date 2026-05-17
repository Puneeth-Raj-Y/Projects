/* ─────────────────────────────────────────────────────────────
   Auth Router
   ───────────────────────────────────────────────────────────── */

import express from 'express';
import * as authController from '../controllers/authController.js';
import { authenticate } from '../middleware/auth.js';
import { requireFields, validateEmail } from '../middleware/validation.js';

const router = express.Router();

/**
 * @route   POST /api/auth/register
 * @desc    Register a new user
 */
router.post('/register', requireFields('name', 'email', 'password'), validateEmail, authController.register);

/**
 * @route   POST /api/auth/login
 * @desc    Authenticate user & get token
 */
router.post('/login', requireFields('email', 'password'), authController.login);

/**
 * @route   GET /api/auth/me
 * @desc    Get current user profile
 */
router.get('/me', authenticate, authController.getMe);

/**
 * @route   PUT /api/auth/profile
 * @desc    Update profile settings
 */
router.put('/profile', authenticate, authController.updateProfile);

export default router;
