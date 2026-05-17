/* ─────────────────────────────────────────────────────────────
   OCR Service — Tesseract.js & Google Vision text extraction
   ───────────────────────────────────────────────────────────── */

import Tesseract from 'tesseract.js';
import crypto from 'crypto';
import fs from 'fs';
import { visionClient } from '../config/vision.js';

/**
 * Extract text from a bill image using Google Vision (if configured) or Tesseract.js fallback.
 * Returns { text, confidence, language }.
 */
export async function extractTextFromImage(imagePath, language = 'eng') {
  try {
    // 1. Try Google Cloud Vision OCR first if client is configured
    if (visionClient) {
      console.log('👁️  Performing OCR via Google Cloud Vision...');
      const [result] = await visionClient.textDetection(imagePath);
      const detections = result.textAnnotations;
      const fullText = detections[0] ? detections[0].description : '';

      return {
        text: fullText,
        confidence: 99,
        language: 'eng',
      };
    }

    // 2. Fall back to local Tesseract.js OCR
    console.log('⚙️  Performing OCR via Tesseract.js (local fallback)...');
    const { data } = await Tesseract.recognize(imagePath, language, {
      logger: (m) => {
        if (m.status === 'recognizing text') {
          // Progress updates can be logged or emitted
        }
      },
    });

    return {
      text: data.text,
      confidence: data.confidence,
      language: data.language || language,
    };
  } catch (err) {
    console.error('OCR extraction failed:', err.message);
    throw new Error(`Failed to extract text from image: ${err.message}`);
  }
}

/**
 * Parse raw OCR text into structured bill data.
 * Uses regex patterns common across Indian & international grocery bills.
 */
export function parseOcrText(rawText) {
  const lines = rawText.split('\n').map((l) => l.trim()).filter(Boolean);

  let storeName = '';
  let billDate = '';
  let totalAmount = 0;
  let taxAmount = 0;
  let discountAmount = 0;
  const items = [];

  // Attempt to extract store name (usually in first 3 lines, all-caps or bold)
  for (let i = 0; i < Math.min(3, lines.length); i++) {
    if (lines[i].length > 3 && !/\d{2}[\/\-]\d{2}/.test(lines[i])) {
      storeName = lines[i];
      break;
    }
  }

  // Date extraction — multiple formats
  const datePatterns = [
    /(\d{2}[\/\-]\d{2}[\/\-]\d{4})/,
    /(\d{4}[\/\-]\d{2}[\/\-]\d{2})/,
    /(\d{2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s\d{4})/i,
  ];
  for (const line of lines) {
    for (const pattern of datePatterns) {
      const match = line.match(pattern);
      if (match) { billDate = match[1]; break; }
    }
    if (billDate) break;
  }

  // Total extraction
  const totalPatterns = [
    /(?:total|grand\s*total|net\s*amount|amount\s*due|payable)[:\s]*[₹$€£]?\s*([\d,]+\.?\d*)/i,
    /[₹$€£]\s*([\d,]+\.?\d*)\s*(?:total)/i,
  ];
  for (const line of lines) {
    for (const pattern of totalPatterns) {
      const match = line.match(pattern);
      if (match) {
        totalAmount = parseFloat(match[1].replace(/,/g, ''));
        break;
      }
    }
  }

  // Tax extraction
  const taxMatch = rawText.match(/(?:tax|gst|vat|cgst|sgst)[:\s]*[₹$€£]?\s*([\d,]+\.?\d*)/i);
  if (taxMatch) taxAmount = parseFloat(taxMatch[1].replace(/,/g, ''));

  // Discount extraction
  const discountMatch = rawText.match(/(?:discount|savings|saved)[:\s]*[₹$€£]?\s*([\d,]+\.?\d*)/i);
  if (discountMatch) discountAmount = parseFloat(discountMatch[1].replace(/,/g, ''));

  // Item extraction — look for lines with a price pattern
  const itemPattern = /^(.+?)\s+(\d+\.?\d*)\s*(?:x|×|@)?\s*[₹$€£]?\s*(\d+\.?\d*)\s*[₹$€£]?\s*(\d+\.?\d*)$/i;
  const simpleItemPattern = /^(.+?)\s+[₹$€£]?\s*(\d+\.?\d{2})$/;

  for (const line of lines) {
    // Skip header/footer lines
    if (/total|subtotal|tax|gst|discount|thank|visit|tel|phone|address/i.test(line)) continue;

    let match = line.match(itemPattern);
    if (match) {
      items.push({
        name: match[1].trim(),
        quantity: parseFloat(match[2]),
        price: parseFloat(match[3]),
        total_price: parseFloat(match[4]),
      });
      continue;
    }

    match = line.match(simpleItemPattern);
    if (match) {
      items.push({
        name: match[1].trim(),
        quantity: 1,
        price: parseFloat(match[2]),
        total_price: parseFloat(match[2]),
      });
    }
  }

  // If no total found, sum item totals
  if (!totalAmount && items.length) {
    totalAmount = items.reduce((sum, it) => sum + it.total_price, 0);
  }

  return { storeName, billDate, totalAmount, taxAmount, discountAmount, items };
}

/**
 * Generate a hash of the bill image for duplicate detection.
 */
export function generateBillHash(imagePath) {
  const buffer = fs.readFileSync(imagePath);
  return crypto.createHash('sha256').update(buffer).digest('hex');
}
