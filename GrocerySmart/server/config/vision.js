/* ─────────────────────────────────────────────────────────────
   Google Cloud Vision — AI OCR Engine Config
   ───────────────────────────────────────────────────────────── */

import vision from '@google-cloud/vision';
import fs from 'fs';
import path from 'path';

let visionClient = null;

// Attempt to resolve google credentials
const credentialsPath = process.env.GOOGLE_APPLICATION_CREDENTIALS || '';
const hasServiceAccountFile = credentialsPath && fs.existsSync(credentialsPath);

if (hasServiceAccountFile || process.env.GOOGLE_CREDENTIALS_JSON) {
  try {
    let options = {};
    if (process.env.GOOGLE_CREDENTIALS_JSON) {
      options = { credentials: JSON.parse(process.env.GOOGLE_CREDENTIALS_JSON) };
    } else if (hasServiceAccountFile) {
      options = { keyFilename: credentialsPath };
    }
    
    visionClient = new vision.ImageAnnotatorClient(options);
    console.log('👁️  Google Cloud Vision client initialized successfully.');
  } catch (err) {
    console.warn('⚠️  Failed to initialize Google Vision client:', err.message);
  }
} else {
  console.log('ℹ️  Google Cloud Vision not configured. Defaulting to Tesseract.js OCR.');
}

export { visionClient };
