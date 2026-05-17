/* ─────────────────────────────────────────────────────────────
   Cloudinary Configuration — Media Asset Cloud Storage
   ───────────────────────────────────────────────────────────── */

import { v2 as cloudinary } from 'cloudinary';
import { CloudinaryStorage } from 'multer-storage-cloudinary';
import multer from 'multer';

// Only attempt configuration if CLOUDINARY_URL or keys are provided
const isCloudinaryConfigured = 
  process.env.CLOUDINARY_CLOUD_NAME && 
  process.env.CLOUDINARY_API_KEY && 
  process.env.CLOUDINARY_API_SECRET;

if (isCloudinaryConfigured) {
  cloudinary.config({
    cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
    api_key: process.env.CLOUDINARY_API_KEY,
    api_secret: process.env.CLOUDINARY_API_SECRET,
  });
}

// Multer Storage Configuration
let storage;

if (isCloudinaryConfigured) {
  storage = new CloudinaryStorage({
    cloudinary: cloudinary,
    params: {
      folder: 'grocerysmart-bills',
      allowed_formats: ['jpg', 'png', 'jpeg', 'webp'],
      transformation: [{ width: 1000, height: 1000, crop: 'limit' }],
    },
  });
  console.log('☁️  Cloudinary storage provider initialized.');
} else {
  // Fallback to local storage (already configured in middleware/upload.js)
  storage = null;
}

export { cloudinary, storage, isCloudinaryConfigured };
