/* ─────────────────────────────────────────────────────────────
   Bills Controller
   ───────────────────────────────────────────────────────────── */

import { v4 as uuid } from 'uuid';
import { getDb } from '../config/database.js';
import { extractTextFromImage, parseOcrText, generateBillHash } from '../services/ocrService.js';
import { scanBarcode } from '../services/barcodeService.js';
import { categorizeItem } from '../services/aiService.js';

/**
 * Upload image, perform OCR, detect barcode, and auto-parse contents
 */
export async function scanBill(req, res, next) {
  if (!req.file) {
    return res.status(400).json({ error: 'Please upload a bill image file' });
  }

  const db = getDb();
  const imagePath = `uploads/${req.file.filename}`;
  const fullLocalPath = req.file.path;

  try {
    // 1. Generate hash for duplicate detection
    const hash = generateBillHash(fullLocalPath);
    const duplicate = db.prepare('SELECT id, store_name, total_amount, bill_date FROM bills WHERE duplicate_hash = ? AND user_id = ?').get(hash, req.user.id);
    
    if (duplicate) {
      return res.status(400).json({
        error: 'Duplicate bill detected! You have already uploaded this exact receipt.',
        bill: duplicate
      });
    }

    // 2. Perform OCR
    const ocrResult = await extractTextFromImage(fullLocalPath);
    
    // 3. Auto-parse structured values from raw text
    const parsed = parseOcrText(ocrResult.text);

    // 4. Barcode scanning simulation
    let detectedBarcode = req.body.barcode || null;
    if (detectedBarcode) {
      await scanBarcode(detectedBarcode); // ensures it runs, throws error if invalid
    }

    // Adjust date from parser to correct SQL format (YYYY-MM-DD)
    let billDate = new Date().toISOString().split('T')[0]; // fallback
    if (parsed.billDate) {
      const parts = parsed.billDate.split(/[\/\-]/);
      if (parts.length === 3) {
        // Assume standard Indian DD-MM-YYYY or general YYYY-MM-DD
        if (parts[2].length === 4) {
          billDate = `${parts[2]}-${parts[1].padStart(2, '0')}-${parts[0].padStart(2, '0')}`;
        } else if (parts[0].length === 4) {
          billDate = `${parts[0]}-${parts[1].padStart(2, '0')}-${parts[2].padStart(2, '0')}`;
        }
      }
    }

    const categories = db.prepare('SELECT id, name FROM categories').all();
    const otherCategoryId = categories.find(c => c.name === 'Others')?.id;

    // AI Categorization for extracted items
    const richItems = [];
    for (const item of parsed.items) {
      const predictedCategoryName = await categorizeItem(item.name);
      const cat = categories.find(c => c.name.toLowerCase() === predictedCategoryName.toLowerCase()) || { id: otherCategoryId };
      
      richItems.push({
        id: uuid(),
        name: item.name,
        quantity: item.quantity || 1,
        price: item.price || item.total_price,
        total_price: item.total_price,
        category_id: cat.id,
        category_name: predictedCategoryName
      });
    }

    res.json({
      message: 'OCR Bill Scanning Complete',
      billDetails: {
        store_name: parsed.storeName || 'Local Supermarket',
        bill_date: billDate,
        total_amount: parsed.totalAmount || 0,
        tax_amount: parsed.taxAmount || 0,
        discount_amount: parsed.discountAmount || 0,
        image_path: imagePath,
        raw_text: ocrResult.text,
        barcode: detectedBarcode,
        duplicate_hash: hash,
        items: richItems
      }
    });
  } catch (err) {
    next(err);
  }
}

/**
 * Confirm scanned details and save into SQL Database
 */
export async function saveBill(req, res, next) {
  const { store_name, bill_date, total_amount, tax_amount, discount_amount, image_path, raw_text, barcode, duplicate_hash, items } = req.body;
  const db = getDb();

  const billId = uuid();

  const insertBill = db.prepare(`
    INSERT INTO bills (id, user_id, store_name, bill_date, total_amount, tax_amount, discount_amount, image_path, raw_text, barcode, duplicate_hash)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);

  const insertItem = db.prepare(`
    INSERT INTO bill_items (id, bill_id, category_id, name, quantity, price, total_price)
    VALUES (?, ?, ?, ?, ?, ?, ?)
  `);

  const insertExpense = db.prepare(`
    INSERT INTO expenses (id, user_id, bill_id, category_id, description, amount, expense_date)
    VALUES (?, ?, ?, ?, ?, ?, ?)
  `);

  const tx = db.transaction(() => {
    // 1. Insert into bills
    insertBill.run(
      billId,
      req.user.id,
      store_name,
      bill_date,
      total_amount,
      tax_amount,
      discount_amount,
      image_path,
      raw_text,
      barcode,
      duplicate_hash
    );

    // 2. Insert items and create proportional expenses
    for (const item of items) {
      const itemId = uuid();
      insertItem.run(
        itemId,
        billId,
        item.category_id,
        item.name,
        item.quantity,
        item.price,
        item.total_price
      );

      // Create linked expense trace
      insertExpense.run(
        uuid(),
        req.user.id,
        billId,
        item.category_id,
        `${item.name} (${item.quantity}x)`,
        item.total_price,
        bill_date
      );
    }

    // 3. Log user activity
    db.prepare(
      'INSERT INTO analytics_logs (id, user_id, action, details) VALUES (?, ?, ?, ?)'
    ).run(uuid(), req.user.id, 'bill_scanned', `Uploaded receipt from ${store_name} worth ₹${total_amount}`);
  });

  try {
    tx();
    res.status(201).json({ message: 'Grocery Bill stored successfully!', billId });
  } catch (err) {
    next(err);
  }
}

/**
 * Fetch scanned bills list with pagination
 */
export function getBills(req, res, next) {
  const db = getDb();
  const { limit, offset } = req.pagination;

  try {
    const bills = db.prepare(`
      SELECT * FROM bills 
      WHERE user_id = ? 
      ORDER BY bill_date DESC, created_at DESC
      LIMIT ? OFFSET ?
    `).all(req.user.id, limit, offset);

    const total = db.prepare('SELECT COUNT(*) as count FROM bills WHERE user_id = ?').get(req.user.id).count;

    res.json({
      bills,
      pagination: {
        page: req.pagination.page,
        limit,
        total,
        totalPages: Math.ceil(total / limit)
      }
    });
  } catch (err) {
    next(err);
  }
}

/**
 * Get bill details with items
 */
export function getBillById(req, res, next) {
  const db = getDb();

  try {
    const bill = db.prepare('SELECT * FROM bills WHERE id = ? AND user_id = ?').get(req.params.id, req.user.id);
    if (!bill) {
      return res.status(404).json({ error: 'Bill not found' });
    }

    const items = db.prepare(`
      SELECT bi.*, c.name as category_name, c.color as category_color, c.icon as category_icon
      FROM bill_items bi
      LEFT JOIN categories c ON bi.category_id = c.id
      WHERE bi.bill_id = ?
    `).all(bill.id);

    res.json({ ...bill, items });
  } catch (err) {
    next(err);
  }
}

/**
 * Delete scanned bill & its items/linked expenses
 */
export function deleteBill(req, res, next) {
  const db = getDb();

  try {
    const bill = db.prepare('SELECT id FROM bills WHERE id = ? AND user_id = ?').get(req.params.id, req.user.id);
    if (!bill) {
      return res.status(404).json({ error: 'Bill not found' });
    }

    // Cascade deletes handle bill_items and linked expenses through transactional setup or manually
    const tx = db.transaction(() => {
      db.prepare('DELETE FROM expenses WHERE bill_id = ?').run(bill.id);
      db.prepare('DELETE FROM bills WHERE id = ?').run(bill.id);
    });

    tx();
    res.json({ message: 'Scanned bill deleted successfully' });
  } catch (err) {
    next(err);
  }
}
