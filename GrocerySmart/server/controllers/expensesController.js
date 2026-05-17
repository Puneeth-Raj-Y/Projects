/* ─────────────────────────────────────────────────────────────
   Expenses Controller
   ───────────────────────────────────────────────────────────── */

import { v4 as uuid } from 'uuid';
import { getDb } from '../config/database.js';
import excelJS from 'xlsx';
import PDFDocument from 'pdfkit';

/**
 * Add manual grocery expense
 */
export function addExpense(req, res, next) {
  const { description, amount, category_id, expense_date } = req.body;
  const db = getDb();

  try {
    const id = uuid();
    db.prepare(`
      INSERT INTO expenses (id, user_id, category_id, description, amount, expense_date)
      VALUES (?, ?, ?, ?, ?, ?)
    `).run(id, req.user.id, category_id, description, parseFloat(amount), expense_date);

    res.status(201).json({ message: 'Expense added successfully', id });
  } catch (err) {
    next(err);
  }
}

/**
 * Fetch expenses list with dynamic filters and pagination
 */
export function getExpenses(req, res, next) {
  const db = getDb();
  const { limit, offset } = req.pagination;
  const { category, search, start_date, end_date } = req.query;

  try {
    let query = `
      SELECT e.*, c.name as category_name, c.color as category_color, c.icon as category_icon, b.store_name
      FROM expenses e
      LEFT JOIN categories c ON e.category_id = c.id
      LEFT JOIN bills b ON e.bill_id = b.id
      WHERE e.user_id = ?
    `;
    const params = [req.user.id];

    if (category) {
      query += ' AND e.category_id = ?';
      params.push(category);
    }
    if (search) {
      query += ' AND e.description LIKE ?';
      params.push(`%${search}%`);
    }
    if (start_date) {
      query += ' AND e.expense_date >= ?';
      params.push(start_date);
    }
    if (end_date) {
      query += ' AND e.expense_date <= ?';
      params.push(end_date);
    }

    query += ' ORDER BY e.expense_date DESC, e.created_at DESC LIMIT ? OFFSET ?';
    params.push(limit, offset);

    const expenses = db.prepare(query).all(...params);

    // Calculate count for pagination
    let countQuery = 'SELECT COUNT(*) as count FROM expenses WHERE user_id = ?';
    const countParams = [req.user.id];

    if (category) {
      countQuery += ' AND category_id = ?';
      countParams.push(category);
    }
    if (search) {
      countQuery += ' AND description LIKE ?';
      countParams.push(`%${search}%`);
    }
    if (start_date) {
      countQuery += ' AND expense_date >= ?';
      countParams.push(start_date);
    }
    if (end_date) {
      countQuery += ' AND expense_date <= ?';
      countParams.push(end_date);
    }

    const total = db.prepare(countQuery).get(...countParams).count;

    res.json({
      expenses,
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
 * Edit manual or parsed expense
 */
export function updateExpense(req, res, next) {
  const { description, amount, category_id, expense_date } = req.body;
  const db = getDb();

  try {
    const expense = db.prepare('SELECT id FROM expenses WHERE id = ? AND user_id = ?').get(req.params.id, req.user.id);
    if (!expense) {
      return res.status(404).json({ error: 'Expense not found' });
    }

    db.prepare(`
      UPDATE expenses 
      SET description = ?, amount = ?, category_id = ?, expense_date = ?, updated_at = datetime('now')
      WHERE id = ?
    `).run(description, parseFloat(amount), category_id, expense_date, req.params.id);

    res.json({ message: 'Expense updated successfully' });
  } catch (err) {
    next(err);
  }
}

/**
 * Delete custom or bill linked expense
 */
export function deleteExpense(req, res, next) {
  const db = getDb();

  try {
    const expense = db.prepare('SELECT id FROM expenses WHERE id = ? AND user_id = ?').get(req.params.id, req.user.id);
    if (!expense) {
      return res.status(404).json({ error: 'Expense not found' });
    }

    db.prepare('DELETE FROM expenses WHERE id = ?').run(req.params.id);
    res.json({ message: 'Expense deleted successfully' });
  } catch (err) {
    next(err);
  }
}

/**
 * Export grocery expenses report as Excel (.xlsx) sheet
 */
export function exportExcel(req, res, next) {
  const db = getDb();

  try {
    const expenses = db.prepare(`
      SELECT e.expense_date as Date, e.description as Description, e.amount as Amount, c.name as Category, b.store_name as Store
      FROM expenses e
      LEFT JOIN categories c ON e.category_id = c.id
      LEFT JOIN bills b ON e.bill_id = b.id
      WHERE e.user_id = ?
      ORDER BY e.expense_date DESC
    `).all(req.user.id);

    const worksheet = excelJS.utils.json_to_sheet(expenses);
    const workbook = excelJS.utils.book_new();
    excelJS.utils.book_append_sheet(workbook, worksheet, 'Grocery Expenses');

    const buffer = excelJS.write(workbook, { type: 'buffer', bookType: 'xlsx' });

    res.setHeader('Content-Type', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet');
    res.setHeader('Content-Disposition', 'attachment; filename=grocery_expense_report.xlsx');
    res.end(buffer);
  } catch (err) {
    next(err);
  }
}

/**
 * Export professional PDF report of expenses
 */
export function exportPdf(req, res, next) {
  const db = getDb();

  try {
    const expenses = db.prepare(`
      SELECT e.expense_date as date, e.description, e.amount, c.name as category
      FROM expenses e
      LEFT JOIN categories c ON e.category_id = c.id
      WHERE e.user_id = ?
      ORDER BY e.expense_date DESC
    `).all(req.user.id);

    const total = expenses.reduce((sum, e) => sum + e.amount, 0);

    const doc = new PDFDocument({ margin: 50 });

    res.setHeader('Content-Type', 'application/pdf');
    res.setHeader('Content-Disposition', 'attachment; filename=grocery_expense_report.pdf');
    doc.pipe(res);

    // Document header
    doc.fontSize(24).font('Helvetica-Bold').fillColor('#10b981').text('GrocerySmart Report', { align: 'center' });
    doc.fontSize(12).fillColor('#4b5563').text(`Generated on: ${new Date().toLocaleDateString()}`, { align: 'center' });
    doc.moveDown(2);

    // Summary Box
    doc.rect(50, 110, 500, 60).fillAndStroke('#f3f4f6', '#e5e7eb');
    doc.fillColor('#1f2937').fontSize(14).font('Helvetica-Bold').text('Summary Statement', 65, 120);
    doc.fontSize(12).font('Helvetica').text(`Total Grocery Spending: ₹${total.toFixed(2)}  |  Total Items Tracked: ${expenses.length}`, 65, 140);
    doc.moveDown(3);

    // Table Header
    let y = 200;
    doc.font('Helvetica-Bold').fillColor('#374151');
    doc.text('Date', 50, y);
    doc.text('Description', 150, y);
    doc.text('Category', 350, y);
    doc.text('Amount', 470, y, { align: 'right' });

    doc.moveTo(50, y + 15).lineTo(550, y + 15).strokeColor('#d1d5db').stroke();
    y += 25;

    // Table Rows
    doc.font('Helvetica').fillColor('#4b5563');
    for (const exp of expenses) {
      if (y > 700) {
        doc.addPage();
        y = 50;
      }
      doc.text(exp.date, 50, y);
      doc.text(exp.description.substring(0, 30), 150, y);
      doc.text(exp.category || 'Others', 350, y);
      doc.text(`₹${exp.amount.toFixed(2)}`, 470, y, { align: 'right' });
      y += 20;
    }

    doc.end();
  } catch (err) {
    next(err);
  }
}
