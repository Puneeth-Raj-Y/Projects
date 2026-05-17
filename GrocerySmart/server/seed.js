/* ─────────────────────────────────────────────────────────────
   Seeding Script — GrocerySmart Database
   ───────────────────────────────────────────────────────────── */

import { getDb, initializeDatabase } from './config/database.js';
import { v4 as uuid } from 'uuid';
import bcrypt from 'bcryptjs';

const db = getDb();

// Make sure the schema exists
initializeDatabase();

console.log('🌱  Seeding database with mock data...');

const tx = db.transaction(() => {
  // 1. Clear existing user/bill/expense records (to reset nicely)
  db.prepare("DELETE FROM users WHERE email != 'admin@grocerysmart.com'").run();
  db.prepare('DELETE FROM bills').run();
  db.prepare('DELETE FROM expenses').run();
  db.prepare('DELETE FROM budgets').run();
  db.prepare('DELETE FROM analytics_logs').run();

  // 2. Add dynamic demo user
  const userId = uuid();
  const passwordHash = bcrypt.hashSync('demo123', 12);
  db.prepare(`
    INSERT INTO users (id, name, email, password, role, currency, theme)
    VALUES (?, ?, ?, ?, ?, ?, ?)
  `).run(
    userId,
    'Puneeth Raj',
    'demo@grocerysmart.com',
    passwordHash,
    'user',
    'INR',
    'dark'
  );

  console.log('👤  Created Demo User: demo@grocerysmart.com / demo123');

  // Get categories
  const categories = db.prepare('SELECT id, name FROM categories').all();
  const catMap = {};
  for (const c of categories) {
    catMap[c.name] = c.id;
  }

  // 3. Add mock budget targets
  const budgetTemplate = [
    { cat: 'Vegetables', amt: 2500 },
    { cat: 'Fruits', amt: 2000 },
    { cat: 'Dairy', amt: 3000 },
    { cat: 'Snacks', amt: 1500 },
    { cat: 'Beverages', amt: 1200 },
    { cat: 'Household', amt: 2000 },
    { cat: 'Grains & Pulses', amt: 3500 },
    { cat: 'Meat & Seafood', amt: 4000 }
  ];

  const currentMonth = new Date().getMonth() + 1;
  const currentYear = new Date().getFullYear();

  for (const b of budgetTemplate) {
    if (catMap[b.cat]) {
      db.prepare(`
        INSERT INTO budgets (id, user_id, category_id, amount, month, year)
        VALUES (?, ?, ?, ?, ?, ?)
      `).run(uuid(), userId, catMap[b.cat], b.amt, currentMonth, currentYear);
    }
  }

  console.log('💰  Seeded budget targets...');

  // Helper date function
  const daysAgo = (num) => {
    const d = new Date();
    d.setDate(d.getDate() - num);
    return d.toISOString().split('T')[0];
  };

  // 4. Create bills & items
  const billsData = [
    {
      store: 'Big Bazaar Supermarket',
      date: daysAgo(2),
      discount: 150,
      tax: 85,
      items: [
        { name: 'Basmati Rice 5kg', cat: 'Grains & Pulses', qty: 1, price: 499 },
        { name: 'Sunflower Oil 2L', cat: 'Grains & Pulses', qty: 1, price: 340 },
        { name: 'Amul Butter 500g', cat: 'Dairy', qty: 1, price: 275 },
        { name: 'Organic Potatoes 2kg', cat: 'Vegetables', qty: 1, price: 80 },
        { name: 'Sweet Onions 2kg', cat: 'Vegetables', qty: 1, price: 90 }
      ]
    },
    {
      store: 'Reliance Fresh',
      date: daysAgo(5),
      discount: 75,
      tax: 45,
      items: [
        { name: 'Alphonso Mangoes 1 Dozen', cat: 'Fruits', qty: 1, price: 650 },
        { name: 'Fresh Bananas 1 Dozen', cat: 'Fruits', qty: 1, price: 70 },
        { name: 'Amul Milk Gold 1L', cat: 'Dairy', qty: 3, price: 66 },
        { name: 'Whole Wheat Bread', cat: 'Bakery', qty: 2, price: 45 }
      ]
    },
    {
      store: 'D-Mart Supermarket',
      date: daysAgo(12),
      discount: 220,
      tax: 120,
      items: [
        { name: 'Liquid Laundry Detergent', cat: 'Household', qty: 1, price: 399 },
        { name: 'Dishwash Bar Gel', cat: 'Household', qty: 2, price: 85 },
        { name: 'Premium Toothbrush 4-Pack', cat: 'Personal Care', qty: 1, price: 140 },
        { name: 'Sensodyne Toothpaste', cat: 'Personal Care', qty: 1, price: 120 },
        { name: 'Paracetamol Tablets 650mg', cat: 'Medicines', qty: 2, price: 35 }
      ]
    },
    {
      store: 'Star Bazaar',
      date: daysAgo(20),
      discount: 110,
      tax: 195,
      items: [
        { name: 'Fresh Chicken Breast 1kg', cat: 'Meat & Seafood', qty: 1.5, price: 280 },
        { name: 'Atlantic Salmon Fillets', cat: 'Meat & Seafood', qty: 1, price: 890 },
        { name: 'Organic Spinach 250g', cat: 'Vegetables', qty: 2, price: 30 },
        { name: 'Broccoli Premium', cat: 'Vegetables', qty: 1, price: 120 }
      ]
    },
    {
      store: 'More Retail Store',
      date: daysAgo(28),
      discount: 40,
      tax: 35,
      items: [
        { name: 'Potato Chips Family Pack', cat: 'Snacks', qty: 3, price: 50 },
        { name: 'Chocolate Chip Cookies', cat: 'Snacks', qty: 2, price: 80 },
        { name: 'Coca-Cola Zero Sugar 2L', cat: 'Beverages', qty: 2, price: 95 },
        { name: 'Nestea Lemon Iced Tea', cat: 'Beverages', qty: 1, price: 145 }
      ]
    }
  ];

  for (const b of billsData) {
    const billId = uuid();
    const itemTotals = b.items.reduce((sum, item) => sum + (item.qty * item.price), 0);
    const finalTotal = itemTotals + b.tax - b.discount;

    // Insert bill
    db.prepare(`
      INSERT INTO bills (id, user_id, store_name, bill_date, total_amount, tax_amount, discount_amount, image_path, status)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      billId,
      userId,
      b.store,
      b.date,
      finalTotal,
      b.tax,
      b.discount,
      'uploads/mock_bill.png',
      'processed'
    );

    // Insert bill items & proportional expenses
    for (const item of b.items) {
      const itemId = uuid();
      const catId = catMap[item.cat] || catMap['Others'];
      const totalPrice = item.qty * item.price;

      // Bill item
      db.prepare(`
        INSERT INTO bill_items (id, bill_id, category_id, name, quantity, price, total_price)
        VALUES (?, ?, ?, ?, ?, ?, ?)
      `).run(itemId, billId, catId, item.name, item.qty, item.price, totalPrice);

      // Link to expense ledger
      db.prepare(`
        INSERT INTO expenses (id, user_id, bill_id, category_id, description, amount, expense_date)
        VALUES (?, ?, ?, ?, ?, ?, ?)
      `).run(
        uuid(),
        userId,
        billId,
        catId,
        `${item.name} (${item.qty}x)`,
        totalPrice,
        b.date
      );
    }
  }

  // 5. Add custom manual expenses (unlinked to bills)
  const manualExpenses = [
    { desc: 'Fresh Coriander & Mint (Local Cart)', cat: 'Vegetables', amt: 40, date: daysAgo(1) },
    { desc: 'Red Apples (Fruit Stand)', cat: 'Fruits', amt: 180, date: daysAgo(3) },
    { desc: 'Coconut Water', cat: 'Beverages', amt: 50, date: daysAgo(4) },
    { desc: 'Multigrain Bread (Baker Cottage)', cat: 'Bakery', amt: 60, date: daysAgo(7) },
    { desc: 'Local Paneer 250g', cat: 'Dairy', amt: 120, date: daysAgo(9) },
    { desc: 'Paneer Butter Masala Pack', cat: 'Snacks', amt: 85, date: daysAgo(15) }
  ];

  for (const e of manualExpenses) {
    const catId = catMap[e.cat] || catMap['Others'];
    db.prepare(`
      INSERT INTO expenses (id, user_id, category_id, description, amount, expense_date)
      VALUES (?, ?, ?, ?, ?, ?)
    `).run(uuid(), userId, catId, e.desc, e.amt, e.date);
  }

  console.log('🛒  Successfully loaded gorgeous mockup items and bills!');
});

try {
  tx();
  console.log('✅  Database seeding complete! Ready for developer showcase.');
  process.exit(0);
} catch (err) {
  console.error('❌  Seeding failed:', err);
  process.exit(1);
}
