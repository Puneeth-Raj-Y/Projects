/* ─────────────────────────────────────────────────────────────
   Analytics Service
   ───────────────────────────────────────────────────────────── */

import { getDb } from '../config/database.js';
import { generateSpendingInsights } from './aiService.js';

/**
 * Fetch comprehensive financial analytics for a user
 */
export async function getDashboardAnalytics(userId) {
  const db = getDb();

  // 1. Total spending summaries (All time vs Current Month vs Prev Month)
  const currentMonthStr = new Date().toISOString().substring(0, 7); // "YYYY-MM"
  const lastMonth = new Date();
  lastMonth.setMonth(lastMonth.getMonth() - 1);
  const lastMonthStr = lastMonth.toISOString().substring(0, 7);

  const totals = db.prepare(`
    SELECT 
      COALESCE(SUM(amount), 0) as all_time,
      COALESCE(SUM(CASE WHEN strftime('%Y-%m', expense_date) = ? THEN amount ELSE 0 END), 0) as current_month,
      COALESCE(SUM(CASE WHEN strftime('%Y-%m', expense_date) = ? THEN amount ELSE 0 END), 0) as prev_month
    FROM expenses
    WHERE user_id = ?
  `).get(currentMonthStr, lastMonthStr, userId);

  // 2. Category-wise split
  const categoryExpenses = db.prepare(`
    SELECT c.name as category, c.color, c.icon, SUM(e.amount) as value
    FROM expenses e
    JOIN categories c ON e.category_id = c.id
    WHERE e.user_id = ?
    GROUP BY c.id
    ORDER BY value DESC
  `).all(userId);

  // 3. Weekly spending trends (last 8 weeks)
  const weeklyTrends = db.prepare(`
    SELECT 
      strftime('%W', expense_date) as week_num,
      min(expense_date) as week_start,
      SUM(amount) as total
    FROM expenses
    WHERE user_id = ? AND expense_date >= date('now', '-56 days')
    GROUP BY week_num
    ORDER BY week_start ASC
  `).all(userId);

  // 4. Top purchased products
  const topProducts = db.prepare(`
    SELECT name, COUNT(*) as purchase_count, SUM(total_price) as total_spent
    FROM bill_items
    WHERE bill_id IN (SELECT id FROM bills WHERE user_id = ?)
    GROUP BY name
    ORDER BY purchase_count DESC
    LIMIT 5
  `).all(userId);

  // 5. Total saving from discounts
  const savings = db.prepare(`
    SELECT COALESCE(SUM(discount_amount), 0) as total_savings
    FROM bills
    WHERE user_id = ?
  `).get(userId);

  // 6. Generate AI spending insights
  const aiInsights = await generateSpendingInsights(userId);

  return {
    summary: {
      totalSpent: totals.all_time,
      monthlySpent: totals.current_month,
      previousMonthlySpent: totals.prev_month,
      totalSavings: savings.total_savings
    },
    categoryExpenses,
    weeklyTrends,
    topProducts,
    aiInsights
  };
}
