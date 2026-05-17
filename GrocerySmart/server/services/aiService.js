/* ─────────────────────────────────────────────────────────────
   AI Service — Category categorization & Spending Insights
   ───────────────────────────────────────────────────────────── */

import { getDb } from '../config/database.js';

// Cache matching category rules locally to avoid repeated OpenAI calls
const categoryRules = [
  { keywords: ['tomato', 'onion', 'potato', 'carrot', 'cabbage', 'spinach', 'garlic', 'ginger', 'chili', 'coriander', 'okra', 'cucumber'], category: 'Vegetables' },
  { keywords: ['apple', 'banana', 'orange', 'mango', 'grape', 'strawberry', 'blueberry', 'watermelon', 'papaya', 'pineapple', 'kiwi'], category: 'Fruits' },
  { keywords: ['milk', 'cheese', 'butter', 'curd', 'paneer', 'yogurt', 'cream', 'ghee', 'dairy'], category: 'Dairy' },
  { keywords: ['chip', 'cookie', 'biscuit', 'snack', 'popcorn', 'chocolate', 'wafer', 'mixture', 'namkeen', 'candy', 'nuts', 'cashew', 'almond'], category: 'Snacks' },
  { keywords: ['coke', 'pepsi', 'soda', 'water', 'juice', 'beer', 'wine', 'sprite', 'coffee', 'tea', 'energy drink', 'beverage', 'drink'], category: 'Beverages' },
  { keywords: ['detergent', 'soap', 'cleaner', 'dish', 'tissue', 'napkin', 'bulb', 'battery', 'garbage bag', 'sponge', 'foil', 'household'], category: 'Household' },
  { keywords: ['shampoo', 'conditioner', 'paste', 'brush', 'lotion', 'deodorant', 'body wash', 'cream', 'perfume', 'razor', 'trimmer'], category: 'Personal Care' },
  { keywords: ['tablet', 'syrup', 'capsule', 'ointment', 'painkiller', 'aspirin', 'paracetamol', 'bandage', 'vitamin', 'supplement'], category: 'Medicines' },
  { keywords: ['rice', 'flour', 'wheat', 'dal', 'pulse', 'grain', 'oil', 'sugar', 'salt', 'spice', 'masala', 'atta', 'basmati'], category: 'Grains & Pulses' },
  { keywords: ['chicken', 'mutton', 'fish', 'prawn', 'egg', 'meat', 'beef', 'pork', 'sausage', 'bacon', 'seafood'], category: 'Meat & Seafood' },
  { keywords: ['bread', 'bun', 'cake', 'pastry', 'muffin', 'croissant', 'loaf', 'bakery'], category: 'Bakery' },
  { keywords: ['frozen', 'ice cream', 'nuggets', 'patty', 'waffle'], category: 'Frozen Foods' }
];

/**
 * Categorize a grocery item name into one of the system categories.
 * Tries local heuristic search first, fallback to OpenAI if API key is set.
 */
export async function categorizeItem(itemName) {
  const nameLower = itemName.toLowerCase();
  
  // 1. Local Rule-based Matching (Highly efficient)
  for (const rule of categoryRules) {
    if (rule.keywords.some(keyword => nameLower.includes(keyword))) {
      return rule.category;
    }
  }

  // 2. OpenAI Fallback (if API key exists)
  if (process.env.OPENAI_API_KEY && !process.env.OPENAI_API_KEY.includes('xxxx')) {
    try {
      const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`
        },
        body: JSON.stringify({
          model: 'gpt-3.5-turbo',
          messages: [
            {
              role: 'system',
              content: 'You are an AI grocery categorizer. Categorize the given grocery item name into exactly one of these categories: Vegetables, Fruits, Dairy, Snacks, Beverages, Household, Personal Care, Medicines, Grains & Pulses, Meat & Seafood, Bakery, Frozen Foods, Others. Reply with only the category name.'
            },
            {
              role: 'user',
              content: itemName
            }
          ],
          temperature: 0.1,
          max_tokens: 10
        })
      });

      const data = await response.json();
      const category = data.choices?.[0]?.message?.content?.trim();
      
      const validCategories = [
        'Vegetables', 'Fruits', 'Dairy', 'Snacks', 'Beverages', 
        'Household', 'Personal Care', 'Medicines', 'Grains & Pulses', 
        'Meat & Seafood', 'Bakery', 'Frozen Foods', 'Others'
      ];

      if (validCategories.includes(category)) {
        return category;
      }
    } catch (err) {
      console.error('OpenAI Item Categorization Error:', err.message);
    }
  }

  return 'Others';
}

/**
 * Generate spending insights using SQLite analytics and/or OpenAI.
 */
export async function generateSpendingInsights(userId) {
  const db = getDb();
  
  // Gather category breakdown of the last 30 days
  const analytics = db.prepare(`
    SELECT c.name as category, SUM(e.amount) as total
    FROM expenses e
    JOIN categories c ON e.category_id = c.id
    WHERE e.user_id = ? AND e.expense_date >= date('now', '-30 days')
    GROUP BY c.id
    ORDER BY total DESC
  `).all(userId);

  const totalSpent = analytics.reduce((sum, item) => sum + item.total, 0);

  if (totalSpent === 0) {
    return [
      "No expenses recorded in the last 30 days. Start scanning bills to receive tailored insights!",
      "Tip: Set budget alerts in the Budget Planner to manage your grocery spending."
    ];
  }

  const insights = [];

  // Rules-based insights
  const highest = analytics[0];
  if (highest) {
    insights.push(`Your highest spending category is **${highest.category}** (₹${highest.total.toFixed(2)}), representing **${((highest.total / totalSpent) * 100).toFixed(0)}%** of your total monthly grocery budget.`);
  }

  // Dairy/Snacks check
  const snacks = analytics.find(item => item.category === 'Snacks');
  if (snacks && snacks.total > (totalSpent * 0.25)) {
    insights.push(`Warning: You're spending **${((snacks.total / totalSpent) * 100).toFixed(0)}%** of your budget on Snacks. Reducing junk food purchases can save you significant money.`);
  }

  // Fruits/Vegetables health-check
  const veggies = analytics.find(item => item.category === 'Vegetables')?.total || 0;
  const fruits = analytics.find(item => item.category === 'Fruits')?.total || 0;
  const greensTotal = veggies + fruits;
  if (greensTotal > 0 && greensTotal < (totalSpent * 0.15)) {
    insights.push("Health Alert: Less than 15% of your grocery budget goes towards fresh Vegetables and Fruits. Consider incorporating more fresh produce into your weekly basket.");
  } else if (greensTotal > (totalSpent * 0.35)) {
    insights.push("Great job! You maintain a highly nutritious shopping behavior, investing over 35% of your total expenditure in fresh vegetables and fruits.");
  }

  // If OpenAI API key is present, get premium AI insights
  if (process.env.OPENAI_API_KEY && !process.env.OPENAI_API_KEY.includes('xxxx')) {
    try {
      const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`
        },
        body: JSON.stringify({
          model: 'gpt-3.5-turbo',
          messages: [
            {
              role: 'system',
              content: 'You are an advanced AI financial analyst and nutritionist. Based on the user\'s grocery spending data, generate 3 action-oriented, precise insights to help them save money or improve their shopping list. Keep comments under 2 sentences each.'
            },
            {
              role: 'user',
              content: `Monthly Spending Data: Total Spent = ₹${totalSpent.toFixed(2)}. Breakdown: ${JSON.stringify(analytics)}`
            }
          ],
          temperature: 0.7,
          max_tokens: 200
        })
      });

      const data = await response.json();
      const openAiInsights = data.choices?.[0]?.message?.content?.split('\n').filter(Boolean);
      if (openAiInsights && openAiInsights.length) {
        return openAiInsights.map(ins => ins.replace(/^\d+[\.\-\s]+/, '').trim());
      }
    } catch (err) {
      console.error('OpenAI Insights Error:', err.message);
    }
  }

  // Fallback to rules-based insights
  if (insights.length < 3) {
    insights.push("Tip: Batch your grocery shopping bi-weekly instead of daily trips to lower impulsive purchase rates.");
    insights.push("Budget reminder: Review and update category limits on your dashboard monthly to stay on target.");
  }

  return insights;
}
