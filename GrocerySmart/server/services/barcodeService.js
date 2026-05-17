/* ─────────────────────────────────────────────────────────────
   Barcode Scanner Service
   ───────────────────────────────────────────────────────────── */

/**
 * Identify product metadata using barcode.
 * Uses open platforms like Open Food Facts or a standard lookup mock dictionary
 * to quickly locate product name, category, and standard size.
 */
export async function scanBarcode(barcode) {
  if (!barcode) return null;

  // 1. Check local lookup for sample testing
  const mockBarcodeDatabase = {
    '8901058002316': { name: 'Britannia Marie Gold Biscuits', category: 'Snacks', price: 30.00 },
    '8901491101836': { name: 'Lays Magic Masala Chips 50g', category: 'Snacks', price: 20.00 },
    '8901719101037': { name: 'Colgate Strong Teeth Toothpaste 100g', category: 'Personal Care', price: 65.00 },
    '8901262010015': { name: 'Amul Butter 100g', category: 'Dairy', price: 56.00 },
    '8901719104069': { name: 'Palmolive Body Wash 250ml', category: 'Personal Care', price: 199.00 }
  };

  if (mockBarcodeDatabase[barcode]) {
    return mockBarcodeDatabase[barcode];
  }

  // 2. Fetch from Open Food Facts API (Real product discovery)
  try {
    const response = await fetch(`https://world.openfoodfacts.org/api/v0/product/${barcode}.json`);
    const data = await response.json();

    if (data.status === 1 && data.product) {
      const p = data.product;
      return {
        name: p.product_name || p.generic_name || 'Unknown Barcode Product',
        category: mapOffCategory(p.categories_tags?.[0]) || 'Others',
        price: 0 // Open Food Facts doesn't contain regional pricing, defaults to 0
      };
    }
  } catch (err) {
    console.error('Barcode discovery failed via Open Food Facts API:', err.message);
  }

  return null;
}

function mapOffCategory(tag) {
  if (!tag) return 'Others';
  const label = tag.toLowerCase();
  if (label.includes('vegetable') || label.includes('plant')) return 'Vegetables';
  if (label.includes('fruit')) return 'Fruits';
  if (label.includes('dairy') || label.includes('milk') || label.includes('cheese')) return 'Dairy';
  if (label.includes('snack') || label.includes('sweet') || label.includes('biscuit') || label.includes('chip')) return 'Snacks';
  if (label.includes('beverage') || label.includes('drink') || label.includes('soda')) return 'Beverages';
  if (label.includes('clean') || label.includes('detergent')) return 'Household';
  if (label.includes('hygiene') || label.includes('care') || label.includes('soap')) return 'Personal Care';
  return 'Others';
}
