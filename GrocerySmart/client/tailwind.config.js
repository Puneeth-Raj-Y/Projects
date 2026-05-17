/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        background: {
          dark: '#030712',
          card: 'rgba(17, 24, 39, 0.7)'
        },
        brand: {
          primary: '#10b981', // Emerald
          secondary: '#3b82f6', // Blue
          accent: '#8b5cf6' // Violet
        }
      },
      backdropBlur: {
        xs: '2px',
      }
    },
  },
  plugins: [],
}
