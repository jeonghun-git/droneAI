
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        navy: '#1e3a8a', // Adding the navy color that's used in Header component
      },
    },
  },
  plugins: [],
}
