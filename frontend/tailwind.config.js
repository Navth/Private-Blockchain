/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        idle: "#3b82f6",
        validating: "#f59e0b",
        committed: "#22c55e",
        surface: "rgba(15, 23, 42, 0.7)"
      },
      boxShadow: {
        glow: "0 0 25px rgba(59, 130, 246, 0.25)",
      },
    },
  },
  plugins: [],
}

