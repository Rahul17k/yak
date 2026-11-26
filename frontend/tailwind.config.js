/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#100c0b',
        ember: '#751f1c',
        vermilion: '#a4322b',
        saffron: '#d8ad5b',
        parchment: '#f5efe5',
      },
      fontFamily: {
        display: ['Georgia', 'serif'],
        sans: ['Inter', 'Noto Sans Kannada', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
