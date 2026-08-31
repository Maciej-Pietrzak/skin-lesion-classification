import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // Matches backend/.env.example's CORS_ORIGINS=http://localhost:3000
    // (Vite's default 5173 would otherwise need a backend CORS change).
    port: 3000,
  },
})
