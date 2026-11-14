import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The Python server (`whistle-house serve`, default :8000) serves /api and /ws.
// In dev we proxy to it; in production it serves manager/dist itself.
const backend = process.env.WHISTLE_HOUSE_API ?? 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    proxy: {
      '/api': backend,
      '/ws': { target: backend.replace(/^http/, 'ws'), ws: true },
    },
  },
})
