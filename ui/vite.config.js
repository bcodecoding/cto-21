import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/models': 'http://localhost:8000',
      '/datasets': 'http://localhost:8000',
      '/train': 'http://localhost:8000',
      '/runs': 'http://localhost:8000',
      '/inference': 'http://localhost:8000'
    }
  },
  build: {
    outDir: 'dist'
  }
})
