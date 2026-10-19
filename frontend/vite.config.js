import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: { port: 5190, proxy: { '/api': 'http://127.0.0.1:8100', '/outputs': 'http://127.0.0.1:8100' } },
})
