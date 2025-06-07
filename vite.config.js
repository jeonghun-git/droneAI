
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { componentTagger } from 'lovable-tagger'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    componentTagger()
  ],
  server: {
    port: 8080
  }
})
