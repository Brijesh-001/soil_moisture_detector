import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  base: '/soil_moisture_detector/',
  plugins: [react()],
})