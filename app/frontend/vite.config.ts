import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// NeuroVox frontend dev server. Talks to the FastAPI backend at VITE_API_URL.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, host: "127.0.0.1" },
});
