import { resolve } from "node:path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig, searchForWorkspaceRoot } from "vite"

// The /case-study page imports the repo's root CASE_STUDY.md (one source of truth), which is outside web/.
// The dev server only serves files from allowed folders, so allow that one file next to the defaults.
const CASE_STUDY = resolve(import.meta.dirname, "../CASE_STUDY.md")

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": resolve(import.meta.dirname, "./src"),
    },
  },
  server: {
    fs: {
      allow: [searchForWorkspaceRoot(process.cwd()), CASE_STUDY],
    },
  },
})
