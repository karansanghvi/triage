import { resolve } from "node:path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig, searchForWorkspaceRoot } from "vite"

// The /case-study and /docs pages import the repo's CASE_STUDY.md and docs/DOCS.md (one source of truth each),
// which are outside web/. The dev server only serves files from allowed folders, so allow those two files next to
// the defaults.
const CASE_STUDY = resolve(import.meta.dirname, "../CASE_STUDY.md")
const DOCS = resolve(import.meta.dirname, "../docs/DOCS.md")

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
      allow: [searchForWorkspaceRoot(process.cwd()), CASE_STUDY, DOCS],
    },
  },
})
