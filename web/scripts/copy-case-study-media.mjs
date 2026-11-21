// Copies the images that the repo's CASE_STUDY.md references (docs/case-study/..., reports/...) into
// public/case-study-media/, keeping their relative paths, so the /case-study page can load them. The markdown keeps
// its GitHub-friendly relative paths; the page rewrites them to /case-study-media/<path> when it renders.
// Runs before `npm run dev` and `npm run build`. The output folder is git-ignored.
import { copyFileSync, existsSync, mkdirSync, readFileSync, rmSync } from "node:fs"
import { dirname, join, normalize, resolve, sep } from "node:path"

const webRoot = resolve(import.meta.dirname, "..")
const repoRoot = resolve(webRoot, "..")
const outDir = join(webRoot, "public", "case-study-media")

const markdown = readFileSync(join(repoRoot, "CASE_STUDY.md"), "utf8")

// ![alt](path) and ![alt](path "title"); only relative paths (no scheme, no leading slash)
const paths = new Set()
for (const match of markdown.matchAll(/!\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+"[^"]*")?\s*\)/g)) {
  const path = match[1]
  if (/^[a-z][a-z0-9+.-]*:/i.test(path) || path.startsWith("/") || path.startsWith("#")) continue
  const clean = normalize(decodeURI(path))
  if (clean.startsWith("..") || clean.split(sep).includes("..")) {
    throw new Error(`CASE_STUDY.md: image path leaves the repository: ${path}`)
  }
  paths.add(clean)
}

rmSync(outDir, { recursive: true, force: true })
const missing = []
for (const path of paths) {
  const from = join(repoRoot, path)
  if (!existsSync(from)) {
    missing.push(path)
    continue
  }
  const to = join(outDir, path)
  mkdirSync(dirname(to), { recursive: true })
  copyFileSync(from, to)
}

if (missing.length) {
  throw new Error(`CASE_STUDY.md references images that don't exist:\n  ${missing.join("\n  ")}`)
}
console.log(`case study: copied ${paths.size} images to public/case-study-media/`)
