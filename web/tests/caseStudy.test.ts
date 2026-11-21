// Run with: npm test. Checks the /case-study page's handling of the real CASE_STUDY.md (the repo's root file).
import { test } from "node:test"
import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { createElement } from "react"
import { renderToStaticMarkup } from "react-dom/server"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { caseStudyUrlTransform, isScreenshot, MEDIA_BASE, remarkFigures, splitCaseStudy } from "../src/lib/caseStudy.ts"
import { headingsOf } from "../src/lib/markdownHeadings.ts"

const source = readFileSync(new URL("../../CASE_STUDY.md", import.meta.url), "utf8")
const parts = splitCaseStudy(source)

test("the title block is split off, so it isn't shown twice", () => {
  assert.equal(parts.title, source.match(/^#\s+(.+)$/m)?.[1].trim())
  assert.ok(parts.subtitle && !parts.subtitle.includes("**"))
  assert.ok(!/^#\s/m.test(parts.body.split("\n").slice(0, 3).join("\n")), "no h1 left at the top of the body")
  assert.ok(!parts.body.includes(`**${parts.subtitle}**`))
  if (parts.tagline) assert.ok(!parts.body.includes(`*${parts.tagline}*`))
})

test("the document's own Contents section is hidden, and no two rules are left in a row", () => {
  assert.ok(source.includes("## Contents"), "the file keeps it for GitHub")
  assert.ok(!/^##\s+Contents/m.test(parts.body))
  assert.ok(!/\n---\s*\n\s*---\s*\n/.test(parts.body))
})

test("every #link in the document points at a heading id on the page", () => {
  const ids = new Set(headingsOf(parts.body).map((h) => h.id))
  const links = [...parts.body.matchAll(/\]\(#([^)]+)\)/g)].map((m) => m[1])
  assert.ok(links.length > 0)
  for (const link of links) assert.ok(ids.has(link), `#${link} has no heading`)
})

test("image paths are served from the copied media folder; other URLs are untouched", () => {
  assert.equal(caseStudyUrlTransform("docs/case-study/x.png"), `${MEDIA_BASE}docs/case-study/x.png`)
  assert.equal(caseStudyUrlTransform("reports/pareto_test.png"), `${MEDIA_BASE}reports/pareto_test.png`)
  assert.equal(caseStudyUrlTransform("#6-evaluation"), "#6-evaluation")
  assert.equal(caseStudyUrlTransform("https://example.com/a.png"), "https://example.com/a.png")
  assert.equal(caseStudyUrlTransform("javascript:alert(1)"), "")
  assert.ok(isScreenshot("/case-study-media/docs/case-study/screens/chat-screen.png"))
  assert.ok(!isScreenshot("/case-study-media/docs/case-study/classifier_auc.png"))
})

test("each image becomes a figure, and the italic line after it a caption", () => {
  const html = renderToStaticMarkup(
    createElement(
      ReactMarkdown,
      { remarkPlugins: [remarkGfm, remarkFigures], urlTransform: caseStudyUrlTransform },
      parts.body
    )
  )
  const images = [...source.matchAll(/!\[[^\]]*\]\(([^)]+)\)/g)].length
  assert.equal((html.match(/<figure data-figure="">/g) ?? []).length, images)
  assert.equal((html.match(/<p data-caption="">/g) ?? []).length, images, "every image in the file has a caption")
  assert.ok(!/<p><img/.test(html), "no image left inside a plain paragraph")
  assert.ok(!html.includes('src="docs/'), "no image left with a repo-relative URL")
})
