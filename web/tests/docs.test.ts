// Run with: npm test. Checks the /docs page's handling of the real docs/DOCS.md (the repo's file).
import { test } from "node:test"
import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { createElement } from "react"
import { renderToStaticMarkup } from "react-dom/server"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { docsUrlTransform, parseCodeTabs, remarkCallouts, remarkCodeTabs, splitDocs, titleOf } from "../src/lib/docs.ts"
import { headingsOf } from "../src/lib/markdownHeadings.ts"

const source = readFileSync(new URL("../../docs/DOCS.md", import.meta.url), "utf8")
const parts = splitDocs(source)

const render = (markdown: string) =>
  renderToStaticMarkup(
    createElement(ReactMarkdown, { remarkPlugins: [remarkGfm, remarkCodeTabs, remarkCallouts], urlTransform: docsUrlTransform }, markdown)
  )

test("the title and lead are split off, so they aren't shown twice", () => {
  assert.equal(parts.title, "Triage documentation")
  assert.ok(parts.lead && parts.lead.length > 20)
  assert.ok(!/^#\s+Triage documentation/m.test(parts.body))
  assert.ok(!parts.body.includes(parts.lead!.slice(0, 40)))
})

test("the nine sections are there, in order", () => {
  const texts = headingsOf(parts.body).map((h) => h.text)
  assert.equal(texts.length, 9)
  texts.forEach((text, i) => assert.ok(text.startsWith(`${i + 1}. `), text))
})

test("every #link in the document points at a heading id on the page", () => {
  const ids = new Set(headingsOf(parts.body).map((h) => h.id))
  const links = [...parts.body.matchAll(/\]\(#([^)]+)\)/g)].map((m) => m[1])
  assert.ok(links.length > 0)
  for (const id of links) assert.ok(ids.has(id), `#${id} has no heading`)
})

test("block titles are read from the fence's meta", () => {
  assert.equal(titleOf('title="Python"'), "Python")
  assert.equal(titleOf("title='Node'"), "Node")
  assert.equal(titleOf("title=curl"), "curl")
  assert.equal(titleOf("other"), null)
  assert.equal(titleOf(null), null)
})

test("consecutive titled blocks become one tab group; a lone titled block keeps its label", () => {
  const html = render('```python title="Python"\nprint(1)\n```\n\n```js title="Node"\nconsole.log(1)\n```\n\ntext\n\n```json title="Body"\n{}\n```\n\n```bash\nls\n```')
  const groups = [...html.matchAll(/data-code-tabs="([^"]*)"/g)]
  assert.equal(groups.length, 1)
  const tabs = parseCodeTabs(groups[0][1].replace(/&quot;/g, '"'))
  assert.deepEqual(tabs?.map((t) => [t.title, t.language, t.code]), [["Python", "python", "print(1)"], ["Node", "js", "console.log(1)"]])
  assert.match(html, /<code class="language-json" data-title="Body">/)
  assert.match(html, /<code class="language-bash">ls/)
})

test("the real document's tab groups are Python / Node / curl, and every block has code", () => {
  const html = render(parts.body)
  const groups = [...html.matchAll(/data-code-tabs="([^"]*)"/g)].map((m) => parseCodeTabs(m[1].replace(/&quot;/g, '"').replace(/&amp;/g, "&"))!)
  assert.ok(groups.length >= 8)
  for (const tabs of groups) {
    for (const tab of tabs) {
      assert.ok(["Python", "Node", "curl"].includes(tab.title), tab.title)
      assert.ok(tab.code.trim().length > 0)
    }
  }
})

test("Tip, Note and Warning quotes become callouts without the leading word", () => {
  const html = render("> Tip: one\n\n> Warning: two\n\n> Note: three\n\n> plain")
  assert.match(html, /<blockquote data-callout="tip">\s*<p>one<\/p>/)
  assert.match(html, /<blockquote data-callout="warning">\s*<p>two<\/p>/)
  assert.match(html, /<blockquote data-callout="note">\s*<p>three<\/p>/)
  assert.match(html, /<blockquote>\s*<p>plain<\/p>/)
})

test("links to the live site stay on the site", () => {
  assert.equal(docsUrlTransform("https://gettriage.dev/privacy"), "/privacy")
  assert.equal(docsUrlTransform("https://gettriage.dev"), "/")
  assert.equal(docsUrlTransform("https://api.gettriage.dev/v1"), "https://api.gettriage.dev/v1")
  assert.equal(docsUrlTransform("https://gettriage.dev.evil.com/x"), "https://gettriage.dev.evil.com/x")
  assert.equal(docsUrlTransform("javascript:alert(1)"), "")
})

test("the document is public-safe: no internal files, config keys or server variables", () => {
  for (const pattern of [/\bconfig\//, /\.ya?ml\b/, /\bapp\//, /\bTRIAGE_[A-Z_]+/, /\.py\b/, /\btau=|server_providers|routing\.yaml/]) {
    assert.ok(!pattern.test(source), `DOCS.md mentions ${pattern}`)
  }
})
