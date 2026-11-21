// Run with: npm test. Renders Markdown the way the chat does (react-markdown + GFM + the <br> plugin).
import { test } from "node:test"
import assert from "node:assert/strict"
import { createElement } from "react"
import { renderToStaticMarkup } from "react-dom/server"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { remarkHtmlBreaks } from "../src/lib/remarkHtmlBreaks.ts"

const render = (text: string) =>
  renderToStaticMarkup(createElement(ReactMarkdown, { remarkPlugins: [remarkGfm, remarkHtmlBreaks] }, text))

test("<br>, <br/> and <br /> in table cells become line breaks", () => {
  const html = render("| A | B |\n|---|---|\n| one<br>two | three<br/>four<BR />five |\n")
  assert.match(html, /<td>one<br\/>\ntwo<\/td>/)
  assert.match(html, /<td>three<br\/>\nfour<br\/>\nfive<\/td>/)
  assert.ok(!html.includes("&lt;br"), html)
})

test("<br> in a paragraph becomes a line break", () => {
  assert.equal(render("first<br>second"), "<p>first<br/>\nsecond</p>")
})

test("no other HTML renders: it stays literal text", () => {
  const html = render('<b>bold</b> <script>alert(1)</script> <img src=x onerror="alert(1)"> <br onclick="x">')
  assert.equal(
    html,
    "<p>&lt;b&gt;bold&lt;/b&gt; &lt;script&gt;alert(1)&lt;/script&gt; &lt;img src=x onerror=&quot;alert(1)&quot;&gt; " +
      "&lt;br onclick=&quot;x&quot;&gt;</p>"                // a <br> with attributes is not a plain break
  )
})

test("<br> inside code stays as written", () => {
  assert.equal(render("`a<br>b`"), "<p><code>a&lt;br&gt;b</code></p>")
  assert.match(render("```\nx<br>y\n```"), /x&lt;br&gt;y/)
})
