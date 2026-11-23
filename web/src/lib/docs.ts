// Turns the repo's docs/DOCS.md (the one source of truth, also read on GitHub) into what the /docs page shows.
// No imports from "@/", so the Node tests can load it directly.
import { defaultUrlTransform } from "react-markdown"

export type DocsParts = {
  title: string        // the "# " heading
  lead: string | null  // the first paragraph under it, shown in the page header
  body: string         // everything after that
}

// Splits off the title and the lead paragraph, which the page draws in its header (so they aren't shown twice)
export function splitDocs(markdown: string): DocsParts {
  const lines = markdown.replace(/\r\n?/g, "\n").split("\n")
  let i = 0
  const skipBlank = () => {
    while (i < lines.length && lines[i].trim() === "") i++
  }

  skipBlank()
  let title = "Documentation"
  if (i < lines.length && /^#\s+/.test(lines[i])) {
    title = lines[i].replace(/^#\s+/, "").replace(/\s+#*\s*$/, "").trim()
    i++
  }
  skipBlank()
  let lead: string | null = null
  // A plain paragraph only: not a heading, quote, list, table or fence
  if (i < lines.length && !/^\s*(#|>|[-*+]\s|\d+\.\s|\||```|~~~)/.test(lines[i])) {
    const start = i
    while (i < lines.length && lines[i].trim() !== "") i++
    lead = lines.slice(start, i).join(" ").replace(/\s+/g, " ").trim()
  }
  return { title, lead, body: lines.slice(i).join("\n").replace(/^\s+/, "") }
}

// Links to the live site (written in full so they also work on GitHub) stay on the site, through the router
const SITE = /^https:\/\/gettriage\.dev(?=\/|$)/

export function docsUrlTransform(url: string): string {
  if (SITE.test(url)) return url.replace(SITE, "") || "/"
  return defaultUrlTransform(url)
}

// ---------- remark plugins ----------

type MdNode = {
  type: string
  value?: string
  lang?: string | null
  meta?: string | null
  children?: MdNode[]
  data?: { hName?: string; hProperties?: Record<string, unknown> }
}

export type CodeTab = { title: string; language: string | null; code: string }

// The title of a fenced block: ```python title="Python"
export function titleOf(meta: string | null | undefined): string | null {
  const match = meta ? /(?:^|\s)title=(?:"([^"]*)"|'([^']*)'|(\S+))/.exec(meta) : null
  const title = match ? (match[1] ?? match[2] ?? match[3]).trim() : ""
  return title || null
}

// Consecutive titled fenced blocks become one tabbed code box: a <div data-code-tabs="[...]"> the page draws.
// A titled block on its own keeps its title as a label (data-title on its <code>).
function groupCodeTabs(node: MdNode) {
  const children = node.children
  if (!children) return
  const out: MdNode[] = []
  for (let i = 0; i < children.length; i++) {
    const child = children[i]
    const run: MdNode[] = []
    while (i + run.length < children.length) {
      const next = children[i + run.length]
      if (next.type !== "code" || !titleOf(next.meta)) break
      run.push(next)
    }
    if (run.length >= 2) {
      const tabs: CodeTab[] = run.map((c) => ({ title: titleOf(c.meta)!, language: c.lang ?? null, code: c.value ?? "" }))
      out.push({ type: "codeTabs", data: { hName: "div", hProperties: { dataCodeTabs: JSON.stringify(tabs) } } })
      i += run.length - 1
      continue
    }
    if (child.type === "code") {
      const title = titleOf(child.meta)
      if (title) child.data = { ...child.data, hProperties: { ...child.data?.hProperties, dataTitle: title } }
    }
    groupCodeTabs(child)
    out.push(child)
  }
  node.children = out
}

export function remarkCodeTabs() {
  return (tree: MdNode) => {
    groupCodeTabs(tree)
  }
}

export type CalloutKind = "tip" | "note" | "warning"
const CALLOUT = /^(Tip|Note|Warning):\s*/

// A blockquote whose text starts with "Tip:", "Note:" or "Warning:" becomes a callout: the word is taken off the
// text (the page shows it as the callout's label) and the kind goes on the <blockquote> as data-callout.
function markCallouts(node: MdNode) {
  for (const child of node.children ?? []) {
    if (child.type === "blockquote") {
      const first = child.children?.[0]
      const text = first?.type === "paragraph" ? first.children?.[0] : undefined
      const match = text?.type === "text" ? CALLOUT.exec(text.value ?? "") : null
      if (text && match) {
        text.value = (text.value ?? "").slice(match[0].length)
        const kind = match[1].toLowerCase() as CalloutKind
        child.data = { ...child.data, hProperties: { ...child.data?.hProperties, dataCallout: kind } }
      }
    }
    markCallouts(child)
  }
}

export function remarkCallouts() {
  return (tree: MdNode) => {
    markCallouts(tree)
  }
}

// The tabs a <div data-code-tabs> carries, or null for any other div
export function parseCodeTabs(value: unknown): CodeTab[] | null {
  if (typeof value !== "string") return null
  try {
    const tabs = JSON.parse(value) as CodeTab[]
    return Array.isArray(tabs) && tabs.length ? tabs : null
  } catch {
    return null
  }
}
