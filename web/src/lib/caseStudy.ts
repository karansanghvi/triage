// Turns the repo's CASE_STUDY.md (the one source of truth, also read on GitHub) into what the /case-study page
// shows. No imports from "@/", so the Node tests can load it directly.
import { defaultUrlTransform } from "react-markdown"

// Where scripts/copy-case-study-media.mjs puts the document's images (web/public/case-study-media/<same path>)
export const MEDIA_BASE = "/case-study-media/"

export type CaseStudyParts = {
  title: string            // the "# " heading
  subtitle: string | null  // the bold line under it, without the ** marks
  tagline: string | null   // an italic line right after the subtitle, without the * marks
  body: string             // everything else, without the title block and the "## Contents" section
}

const stripMarks = (line: string, mark: string) => line.trim().slice(mark.length, -mark.length).trim()

// Splits off the title block the page draws itself (so it isn't shown twice) and drops the document's own
// "## Contents" section, which the page's "On this page" menu replaces. The file itself keeps both for GitHub.
export function splitCaseStudy(markdown: string): CaseStudyParts {
  const lines = markdown.replace(/\r\n?/g, "\n").split("\n")
  let i = 0
  const skipBlank = () => {
    while (i < lines.length && lines[i].trim() === "") i++
  }

  skipBlank()
  let title = "Case study"
  if (i < lines.length && /^#\s+/.test(lines[i])) {
    title = lines[i].replace(/^#\s+/, "").replace(/\s+#*\s*$/, "").trim()
    i++
  }
  skipBlank()
  let subtitle: string | null = null
  if (i < lines.length && /^\*\*[^*].*\*\*$/.test(lines[i].trim())) {
    subtitle = stripMarks(lines[i], "**")
    i++
    skipBlank()
  }
  let tagline: string | null = null
  if (subtitle !== null && i < lines.length && /^\*[^*].*\*$/.test(lines[i].trim())) {
    tagline = stripMarks(lines[i], "*")
    i++
  }

  // Drop "## Contents" up to the next "## " heading (outside code fences), its closing "---" included
  const rest = lines.slice(i)
  const out: string[] = []
  let fenced = false
  let skipping = false
  for (const line of rest) {
    if (/^\s*(```|~~~)/.test(line)) fenced = !fenced
    if (!fenced && /^##\s+/.test(line)) skipping = /^##\s+contents\s*$/i.test(line.trim())
    if (!skipping) out.push(line)
  }
  // The title block and the contents each ended with a "---"; never leave two rules in a row
  const body = out.join("\n").replace(/\n---\s*\n(\s*\n)*---\s*\n/g, "\n---\n")
  return { title, subtitle, tagline, body: body.replace(/^\s+/, "") }
}

// The document's images are relative to the repo root (so they work on GitHub); on the site they're served from
// MEDIA_BASE. Everything else goes through react-markdown's own safe transform.
export function caseStudyUrlTransform(url: string): string {
  const isRelative = !/^[a-z][a-z0-9+.-]*:/i.test(url) && !url.startsWith("/") && !url.startsWith("#") && !url.startsWith("?")
  if (isRelative && /\.(png|jpe?g|gif|webp|svg)$/i.test(url.split(/[?#]/)[0])) {
    return MEDIA_BASE + url.replace(/^\.\//, "")
  }
  return defaultUrlTransform(url)
}

// A figure is a paragraph that holds only an image; its caption is an italic-only paragraph right after it.
// This remark plugin marks both, so the page can draw a figure (not an image inside a <p>) and a caption.
type MdNode = { type: string; value?: string; children?: MdNode[]; data?: { hName?: string; hProperties?: Record<string, unknown> } }

const meaningful = (node: MdNode) => node.children?.filter((c) => !(c.type === "text" && !c.value?.trim())) ?? []
const isFigure = (node: MdNode) => node.type === "paragraph" && meaningful(node).length === 1 && meaningful(node)[0].type === "image"
const isItalicOnly = (node: MdNode) => node.type === "paragraph" && meaningful(node).length === 1 && meaningful(node)[0].type === "emphasis"

function mark(node: MdNode) {
  const children = node.children
  if (!children) return
  children.forEach((child, index) => {
    if (isFigure(child)) {
      child.data = { ...child.data, hName: "figure", hProperties: { ...child.data?.hProperties, dataFigure: "" } }
    } else if (index > 0 && isFigure(children[index - 1]) && isItalicOnly(child)) {
      // Stays a <p> (a <figcaption> would have to sit inside the figure); the page styles it as a caption
      child.data = { ...child.data, hProperties: { ...child.data?.hProperties, dataCaption: "" } }
    }
    mark(child)
  })
}

export function remarkFigures() {
  return (tree: MdNode) => {
    mark(tree)
  }
}

// Charts are drawn on light backgrounds; the app screenshots are dark. They get different cards.
export const isScreenshot = (src: string) => /(^|\/)screens\//.test(src)
