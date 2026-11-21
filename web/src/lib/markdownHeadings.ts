// The "## " headings of a markdown document, for an "On this page" menu. Shared by /privacy and /case-study.
// Ids follow GitHub's anchors ("## 1. At a glance" -> "1-at-a-glance"), so a document's own #links work on the site.

export type Heading = { id: string; text: string; line: number }

export function slugify(text: string): string {
  return text
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[̀-ͯ]/g, "")
    .replace(/[^a-z0-9\s-]/g, "")
    .trim()
    .replace(/\s+/g, "-")
}

// Inline markdown to plain text: links keep their label, emphasis and code marks go
export function plain(markdown: string): string {
  return markdown
    .replace(/!?\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/[*_`~]/g, "")
    .trim()
}

// Every "## " heading outside code fences, with the source line react-markdown reports for it, so a heading's id
// is found by its line (exact even if two headings share a name)
export function headingsOf(markdown: string): Heading[] {
  const used = new Map<string, number>()
  const out: Heading[] = []
  let fenced = false
  markdown.split("\n").forEach((raw, index) => {
    if (/^\s*(```|~~~)/.test(raw)) fenced = !fenced
    const match = !fenced && /^##\s+(.+?)\s*#*\s*$/.exec(raw)
    if (!match) return
    const text = plain(match[1])
    const base = slugify(text) || "section"
    const count = used.get(base) ?? 0
    used.set(base, count + 1)
    out.push({ id: count ? `${base}-${count + 1}` : base, text, line: index + 1 })
  })
  return out
}
