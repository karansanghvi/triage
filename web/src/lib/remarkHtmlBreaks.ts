// Models often write <br> for a line break inside a Markdown table cell. The chat renders no raw HTML, so it
// showed as the literal text "<br>". This remark plugin turns exactly <br>, <br/> and <br /> (any case) into
// Markdown line breaks. Every other piece of HTML stays an "html" node, which react-markdown shows as plain
// text, so no other HTML can render. <br> inside code (`<br>` or a fenced block) is code, not an html node, and
// stays as written. No imports from "@/", so the Node tests can load it directly.

type MdNode = { type: string; value?: string; children?: MdNode[] }

const BREAK_TAG = /^<br\s*\/?>$/i

function replaceBreaks(node: MdNode) {
  if (!node.children) return
  node.children = node.children.map((child) =>
    child.type === "html" && BREAK_TAG.test((child.value ?? "").trim()) ? { type: "break" } : child
  )
  for (const child of node.children) replaceBreaks(child)
}

export function remarkHtmlBreaks() {
  return (tree: MdNode) => {
    replaceBreaks(tree)
  }
}
