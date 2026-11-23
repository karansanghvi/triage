import type { ComponentProps, ReactNode } from "react"
import { Link } from "react-router-dom"
import type { Components } from "react-markdown"
import { cn } from "@/lib/utils"
import { docFocusRing } from "./OnThisPage"

// The markdown styles the long public documents share (/case-study, /docs). Each page adds its own code blocks,
// quotes and figures on top.

export type HastNode = {
  type: string
  value?: string
  tagName?: string
  properties?: Record<string, unknown>
  children?: HastNode[]
}

export const textOf = (node: HastNode | undefined): string =>
  !node ? "" : node.type === "text" ? node.value ?? "" : (node.children ?? []).map(textOf).join("")

// The language of a fenced block, from its <code class="language-x">
export function languageOf(code: HastNode | undefined): string | null {
  const classes = code?.properties?.className
  const list = Array.isArray(classes) ? classes : typeof classes === "string" ? classes.split(" ") : []
  const match = list.map(String).find((c) => c.startsWith("language-"))
  return match ? match.slice("language-".length) : null
}

// A fenced block's <code> element and its text, from the <pre> react-markdown hands over
export function fencedCode(pre: HastNode | undefined): { code: HastNode | undefined; text: string } {
  const code = pre?.children?.find((c) => c.tagName === "code")
  return { code, text: textOf(code).replace(/\n$/, "") }
}

export function DocLink({ href = "", children }: { href?: string; children?: ReactNode }) {
  const className = cn("text-[#60A5FA] underline-offset-2 transition-colors hover:text-[#93C5FD] hover:underline", docFocusRing)
  if (href.startsWith("/") && !href.startsWith("//")) {
    return (
      <Link to={href} className={className}>
        {children}
      </Link>
    )
  }
  const external = /^https?:\/\//.test(href)
  return (
    <a href={href} className={className} {...(external ? { target: "_blank", rel: "noreferrer" } : {})}>
      {children}
    </a>
  )
}

// Headings, text, lists, links, inline code and tables. `idByLine`: each "## " heading's id by its source line
// (from headingsOf), so the "On this page" menu can link to it.
export function sharedDocComponents(idByLine: Map<number, string>): Components {
  return {
    h1: () => null, // the page draws the title itself
    h2: ({ node, children }) => {
      const id = idByLine.get(node?.position?.start.line ?? -1)
      return (
        <h2 id={id} className="mt-14 scroll-mt-[calc(72px+env(safe-area-inset-top))] text-[24px] font-semibold tracking-[-0.01em] text-[#F5F7FA] first:mt-0">
          {children}
        </h2>
      )
    },
    h3: ({ children }) => <h3 className="mt-9 scroll-mt-[calc(72px+env(safe-area-inset-top))] text-[18px] font-semibold text-[#F5F7FA]">{children}</h3>,
    h4: ({ children }) => <h4 className="mt-6 text-[15px] font-semibold text-[#F5F7FA]">{children}</h4>,
    p: ({ children }) => <p className="mt-4 text-[15px] leading-[1.8]">{children}</p>,
    ul: ({ children }) => <ul className="mt-4 list-disc space-y-2 pl-5 text-[15px] leading-[1.8] marker:text-[#4B5068]">{children}</ul>,
    ol: ({ children }) => <ol className="mt-4 list-decimal space-y-2 pl-5 text-[15px] leading-[1.8] marker:text-[#4B5068]">{children}</ol>,
    li: ({ children }) => <li className="pl-1 [&>p:first-child]:mt-0">{children}</li>,
    strong: ({ children }) => <strong className="font-semibold text-[#F5F7FA]">{children}</strong>,
    em: ({ children }) => <em className="italic">{children}</em>,
    a: ({ href, children }) => <DocLink href={href}>{children}</DocLink>,
    hr: () => <hr className="my-12 border-[#1A1C26]" />,
    // Inline code; fenced blocks are drawn by each page's `pre`
    code: ({ children }) => (
      <code className="rounded-md bg-[#1A1C26] px-1.5 py-0.5 font-mono text-[13px] text-[#F5F7FA] [font-variant-ligatures:none] [overflow-wrap:anywhere]">
        {children}
      </code>
    ),
    // Tables scroll inside their own box on narrow screens; the page itself never scrolls sideways
    table: ({ children }) => (
      <div className="mt-6 overflow-x-auto rounded-[14px] border border-[#1F2130]">
        <table className="w-full min-w-[520px] border-collapse text-left text-[14px] leading-[1.6] [&_td:first-child]:font-medium [&_td:first-child]:text-[#F5F7FA]">
          {children}
        </table>
      </div>
    ),
    thead: ({ children }) => <thead className="bg-[#0E0F15]">{children}</thead>,
    th: ({ children, style }: ComponentProps<"th">) => (
      <th style={style} className="px-4 py-3 text-[11px] font-semibold uppercase tracking-[0.08em] text-[#8B90A0]">
        {children}
      </th>
    ),
    td: ({ children, style }: ComponentProps<"td">) => (
      <td style={style} className="border-t border-[#1A1C26] px-4 py-3 align-top">
        {children}
      </td>
    ),
  }
}

// The dark background and blue glow at the top of the long public pages
export function DocGlow() {
  return (
    <div
      aria-hidden="true"
      className="pointer-events-none absolute -top-[260px] left-1/2 h-[560px] w-[1000px] max-w-[160vw] -translate-x-1/2 rounded-full"
      style={{
        background: "radial-gradient(closest-side, rgba(37,99,235,0.28), rgba(37,99,235,0.07) 55%, rgba(7,7,11,0) 100%)",
      }}
    />
  )
}
