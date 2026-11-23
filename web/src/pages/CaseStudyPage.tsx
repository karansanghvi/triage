import { useEffect, useMemo, type ComponentProps, type ReactNode } from "react"
import { Link } from "react-router-dom"
import ReactMarkdown, { type Components } from "react-markdown"
import remarkGfm from "remark-gfm"
import { Sparkles } from "lucide-react"
import LandingNav from "@/components/landing/LandingNav"
import Footer from "@/components/landing/Footer"
import OnThisPage, { docFocusRing, useActiveHeading } from "@/components/docs/OnThisPage"
import MermaidDiagram, { CodeBlock } from "@/components/caseStudy/MermaidDiagram"
import { cn } from "@/lib/utils"
import { headingsOf } from "@/lib/markdownHeadings"
import { caseStudyUrlTransform, isScreenshot, remarkFigures, splitCaseStudy } from "@/lib/caseStudy"
// The repo's root CASE_STUDY.md: the one source of truth, also read on GitHub (allowed in vite.config.ts)
import source from "../../../CASE_STUDY.md?raw"

const PAGE_TITLE = "Case study – Triage"

// ---------- small pieces ----------

type HastNode = { type: string; value?: string; tagName?: string; properties?: Record<string, unknown>; children?: HastNode[] }

const textOf = (node: HastNode | undefined): string =>
  !node ? "" : node.type === "text" ? node.value ?? "" : (node.children ?? []).map(textOf).join("")

// The language of a fenced block, from its <code class="language-x">
function languageOf(code: HastNode | undefined): string | null {
  const classes = code?.properties?.className
  const list = Array.isArray(classes) ? classes : typeof classes === "string" ? classes.split(" ") : []
  const match = list.map(String).find((c) => c.startsWith("language-"))
  return match ? match.slice("language-".length) : null
}

function DocLink({ href = "", children }: { href?: string; children?: ReactNode }) {
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

// ---------- the case study's markdown, styled like /privacy ----------

function markdownComponents(idByLine: Map<number, string>): Components {
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
    p: ({ node, children }) => {
      // An italic line right after an image is its caption
      if (node?.properties && "dataCaption" in node.properties) {
        return <p className="mx-auto mt-3 max-w-[640px] text-center text-[13px] leading-relaxed text-[#8B90A0]">{children}</p>
      }
      return <p className="mt-4 text-[15px] leading-[1.8]">{children}</p>
    },
    ul: ({ children }) => <ul className="mt-4 list-disc space-y-2 pl-5 text-[15px] leading-[1.8] marker:text-[#4B5068]">{children}</ul>,
    ol: ({ children }) => <ol className="mt-4 list-decimal space-y-2 pl-5 text-[15px] leading-[1.8] marker:text-[#4B5068]">{children}</ol>,
    li: ({ children }) => <li className="pl-1 [&>p:first-child]:mt-0">{children}</li>,
    strong: ({ children }) => <strong className="font-semibold text-[#F5F7FA]">{children}</strong>,
    em: ({ children }) => <em className="italic">{children}</em>,
    a: ({ href, children }) => <DocLink href={href}>{children}</DocLink>,
    hr: () => <hr className="my-12 border-[#1A1C26]" />,
    // Inline code; fenced blocks are drawn by `pre` below
    code: ({ children }) => (
      <code className="rounded-md bg-[#1A1C26] px-1.5 py-0.5 font-mono text-[13px] text-[#F5F7FA] [overflow-wrap:anywhere]">
        {children}
      </code>
    ),
    pre: ({ node }) => {
      const code = (node?.children as HastNode[] | undefined)?.find((c) => c.tagName === "code")
      const text = textOf(code).replace(/\n$/, "")
      const language = languageOf(code)
      if (language === "mermaid") return <MermaidDiagram source={text} />
      return <CodeBlock code={text} label={language ?? undefined} />
    },
    // The "Highlights" box: a blue-tinted card
    blockquote: ({ children }) => (
      <div className="mt-8 flex gap-3 rounded-[16px] border border-[rgba(59,130,246,0.32)] bg-[rgba(37,99,235,0.08)] px-5 py-4 text-[#C9D6EE] [&_li]:leading-[1.7] [&_p:first-child]:mt-0 [&_p]:mt-2 [&_strong]:text-[#F5F7FA] [&_ul]:mt-3">
        <Sparkles aria-hidden="true" className="mt-[5px] h-4 w-4 shrink-0 text-[#60A5FA]" />
        <div className="min-w-0">{children}</div>
      </div>
    ),
    // A paragraph that holds only an image (marked by remarkFigures)
    figure: ({ children }) => <figure className="mt-8">{children}</figure>,
    img: ({ src, alt }) => {
      const url = typeof src === "string" ? src : ""
      const dark = isScreenshot(url)
      return (
        <span
          className={cn(
            "mx-auto block w-fit max-w-full overflow-hidden rounded-[16px] border",
            dark ? "border-[#1F2130] bg-[#0E0F15]" : "border-[#2A2D3D] bg-white p-3 sm:p-4"
          )}
        >
          <img
            src={url}
            alt={alt ?? ""}
            loading="lazy"
            decoding="async"
            className={cn("mx-auto block h-auto max-w-full", dark ? "" : "rounded-[8px]")}
          />
        </span>
      )
    },
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

// ---------- the page ----------

export default function CaseStudyPage() {
  const { title, subtitle, tagline, body } = useMemo(() => splitCaseStudy(source), [])
  const headings = useMemo(() => headingsOf(body), [body])
  const ids = useMemo(() => headings.map((h) => h.id), [headings])
  const components = useMemo(() => markdownComponents(new Map(headings.map((h) => [h.line, h.id]))), [headings])
  const active = useActiveHeading(ids)

  useEffect(() => {
    const previous = document.title
    document.title = PAGE_TITLE
    return () => {
      document.title = previous
    }
  }, [])

  // A link straight to a section (/case-study#6-evaluation): jump there once the document is on the page
  useEffect(() => {
    const id = decodeURIComponent(window.location.hash.slice(1))
    if (id) document.getElementById(id)?.scrollIntoView()
  }, [])

  return (
    // overflow-x-clip, not hidden: hidden would make this a scroll container and stop the menu from sticking
    <div className="relative min-h-screen overflow-x-clip bg-[#07070B] text-foreground">
      <div
        aria-hidden="true"
        className="pointer-events-none absolute -top-[260px] left-1/2 h-[560px] w-[1000px] max-w-[160vw] -translate-x-1/2 rounded-full"
        style={{
          background:
            "radial-gradient(closest-side, rgba(37,99,235,0.28), rgba(37,99,235,0.07) 55%, rgba(7,7,11,0) 100%)",
        }}
      />

      <LandingNav />

      <main className="relative mx-auto max-w-[1180px] px-4 pb-24 pt-10 sm:px-6 sm:pt-14">
        <header className="max-w-[860px]">
          <p className="text-[13px] font-semibold uppercase tracking-[0.1em] text-[#60A5FA]">Case study</p>
          <h1 className="mt-3 text-4xl font-bold tracking-[-0.02em] sm:text-[48px] sm:leading-[1.1]">{title}</h1>
          {subtitle && <p className="mt-4 text-lg font-medium leading-snug text-[#E2E5EC] sm:text-xl">{subtitle}</p>}
          {tagline && <p className="mt-3 text-[15px] leading-relaxed text-[#8B90A0]">{tagline}</p>}
        </header>

        <div className="mt-10 grid gap-10 lg:mt-14 lg:grid-cols-[240px_minmax(0,1fr)] lg:gap-16">
          {/* Sticks below the 64px sticky nav; on phones it sits above the content */}
          <aside className="min-w-0 lg:sticky lg:top-[88px] lg:max-h-[calc(100vh-112px)] lg:self-start lg:overflow-y-auto">
            <OnThisPage headings={headings} active={active} />
          </aside>

          <article className="min-w-0 max-w-[800px] break-words text-[#C9CDD8]">
            <ReactMarkdown
              remarkPlugins={[remarkGfm, remarkFigures]}
              urlTransform={caseStudyUrlTransform}
              components={components}
            >
              {body}
            </ReactMarkdown>
          </article>
        </div>
      </main>

      <Footer />
    </div>
  )
}
