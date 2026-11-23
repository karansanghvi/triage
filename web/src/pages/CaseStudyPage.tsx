import { useEffect, useMemo } from "react"
import ReactMarkdown, { type Components } from "react-markdown"
import remarkGfm from "remark-gfm"
import { Sparkles } from "lucide-react"
import LandingNav from "@/components/landing/LandingNav"
import Footer from "@/components/landing/Footer"
import OnThisPage, { useActiveHeading } from "@/components/docs/OnThisPage"
import { DocGlow, fencedCode, languageOf, sharedDocComponents } from "@/components/docs/docMarkdown"
import MermaidDiagram, { CodeBlock } from "@/components/caseStudy/MermaidDiagram"
import { cn } from "@/lib/utils"
import { headingsOf } from "@/lib/markdownHeadings"
import { caseStudyUrlTransform, isScreenshot, remarkFigures, splitCaseStudy } from "@/lib/caseStudy"
// The repo's root CASE_STUDY.md: the one source of truth, also read on GitHub (allowed in vite.config.ts)
import source from "../../../CASE_STUDY.md?raw"

const PAGE_TITLE = "Case study – Triage"

// ---------- the case study's markdown, styled like /privacy ----------

function markdownComponents(idByLine: Map<number, string>): Components {
  return {
    ...sharedDocComponents(idByLine),
    p: ({ node, children }) => {
      // An italic line right after an image is its caption
      if (node?.properties && "dataCaption" in node.properties) {
        return <p className="mx-auto mt-3 max-w-[640px] text-center text-[13px] leading-relaxed text-[#8B90A0]">{children}</p>
      }
      return <p className="mt-4 text-[15px] leading-[1.8]">{children}</p>
    },
    pre: ({ node }) => {
      const { code, text } = fencedCode(node)
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
      <DocGlow />

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
