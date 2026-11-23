import { useEffect, useMemo } from "react"
import ReactMarkdown, { type Components } from "react-markdown"
import remarkGfm from "remark-gfm"
import { Info, Lightbulb, TriangleAlert } from "lucide-react"
import LandingNav from "@/components/landing/LandingNav"
import Footer from "@/components/landing/Footer"
import OnThisPage, { useActiveHeading } from "@/components/docs/OnThisPage"
import { DocGlow, fencedCode, languageOf, sharedDocComponents } from "@/components/docs/docMarkdown"
import { CodeLanguageProvider, DocCodeBlock, DocCodeTabs } from "@/components/docs/DocCode"
import { cn } from "@/lib/utils"
import { headingsOf } from "@/lib/markdownHeadings"
import { docsUrlTransform, parseCodeTabs, remarkCallouts, remarkCodeTabs, splitDocs, type CalloutKind } from "@/lib/docs"
// The repo's docs/DOCS.md: the one source of truth, also read on GitHub (allowed in vite.config.ts)
import source from "../../../docs/DOCS.md?raw"

const PAGE_TITLE = "Docs – Triage"

// Blue for tips and notes, amber for warnings
const CALLOUTS: Record<CalloutKind, { label: string; icon: typeof Info; box: string; accent: string }> = {
  tip: {
    label: "Tip",
    icon: Lightbulb,
    box: "border-[rgba(59,130,246,0.32)] bg-[rgba(37,99,235,0.08)] text-[#C9D6EE]",
    accent: "text-[#60A5FA]",
  },
  note: {
    label: "Note",
    icon: Info,
    box: "border-[rgba(59,130,246,0.32)] bg-[rgba(37,99,235,0.08)] text-[#C9D6EE]",
    accent: "text-[#60A5FA]",
  },
  warning: {
    label: "Warning",
    icon: TriangleAlert,
    box: "border-[rgba(245,158,11,0.32)] bg-[rgba(245,158,11,0.07)] text-[#F3E2BF]",
    accent: "text-[#FBBF24]",
  },
}

function markdownComponents(idByLine: Map<number, string>): Components {
  return {
    ...sharedDocComponents(idByLine),
    pre: ({ node }) => {
      const { code, text } = fencedCode(node)
      const title = code?.properties?.dataTitle
      return <DocCodeBlock code={text} label={typeof title === "string" ? title : languageOf(code)} />
    },
    // Consecutive titled code blocks, grouped by remarkCodeTabs
    div: ({ node, children }) => {
      const tabs = parseCodeTabs(node?.properties?.dataCodeTabs)
      return tabs ? <DocCodeTabs tabs={tabs} /> : <div>{children}</div>
    },
    // "> Tip:", "> Note:" (blue) and "> Warning:" (amber), marked by remarkCallouts; any other quote is a note
    blockquote: ({ node, children }) => {
      const kind = node?.properties?.dataCallout
      const callout = CALLOUTS[(typeof kind === "string" && kind in CALLOUTS ? kind : "note") as CalloutKind]
      const Icon = callout.icon
      return (
        <div
          role="note"
          aria-label={callout.label}
          className={cn(
            "mt-6 flex gap-3 rounded-[14px] border px-4 py-3.5 text-[14px] leading-[1.7] [&_code]:text-[#F5F7FA] [&_p:first-child]:mt-0 [&_p]:mt-2 [&_p]:text-[14px] [&_p]:leading-[1.7] [&_strong]:text-[#F5F7FA]",
            callout.box
          )}
        >
          <Icon aria-hidden="true" className={cn("mt-[4px] h-4 w-4 shrink-0", callout.accent)} />
          <div className="min-w-0">
            <p className={cn("font-semibold", callout.accent)}>{callout.label}</p>
            {children}
          </div>
        </div>
      )
    },
  }
}

export default function DocsPage() {
  const { title, lead, body } = useMemo(() => splitDocs(source), [])
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

  // A link straight to a section (/docs#7-errors): jump there once the document is on the page
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
          <p className="text-[13px] font-semibold uppercase tracking-[0.1em] text-[#60A5FA]">Docs</p>
          <h1 className="mt-3 text-4xl font-bold tracking-[-0.02em] sm:text-[48px] sm:leading-[1.1]">{title}</h1>
          {lead && <p className="mt-4 text-[17px] leading-relaxed text-[#A7ACBA]">{lead}</p>}
        </header>

        <div className="mt-10 grid gap-10 lg:mt-14 lg:grid-cols-[240px_minmax(0,1fr)] lg:gap-16">
          {/* Sticks below the 64px sticky nav; on phones it sits above the content */}
          <aside className="min-w-0 lg:sticky lg:top-[88px] lg:max-h-[calc(100vh-112px)] lg:self-start lg:overflow-y-auto">
            <OnThisPage headings={headings} active={active} />
          </aside>

          <article className="min-w-0 max-w-[800px] break-words text-[#C9CDD8]">
            <CodeLanguageProvider>
              <ReactMarkdown
                remarkPlugins={[remarkGfm, remarkCodeTabs, remarkCallouts]}
                urlTransform={docsUrlTransform}
                components={components}
              >
                {body}
              </ReactMarkdown>
            </CodeLanguageProvider>
          </article>
        </div>
      </main>

      <Footer />
    </div>
  )
}
