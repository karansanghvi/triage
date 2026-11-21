import { useEffect, useMemo, useState, type ComponentProps, type ReactNode } from "react"
import { Link } from "react-router-dom"
import ReactMarkdown, { type Components } from "react-markdown"
import remarkGfm from "remark-gfm"
import { Info } from "lucide-react"
import LandingNav from "@/components/landing/LandingNav"
import Footer from "@/components/landing/Footer"
import { cn } from "@/lib/utils"
import policy from "@/content/privacy.md?raw"

// The date shown under the title. Change it together with the policy text.
const LAST_UPDATED = "November 19, 2026"
const PAGE_TITLE = "Privacy Policy – Triage"

const focusRing =
  "rounded-[4px] outline-none focus-visible:ring-2 focus-visible:ring-[#60A5FA] focus-visible:ring-offset-2 focus-visible:ring-offset-[#07070B]"

// ---------- the "On this page" menu, from the markdown's h2 headings ----------

type Heading = { id: string; text: string; line: number }

function slugify(text: string): string {
  return text
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[̀-ͯ]/g, "")
    .replace(/[^a-z0-9\s-]/g, "")
    .trim()
    .replace(/\s+/g, "-")
}

// Inline markdown to plain text: links keep their label, emphasis and code marks go
function plain(markdown: string): string {
  return markdown
    .replace(/!?\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/[*_`~]/g, "")
    .trim()
}

// Every "## " heading outside code fences, with the source line react-markdown reports for it, so a heading's id
// is found by its line (exact even if two headings share a name)
function headingsOf(markdown: string): Heading[] {
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

// The heading currently being read: the last one that has passed the top fifth of the screen
function useActiveHeading(ids: string[]): string | null {
  const [active, setActive] = useState<string | null>(ids[0] ?? null)

  useEffect(() => {
    const elements = ids.map((id) => document.getElementById(id)).filter((el): el is HTMLElement => !!el)
    if (!elements.length || typeof IntersectionObserver === "undefined") return

    const pick = () => {
      const line = window.innerHeight * 0.2
      let current = elements[0].id
      for (const el of elements) {
        if (el.getBoundingClientRect().top <= line) current = el.id
      }
      // At the very bottom, the last section is the one in view even if its heading can't reach the line
      if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2) {
        current = elements[elements.length - 1].id
      }
      setActive(current)
    }

    const observer = new IntersectionObserver(pick, { rootMargin: "0px 0px -80% 0px", threshold: [0, 1] })
    elements.forEach((el) => observer.observe(el))
    window.addEventListener("scroll", pick, { passive: true })
    return () => {
      observer.disconnect()
      window.removeEventListener("scroll", pick)
    }
  }, [ids])

  return active
}

function OnThisPage({ headings, active }: { headings: Heading[]; active: string | null }) {
  return (
    <nav aria-label="On this page">
      <p className="text-xs font-semibold uppercase tracking-[0.1em] text-[#8B90A0]">On this page</p>
      <ul className="mt-4 space-y-1 border-l border-[#1F2130]">
        {headings.map((h) => {
          const current = h.id === active
          return (
            <li key={h.id}>
              <a
                href={`#${h.id}`}
                aria-current={current ? "location" : undefined}
                className={cn(
                  "-ml-px block border-l-2 py-1.5 pl-4 text-[13px] leading-snug transition-colors max-md:flex max-md:min-h-11 max-md:items-center pointer-coarse:flex pointer-coarse:min-h-11 pointer-coarse:items-center",
                  focusRing,
                  current
                    ? "border-[#60A5FA] font-medium text-[#F5F7FA]"
                    : "border-transparent text-[#8B90A0] hover:border-[#3B4058] hover:text-[#C9CDD8]"
                )}
              >
                {h.text}
              </a>
            </li>
          )
        })}
      </ul>
    </nav>
  )
}

// ---------- the policy's markdown, styled ----------

function PolicyLink({ href = "", children }: { href?: string; children?: ReactNode }) {
  const className = cn("text-[#60A5FA] underline-offset-2 transition-colors hover:text-[#93C5FD] hover:underline", focusRing)
  if (href.startsWith("/")) {
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

function markdownComponents(idByLine: Map<number, string>): Components {
  return {
    // The page has its own h1 (one per page); a title line in the file isn't repeated
    h1: () => null,
    h2: ({ node, children }) => {
      const id = idByLine.get(node?.position?.start.line ?? -1)
      return (
        <h2 id={id} className="mt-12 scroll-mt-[72px] text-[22px] font-semibold tracking-[-0.01em] text-[#F5F7FA] first:mt-0">
          {children}
        </h2>
      )
    },
    h3: ({ children }) => <h3 className="mt-8 text-[17px] font-semibold text-[#F5F7FA]">{children}</h3>,
    p: ({ children }) => <p className="mt-4 text-[15px] leading-[1.8]">{children}</p>,
    ul: ({ children }) => <ul className="mt-4 list-disc space-y-2 pl-5 text-[15px] leading-[1.8] marker:text-[#4B5068]">{children}</ul>,
    ol: ({ children }) => <ol className="mt-4 list-decimal space-y-2 pl-5 text-[15px] leading-[1.8] marker:text-[#4B5068]">{children}</ol>,
    li: ({ children }) => <li className="pl-1">{children}</li>,
    strong: ({ children }) => <strong className="font-semibold text-[#F5F7FA]">{children}</strong>,
    a: ({ href, children }) => <PolicyLink href={href}>{children}</PolicyLink>,
    code: ({ children }) => (
      <code className="rounded-md bg-[#1A1C26] px-1.5 py-0.5 font-mono text-[13px] text-[#F5F7FA]">{children}</code>
    ),
    hr: () => <hr className="my-10 border-[#1A1C26]" />,
    // An amber notice box (the policy uses one for the free demo and Google note)
    blockquote: ({ children }) => (
      <div
        role="note"
        className="mt-6 flex gap-3 rounded-[14px] border border-[rgba(245,158,11,0.28)] bg-[rgba(245,158,11,0.07)] px-4 py-3.5 text-[#FCD34D] [&_a]:text-[#FCD34D] [&_a]:underline [&_p:first-child]:mt-0 [&_p]:mt-2 [&_strong]:text-[#FDE68A]"
      >
        <Info aria-hidden="true" className="mt-[5px] h-4 w-4 shrink-0" />
        <div className="min-w-0">{children}</div>
      </div>
    ),
    // Tables scroll inside their own box on narrow screens; the page itself never scrolls sideways
    table: ({ children }) => (
      <div className="mt-6 overflow-x-auto rounded-[14px] border border-[#1F2130]">
        <table className="w-full min-w-[480px] border-collapse text-left text-[14px] leading-[1.6] [&_td:first-child]:font-medium [&_td:first-child]:text-[#F5F7FA]">
          {children}
        </table>
      </div>
    ),
    thead: ({ children }) => <thead className="bg-[#0E0F15]">{children}</thead>,
    th: ({ children, style }: ComponentProps<"th">) => (
      <th
        style={style}
        className="px-4 py-3 text-[11px] font-semibold uppercase tracking-[0.08em] text-[#8B90A0]"
      >
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

export default function PrivacyPage() {
  const headings = useMemo(() => headingsOf(policy), [])
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

  // A link straight to a section (/privacy#security): the browser looks for the heading before React has drawn
  // it, so jump there once the policy is on the page
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

      <main className="relative mx-auto max-w-[1180px] px-6 pb-24 pt-10 sm:pt-14">
        <header>
          <p className="text-[13px] font-semibold uppercase tracking-[0.1em] text-[#60A5FA]">Legal</p>
          <h1 className="mt-3 text-4xl font-bold tracking-[-0.02em] sm:text-[48px] sm:leading-[1.1]">Privacy Policy</h1>
          <p className="mt-3 text-sm text-[#8B90A0]">Last updated {LAST_UPDATED}</p>
        </header>

        <div className="mt-10 grid gap-10 lg:mt-14 lg:grid-cols-[240px_minmax(0,1fr)] lg:gap-16">
          {/* Sticks below the 64px sticky nav, with the same 24px gap it had from the top before */}
          <aside className="min-w-0 lg:self-start lg:sticky lg:top-[88px]">
            <OnThisPage headings={headings} active={active} />
          </aside>

          <article className="min-w-0 max-w-[760px] break-words text-[#C9CDD8]">
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
              {policy}
            </ReactMarkdown>
          </article>
        </div>
      </main>

      <Footer />
    </div>
  )
}
