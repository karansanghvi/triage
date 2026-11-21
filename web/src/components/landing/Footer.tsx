import { Link } from "react-router-dom"
import { Split } from "lucide-react"

const GITHUB_URL = "https://github.com/karansanghvi/triage"
const AUTHOR_URL = "https://www.linkedin.com/in/karansanghvi03/"

type FooterLink = { label: string; to: string; external?: boolean }

const COLUMNS: { title: string; links: FooterLink[] }[] = [
  {
    title: "Product",
    links: [
      { label: "Chat", to: "/chat" },
      { label: "Dashboard", to: "/dashboard" },
      { label: "Benchmark", to: "/dashboard/benchmark" },
    ],
  },
  {
    title: "Developers",
    links: [
      { label: "Connect your app", to: "/settings/get-started" },
      { label: "Case study", to: "/case-study" },
      { label: "GitHub", to: GITHUB_URL, external: true },
    ],
  },
  {
    title: "Legal",
    links: [{ label: "Privacy", to: "/privacy" }],
  },
]

const linkClass = "text-sm text-[#A7ACBA] transition-colors hover:text-foreground"

export default function Footer() {
  return (
    <footer className="border-t border-[#1A1C26] px-6">
      {/* Top: brand and links */}
      <div className="mx-auto flex max-w-6xl flex-wrap justify-between gap-12 py-14">
        <div className="max-w-xs">
          <Link to="/" className="inline-flex items-center gap-2.5">
            <span className="flex h-8 w-8 items-center justify-center rounded-[10px] bg-[#2563EB]">
              <Split className="h-4 w-4 text-white" strokeWidth={2.2} />
            </span>
            <span className="text-lg font-semibold">Triage</span>
          </Link>
          <p className="mt-4 text-sm leading-relaxed text-[#8B90A0]">
            Every prompt, routed to the right model. Lower costs, near-identical quality.
          </p>
          <p className="mt-4 text-sm text-[#8B90A0]">
            Built by{" "}
            <a
              href={AUTHOR_URL}
              target="_blank"
              rel="noreferrer"
              className="font-medium text-[#C9CDD8] underline-offset-4 transition-colors hover:text-foreground hover:underline"
            >
              Karan Sanghvi
            </a>
          </p>
        </div>

        <nav aria-label="Footer" className="grid grid-cols-2 gap-x-16 gap-y-10 sm:grid-cols-3">
          {COLUMNS.map((column) => (
            <div key={column.title}>
              <p className="text-xs font-semibold uppercase tracking-[0.08em] text-[#6B7080]">{column.title}</p>
              <ul className="mt-4 space-y-3">
                {column.links.map((link) => (
                  <li key={link.label}>
                    {link.external ? (
                      <a href={link.to} target="_blank" rel="noreferrer" className={linkClass}>
                        {link.label}
                      </a>
                    ) : (
                      <Link to={link.to} className={linkClass}>
                        {link.label}
                      </Link>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </nav>
      </div>

      {/* Bottom bar: copyright and legal */}
      <div className="mx-auto flex max-w-6xl flex-wrap items-start justify-between gap-x-10 gap-y-3 border-t border-[#1A1C26] py-6">
        <p className="text-xs text-[#8B90A0]">© 2026 Triage. All rights reserved.</p>
        <p className="max-w-2xl text-xs leading-relaxed text-[#6B7080] sm:text-right">
          Provider names are trademarks of their respective owners; Triage is not affiliated with or endorsed by them.
          Groq is a trademark of Groq LLC and/or its affiliates.
        </p>
      </div>
    </footer>
  )
}