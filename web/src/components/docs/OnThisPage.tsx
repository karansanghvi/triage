import { useEffect, useState } from "react"
import { cn } from "@/lib/utils"
import type { Heading } from "@/lib/markdownHeadings"

// The "On this page" menu of the long public documents (/privacy, /case-study), and which section is in view.

export const docFocusRing =
  "rounded-[4px] outline-none focus-visible:ring-2 focus-visible:ring-[#60A5FA] focus-visible:ring-offset-2 focus-visible:ring-offset-[#07070B]"

// The heading currently being read: the last one that has passed the top fifth of the screen
export function useActiveHeading(ids: string[]): string | null {
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

export default function OnThisPage({ headings, active }: { headings: Heading[]; active: string | null }) {
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
                  docFocusRing,
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
