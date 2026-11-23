import { createContext, useContext, useId, useRef, useState, type KeyboardEvent, type ReactNode } from "react"
import { cn } from "@/lib/utils"
import { Highlighted } from "@/components/common/CodeSnippet"
import CopyButton from "@/components/common/CopyButton"
import type { CodeTab } from "@/lib/docs"

// The /docs page's code: one box per fenced block, or one tabbed box (Python / Node / curl) for consecutive titled
// blocks. Same look as the landing page's snippet (JetBrains Mono; long lines scroll inside the box).
// Picking a tab picks it in every box that has it, and is remembered on this device.

const STORAGE_KEY = "triage.docs.language"

function readSaved(): string | null {
  try {
    return window.localStorage.getItem(STORAGE_KEY)
  } catch {
    return null
  }
}

type LanguageChoice = { language: string | null; choose: (title: string) => void }
const LanguageContext = createContext<LanguageChoice>({ language: null, choose: () => {} })

export function CodeLanguageProvider({ children }: { children: ReactNode }) {
  const [language, setLanguage] = useState<string | null>(readSaved)
  const choose = (title: string) => {
    setLanguage(title)
    try {
      window.localStorage.setItem(STORAGE_KEY, title)
    } catch {
      // Storage blocked: the choice still holds on this page
    }
  }
  return <LanguageContext.Provider value={{ language, choose }}>{children}</LanguageContext.Provider>
}

function CodeLines({ code }: { code: string }) {
  return (
    <div className="overflow-x-auto">
      <pre className="py-5 font-mono text-[13px] leading-[1.8] text-[#C9CDD8] [font-variant-ligatures:none]">
        <code className="block w-max min-w-full px-5">
          {code.split("\n").map((line, i) => (
            <span key={i} className="block">
              {line ? <Highlighted text={line} /> : " "}
            </span>
          ))}
        </code>
      </pre>
    </div>
  )
}

const boxClass = "mt-6 overflow-hidden rounded-2xl border border-[#1F2130] bg-[#0B0C12]"
const barClass = "flex min-h-12 items-center justify-between gap-2 border-b border-[#1A1C26] bg-[#0E0F15] px-3 py-1.5"

// One fenced block, with its title (or language) as a label
export function DocCodeBlock({ code, label }: { code: string; label?: string | null }) {
  return (
    <div className={boxClass}>
      <div className={barClass}>
        <span className="px-2 text-[12px] text-[#8B90A0]">{label ?? "Code"}</span>
        <CopyButton text={code} label="Copy" ariaLabel={label ? `Copy ${label} code` : "Copy code"} />
      </div>
      <CodeLines code={code} />
    </div>
  )
}

// Consecutive titled blocks as one box with a tab per block
export function DocCodeTabs({ tabs }: { tabs: CodeTab[] }) {
  const { language, choose } = useContext(LanguageContext)
  const base = useId()
  const buttons = useRef<(HTMLButtonElement | null)[]>([])
  const found = tabs.findIndex((t) => t.title === language)
  const index = found >= 0 ? found : 0
  const current = tabs[index]

  // Arrow keys move between tabs (the WAI-ARIA tabs pattern); Home and End jump to the ends
  const onKeyDown = (event: KeyboardEvent<HTMLButtonElement>) => {
    const last = tabs.length - 1
    const next =
      event.key === "ArrowRight" ? (index === last ? 0 : index + 1)
      : event.key === "ArrowLeft" ? (index === 0 ? last : index - 1)
      : event.key === "Home" ? 0
      : event.key === "End" ? last
      : null
    if (next === null) return
    event.preventDefault()
    choose(tabs[next].title)
    buttons.current[next]?.focus()
  }

  return (
    <div className={boxClass}>
      <div className={barClass}>
        <div role="tablist" aria-label="Language" className="flex min-w-0 gap-1 overflow-x-auto">
          {tabs.map((tab, i) => {
            const selected = i === index
            return (
              <button
                key={tab.title}
                ref={(el) => {
                  buttons.current[i] = el
                }}
                type="button"
                role="tab"
                id={`${base}-tab-${i}`}
                aria-selected={selected}
                aria-controls={`${base}-panel`}
                tabIndex={selected ? 0 : -1}
                onClick={() => choose(tab.title)}
                onKeyDown={onKeyDown}
                className={cn(
                  "min-h-8 shrink-0 rounded-lg px-3 text-[13px] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#60A5FA] max-md:min-h-11 pointer-coarse:min-h-11",
                  selected ? "bg-[#1A1C26] text-foreground" : "text-[#8B90A0] hover:text-foreground"
                )}
              >
                {tab.title}
              </button>
            )
          })}
        </div>
        <CopyButton text={current.code} label="Copy" ariaLabel={`Copy ${current.title} code`} />
      </div>
      <div role="tabpanel" id={`${base}-panel`} aria-labelledby={`${base}-tab-${index}`}>
        <CodeLines code={current.code} />
      </div>
    </div>
  )
}
