import { useState } from "react"
import { Check, Copy } from "lucide-react"
import { cn } from "@/lib/utils"
import SectionHeading from "./SectionHeading"

// Replace with the real address once Triage is deployed
const TRIAGE_URL = `${import.meta.env.VITE_TRIAGE_URL ?? "https://your-triage-domain"}/v1`

type Line = { text: string; highlight?: boolean }

const SNIPPETS: Record<"Python" | "Node" | "curl", Line[]> = {
  Python: [
    { text: "from openai import OpenAI" },
    { text: "" },
    { text: "client = OpenAI(" },
    { text: `    base_url="${TRIAGE_URL}",`, highlight: true },
    { text: '    api_key="trk_...",  # your Triage key', highlight: true },
    { text: ")" },
    { text: "" },
    { text: "resp = client.chat.completions.create(" },
    { text: '    model="auto",' },
    { text: '    messages=[{"role": "user", "content": "Explain RAG in one line"}],' },
    { text: ")" },
  ],
  Node: [
    { text: 'import OpenAI from "openai";' },
    { text: "" },
    { text: "const client = new OpenAI({" },
    { text: `  baseURL: "${TRIAGE_URL}",`, highlight: true },
    { text: '  apiKey: "trk_...", // your Triage key', highlight: true },
    { text: "});" },
    { text: "" },
    { text: "const resp = await client.chat.completions.create({" },
    { text: '  model: "auto",' },
    { text: '  messages: [{ role: "user", content: "Explain RAG in one line" }],' },
    { text: "});" },
  ],
  curl: [
    { text: `curl ${TRIAGE_URL}/chat/completions \\`, highlight: true },
    { text: '  -H "Authorization: Bearer trk_..." \\', highlight: true },
    { text: '  -H "Content-Type: application/json" \\' },
    { text: `  -d '{"model": "auto", "messages": [{"role": "user", "content": "Hi"}]}'` },
  ],
}

type Tab = keyof typeof SNIPPETS
const TABS: Tab[] = ["Python", "Node", "curl"]

// A tiny highlighter: strings in green, comments in gray, a few keywords in purple
const TOKEN = /("[^"]*"|'[^']*'|#.*$|\/\/.*$|\b(?:from|import|const|await|new)\b)/g

function Highlighted({ text }: { text: string }) {
  return (
    <>
      {text.split(TOKEN).map((part, i) => {
        if (!part) return null
        let color = ""
        if (part.startsWith('"') || part.startsWith("'")) color = "text-[#86EFAC]"
        else if (part.startsWith("#") || part.startsWith("//")) color = "text-[#6B7080]"
        else if (/^(from|import|const|await|new)$/.test(part)) color = "text-[#C084FC]"
        return (
          <span key={i} className={color}>
            {part}
          </span>
        )
      })}
    </>
  )
}

const STEPS = [
  { title: "Create an account", body: "Free, with email and password." },
  { title: "Add your provider keys", body: "Encrypted the moment you save them." },
  { title: "Use your Triage key", body: "Point your client at Triage and set the model to auto." },
]

export default function Integrate() {
  const [tab, setTab] = useState<Tab>("Python")
  const [copied, setCopied] = useState(false)
  const lines = SNIPPETS[tab]

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(lines.map((l) => l.text).join("\n"))
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      // Clipboard blocked (e.g. not on https): nothing to do
    }
  }

  return (
    <section id="integrate" className="relative scroll-mt-8 px-6 pb-[120px]">
      <div
        aria-hidden="true"
        className="pointer-events-none absolute right-[2%] top-10 h-[520px] w-[820px] rounded-full"
        style={{ background: "radial-gradient(closest-side, rgba(37,99,235,0.16), rgba(7,7,11,0))" }}
      />

      <div className="relative mx-auto flex max-w-6xl flex-wrap items-start gap-14">
        {/* Steps */}
        <div className="min-w-0 flex-[1_1_380px]">
          <SectionHeading
            eyebrow="Integrate"
            title="Two lines to switch."
            description="Triage speaks the OpenAI API. Keep your code, change where it points."
          />
          <ol className="mt-8 space-y-5">
            {STEPS.map((step, i) => (
              <li key={step.title} className="flex gap-4">
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-[10px] border border-[#2A2D3D] text-[13px] font-semibold text-[#93C5FD]">
                  {i + 1}
                </span>
                <div>
                  <p className="text-[15px] font-semibold">{step.title}</p>
                  <p className="mt-1 text-sm text-[#8B90A0]">{step.body}</p>
                </div>
              </li>
            ))}
          </ol>
        </div>

        {/* Code card */}
        <div className="min-w-0 flex-[1_1_520px] overflow-hidden rounded-[20px] border border-[#1F2130] bg-[#0B0C12] shadow-[0_30px_100px_rgba(37,99,235,0.12)]">
          <div className="flex items-center justify-between gap-2 border-b border-[#1A1C26] bg-[#0E0F15] px-3 py-2.5">
            <div role="tablist" aria-label="Language" className="flex gap-1">
              {TABS.map((name) => (
                <button
                  key={name}
                  role="tab"
                  aria-selected={tab === name}
                  onClick={() => setTab(name)}
                  className={cn(
                    "min-h-[34px] rounded-lg px-3.5 text-[13px] transition-colors",
                    tab === name ? "bg-[#1A1C26] text-foreground" : "text-[#8B90A0] hover:text-foreground"
                  )}
                >
                  {name}
                </button>
              ))}
            </div>
            <button
              onClick={handleCopy}
              aria-label="Copy code"
              className="flex min-h-[34px] items-center gap-1.5 rounded-lg px-3 text-[13px] text-[#8B90A0] transition-colors hover:text-foreground"
            >
              {copied ? <Check className="h-3.5 w-3.5 text-[#4ADE80]" /> : <Copy className="h-3.5 w-3.5" />}
              {copied ? "Copied" : "Copy"}
            </button>
          </div>

          <div className="overflow-x-auto">
            <pre className="py-6 font-mono text-sm leading-[1.8] text-[#C9CDD8]">
              {lines.map((line, i) => (
                <div
                  key={i}
                  className={cn(
                    "border-l-2 px-6",
                    line.highlight ? "border-[#3B82F6] bg-[#2563EB]/14" : "border-transparent"
                  )}
                >
                  {line.text ? <Highlighted text={line.text} /> : "\u00A0"}
                </div>
              ))}
            </pre>
          </div>
        </div>
      </div>
    </section>
  )
}