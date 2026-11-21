import { useState } from "react"
import { cn } from "@/lib/utils"
import { TRIAGE_API_BASE } from "@/lib/triageUrl"
import CopyButton from "./CopyButton"

type Line = { text: string; highlight?: boolean }
type Tab = "Python" | "Node" | "curl"
const TABS: Tab[] = ["Python", "Node", "curl"]

function snippets(baseUrl: string): Record<Tab, Line[]> {
  return {
    Python: [
      { text: "from openai import OpenAI" },
      { text: "" },
      { text: "client = OpenAI(" },
      { text: `    base_url="${baseUrl}",`, highlight: true },
      { text: '    api_key="trk_...",  # your Triage API key', highlight: true },
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
      { text: `  baseURL: "${baseUrl}",`, highlight: true },
      { text: '  apiKey: "trk_...", // your Triage API key', highlight: true },
      { text: "});" },
      { text: "" },
      { text: "const resp = await client.chat.completions.create({" },
      { text: '  model: "auto",' },
      { text: '  messages: [{ role: "user", content: "Explain RAG in one line" }],' },
      { text: "});" },
    ],
    curl: [
      { text: `curl ${baseUrl}/chat/completions \\`, highlight: true },
      { text: '  -H "Authorization: Bearer trk_..." \\', highlight: true },
      { text: '  -H "Content-Type: application/json" \\' },
      { text: `  -d '{"model": "auto", "messages": [{"role": "user", "content": "Hi"}]}'` },
    ],
  }
}

// A tiny highlighter: strings green, comments gray, a few keywords purple
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

type CodeSnippetProps = {
  baseUrl?: string
  className?: string
}

export default function CodeSnippet({ baseUrl = TRIAGE_API_BASE, className }: CodeSnippetProps) {
  const [tab, setTab] = useState<Tab>("Python")
  const lines = snippets(baseUrl)[tab]

  return (
    <div className={cn("overflow-hidden rounded-2xl border border-[#1F2130] bg-[#0B0C12]", className)}>
      <div className="flex items-center justify-between gap-2 border-b border-[#1A1C26] bg-[#0E0F15] px-3 py-2">
        <div role="tablist" aria-label="Language" className="flex gap-1">
          {TABS.map((name) => (
            <button
              key={name}
              type="button"
              role="tab"
              aria-selected={tab === name}
              onClick={() => setTab(name)}
              className={cn(
                "min-h-8 rounded-lg px-3 text-[13px] transition-colors max-md:min-h-11 pointer-coarse:min-h-11",
                tab === name ? "bg-[#1A1C26] text-foreground" : "text-[#8B90A0] hover:text-foreground"
              )}
            >
              {name}
            </button>
          ))}
        </div>
        <CopyButton text={lines.map((l) => l.text).join("\n")} label="Copy" ariaLabel="Copy code" />
      </div>
      <div className="overflow-x-auto">
        <pre className="py-5 font-mono text-[13px] leading-[1.8] text-[#C9CDD8]">
          {lines.map((line, i) => (
            <div
              key={i}
              className={cn("border-l-2 px-5", line.highlight ? "border-[#3B82F6] bg-[#2563EB]/14" : "border-transparent")}
            >
              {line.text ? <Highlighted text={line.text} /> : "\u00A0"}
            </div>
          ))}
        </pre>
      </div>
    </div>
  )
}