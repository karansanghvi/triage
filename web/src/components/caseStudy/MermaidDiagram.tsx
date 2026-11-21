import { useEffect, useId, useRef, useState } from "react"
import { Maximize2, X } from "lucide-react"

// A ```mermaid block drawn as a diagram. The mermaid library is loaded only when a diagram is on screen, as its own
// chunk, so no other page pays for it. If it fails to load or to draw, the diagram's source is shown instead.

type MermaidApi = typeof import("mermaid").default

let mermaidPromise: Promise<MermaidApi> | null = null

function loadMermaid(): Promise<MermaidApi> {
  mermaidPromise ??= import("mermaid").then(({ default: mermaid }) => {
    mermaid.initialize({
      startOnLoad: false,
      securityLevel: "strict",
      // Mermaid 12 defaults to the ELK layout, a ~440 kB (gzip) download; the classic layout draws these diagrams
      layout: "dagre",
      theme: "base",
      darkMode: true,
      fontFamily: "Poppins, ui-sans-serif, sans-serif",
      // Natural width: wide diagrams scroll inside their box on phones instead of shrinking to unreadable text
      flowchart: { useMaxWidth: false, htmlLabels: true, curve: "basis", padding: 12 },
      sequence: { useMaxWidth: false, mirrorActors: false },
      themeVariables: {
        fontSize: "14px",
        background: "#0B0C12",
        primaryColor: "#13203B",
        primaryTextColor: "#F5F7FA",
        primaryBorderColor: "#3B82F6",
        secondaryColor: "#111320",
        tertiaryColor: "#0E0F15",
        lineColor: "#60A5FA",
        textColor: "#C9CDD8",
        mainBkg: "#13203B",
        nodeBorder: "#3B82F6",
        clusterBkg: "#0E0F15",
        clusterBorder: "#2A2D3D",
        titleColor: "#C9CDD8",
        edgeLabelBackground: "#0B0C12",
        actorBkg: "#13203B",
        actorBorder: "#3B82F6",
        actorTextColor: "#F5F7FA",
        actorLineColor: "#3B4058",
        signalColor: "#93C5FD",
        signalTextColor: "#C9CDD8",
        labelBoxBkgColor: "#13203B",
        labelBoxBorderColor: "#3B82F6",
        labelTextColor: "#F5F7FA",
        loopTextColor: "#C9CDD8",
        noteBkgColor: "#1A1C26",
        noteBorderColor: "#2A2D3D",
        noteTextColor: "#C9CDD8",
        activationBkgColor: "#1A2A4A",
        activationBorderColor: "#3B82F6",
        sequenceNumberColor: "#07070B",
      },
    })
    return mermaid
  })
  return mermaidPromise
}

export function CodeBlock({ code, label }: { code: string; label?: string }) {
  return (
    <div className="mt-6 overflow-hidden rounded-2xl border border-[#1F2130] bg-[#0B0C12]">
      {label && (
        <div className="border-b border-[#1A1C26] bg-[#0E0F15] px-5 py-2 text-[12px] text-[#8B90A0]">{label}</div>
      )}
      <div className="overflow-x-auto">
        <pre className="px-5 py-5 font-mono text-[13px] leading-[1.8] text-[#C9CDD8]">
          <code>{code}</code>
        </pre>
      </div>
    </div>
  )
}

// A short accessible name from the diagram's type
function labelFor(source: string): string {
  const kind = source.trim().split(/\s/)[0]
  if (kind === "flowchart" || kind === "graph") return "Architecture diagram"
  if (kind === "sequenceDiagram") return "Sequence diagram of one request"
  if (kind === "gantt") return "Timeline chart"
  return "Diagram"
}

// Big diagrams are drawn smaller to fit, but never below this share of their natural size, so their text stays
// readable; past that they scroll sideways inside their box. "Expand" shows them at full size.
const MIN_SCALE = 0.6

// Mermaid sets a fixed width and height on the SVG; size it from its viewBox instead, within the limits above
function fitSvg(svg: SVGSVGElement, scale: "fit" | "full") {
  const natural = Number(svg.getAttribute("width")) || svg.viewBox.baseVal?.width || 0
  svg.removeAttribute("height")
  svg.style.height = "auto"
  svg.style.display = "block"
  svg.style.margin = "0 auto"
  if (!natural) return
  if (scale === "full") {
    svg.style.width = `${natural}px`
    svg.style.maxWidth = "none"
  } else {
    svg.style.width = `max(100%, ${Math.round(natural * MIN_SCALE)}px)`
    svg.style.maxWidth = `${natural}px`
  }
}

function ExpandedDiagram({ svg, label, onClose }: { svg: string; label: string; onClose: () => void }) {
  const box = useRef<HTMLDivElement>(null)
  const close = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    if (box.current) {
      box.current.innerHTML = svg
      const el = box.current.querySelector("svg")
      if (el) fitSvg(el, "full")
    }
    close.current?.focus()
    const previous = document.body.style.overflow
    document.body.style.overflow = "hidden"
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose()
    window.addEventListener("keydown", onKey)
    return () => {
      document.body.style.overflow = previous
      window.removeEventListener("keydown", onKey)
    }
  }, [svg, onClose])

  return (
    <div role="dialog" aria-modal="true" aria-label={label} className="fixed inset-0 z-[60] flex flex-col bg-[#07070B]/95 backdrop-blur-sm">
      <div className="flex items-center justify-between gap-4 border-b border-[#1A1C26] px-4 py-3 sm:px-6">
        <p className="text-sm font-medium text-[#C9CDD8]">{label}</p>
        <button
          ref={close}
          type="button"
          onClick={onClose}
          className="inline-flex min-h-10 items-center gap-2 rounded-[10px] px-3 text-sm text-[#C9CDD8] transition-colors hover:bg-white/5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#60A5FA] max-md:min-h-11"
        >
          <X aria-hidden="true" className="h-4 w-4" /> Close
        </button>
      </div>
      <div className="min-h-0 flex-1 overflow-auto p-6">
        <div ref={box} role="img" aria-label={label} />
      </div>
    </div>
  )
}

export default function MermaidDiagram({ source }: { source: string }) {
  const label = labelFor(source)
  const id = `mermaid-${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`
  const box = useRef<HTMLDivElement>(null)
  const [state, setState] = useState<"loading" | "done" | "failed">("loading")
  const [svg, setSvg] = useState("")
  const [expanded, setExpanded] = useState(false)

  useEffect(() => {
    let cancelled = false
    loadMermaid()
      .then((mermaid) => mermaid.render(id, source))
      .then(({ svg, bindFunctions }) => {
        if (cancelled || !box.current) return
        box.current.innerHTML = svg
        bindFunctions?.(box.current)
        const el = box.current.querySelector("svg")
        if (el) fitSvg(el, "fit")
        setSvg(svg)
        setState("done")
      })
      .catch(() => {
        // A failed render can leave mermaid's temporary element behind
        document.getElementById(`d${id}`)?.remove()
        if (!cancelled) setState("failed")
      })
    return () => {
      cancelled = true
    }
  }, [id, source])

  if (state === "failed") return <CodeBlock code={source} label="Diagram (source)" />

  return (
    <div className="mt-6 overflow-hidden rounded-2xl border border-[#1F2130] bg-[#0B0C12]">
      {state === "done" && (
        <div className="flex items-center justify-between gap-3 border-b border-[#1A1C26] bg-[#0E0F15] px-4 py-1.5">
          <span className="text-[12px] text-[#8B90A0]">{label}</span>
          <button
            type="button"
            onClick={() => setExpanded(true)}
            className="inline-flex min-h-8 items-center gap-1.5 rounded-lg px-2.5 text-[12px] text-[#A7ACBA] transition-colors hover:bg-white/5 hover:text-[#F5F7FA] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#60A5FA] max-md:min-h-11"
          >
            <Maximize2 aria-hidden="true" className="h-3.5 w-3.5" /> Expand
          </button>
        </div>
      )}
      <div className="overflow-x-auto px-4 py-6">
        {state === "loading" && (
          <div className="h-40 w-full animate-pulse rounded-xl bg-white/[0.03]" aria-hidden="true" />
        )}
        {/* Mermaid writes the SVG into this box, so React never renders children inside it */}
        <div ref={box} role="img" aria-label={label} />
      </div>
      {expanded && <ExpandedDiagram svg={svg} label={label} onClose={() => setExpanded(false)} />}
    </div>
  )
}
