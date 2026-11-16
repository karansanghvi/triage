import { useEffect, useState } from "react"
import { Split } from "lucide-react"
import { cn } from "@/lib/utils"

type Scene = {
  prompt: string
  probs: string
  result: string
  winner: 0 | 1 | 2 // which model node lights up
}

const SCENES: Scene[] = [
  {
    prompt: "What is 15% of 80?",
    probs: "Cheapest 92% likely to answer well · needed 86%",
    result: "Cheapest model answered · saved 75%",
    winner: 0,
  },
  {
    prompt: "Explain how a hash map works",
    probs: "Cheapest 45% · Mid-range 88% · needed 86%",
    result: "Mid-range model answered · saved 50%",
    winner: 1,
  },
  {
    prompt: "Implement an LRU cache with per-key TTL",
    probs: "Cheapest 28% · Mid-range 72% · needed 86%",
    result: "Needed the most powerful model",
    winner: 2,
  },
]

const MODELS = [
  { tier: "Cheapest", model: "groq-allam", color: "text-[#4ADE80]" },
  { tier: "Mid-range", model: "claude-sonnet", color: "text-[#60A5FA]" },
  { tier: "Most powerful", model: "gpt-6.1-sol", color: "text-[#FBBF24]" },
]

// Curves from the Triage node down to each model (in the card's 500 x 460 drawing space)
const BRANCHES = [
  "M250 246 C250 300, 90 300, 90 352",
  "M250 246 L250 352",
  "M250 246 C250 300, 410 300, 410 352",
]

export default function RoutingCard() {
  const [sceneIndex, setSceneIndex] = useState(0)
  const [paused, setPaused] = useState(false)

  // Move to the next scene every 3.2 s, unless the visitor is hovering to read
  useEffect(() => {
    if (paused) return
    const timer = setInterval(() => {
      setSceneIndex((i) => (i + 1) % SCENES.length)
    }, 3200)
    return () => clearInterval(timer)
  }, [paused])

  const scene = SCENES[sceneIndex]

  return (
    <div
      role="img"
      aria-label="Example: Triage sends each prompt to the cheapest model that can answer it well."
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}
      className="relative h-[460px] w-full max-w-[500px] rounded-[28px] border border-[#1F2130] bg-gradient-to-b from-[#12131A]/95 to-[#0A0A0F]/95 shadow-[0_30px_120px_rgba(37,99,235,0.18)]"
    >
      <div aria-hidden="true">
        {/* Connecting lines */}
        <svg viewBox="0 0 500 460" preserveAspectRatio="none" className="absolute inset-0 h-full w-full">
          <path d="M250 112 L250 168" fill="none" stroke="#3B82F6" strokeWidth="2" className="triage-flow" />
          {BRANCHES.map((d, i) => {
            const active = i === scene.winner
            return (
              <path
                key={d}
                d={d}
                fill="none"
                stroke={active ? "#3B82F6" : "#1F2130"}
                strokeWidth={active ? 2.5 : 1.5}
                className={cn("transition-[stroke] duration-500", active && "triage-flow")}
              />
            )
          })}
        </svg>

        {/* Incoming prompt */}
        <div className="absolute left-1/2 top-10 flex w-[82%] -translate-x-1/2 items-center gap-3 rounded-[14px] border border-[#2A2D3D] bg-[#0E0F15] px-4 py-3.5">
          <span className="shrink-0 text-[11px] font-semibold uppercase tracking-[0.08em] text-[#8B90A0]">Prompt</span>
          <span key={sceneIndex} className="animate-in fade-in truncate text-sm duration-500">
            {scene.prompt}
          </span>
        </div>

        {/* Triage node */}
        <div className="absolute left-1/2 top-[168px] w-[64%] -translate-x-1/2 rounded-2xl border border-[#60A5FA]/45 bg-[#2563EB]/12 px-4 py-3.5 text-center shadow-[0_0_40px_rgba(37,99,235,0.25)]">
          <div className="flex items-center justify-center gap-2">
            <Split className="h-4 w-4 text-[#93C5FD]" />
            <span className="text-sm font-semibold">Triage</span>
          </div>
          <p key={sceneIndex} className="animate-in fade-in mt-1.5 text-xs text-[#A7ACBA] duration-500">
            {scene.probs}
          </p>
        </div>

        {/* Three models */}
        <div className="absolute inset-x-0 top-[352px] grid grid-cols-3 gap-2.5 px-4">
          {MODELS.map((m, i) => {
            const active = i === scene.winner
            return (
              <div
                key={m.tier}
                className={cn(
                  "rounded-[14px] border px-2 py-3 text-center transition-all duration-500",
                  active
                    ? "border-[#60A5FA]/70 bg-[#2563EB]/16 opacity-100 shadow-[0_0_30px_rgba(37,99,235,0.35)]"
                    : "border-[#1F2130] bg-[#0E0F15] opacity-55"
                )}
              >
                <p className={cn("text-xs font-semibold", m.color)}>{m.tier}</p>
                <p className="mt-0.5 text-[11px] text-[#A7ACBA]">{m.model}</p>
              </div>
            )
          })}
        </div>

        {/* Result */}
        <p
          key={`result-${sceneIndex}`}
          className="animate-in fade-in absolute inset-x-0 bottom-[22px] text-center text-[13px] text-[#A7ACBA] duration-500"
        >
          {scene.result}
        </p>
      </div>
    </div>
  )
}