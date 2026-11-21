import { useState } from "react"
import { benchmark, curveAt } from "@/lib/benchmark"
import { formatPercent } from "@/lib/format"
import SectionHeading from "./SectionHeading"

// The dial is the quality setting (tau, shown as a percent). Everything it shows is the benchmark's real result at
// that setting (src/content/benchmark.json curve), so it can't contradict the Results section.
const MIN_BAR = 50
const MAX_BAR = 95
const PRESETS = benchmark.presets

function trafficMix(bar: number) {
  const mix = curveAt(bar / 100).mix ?? { low: 0, medium: 0, high: 0 }
  const low = Math.round(mix.low * 100)
  const high = Math.round(mix.high * 100)
  return { low, mid: Math.max(0, 100 - low - high), high }
}

// Named after the nearest preset: halfway between two presets is where one name gives way to the next
function dialLabel(bar: number): string {
  const tau = bar / 100
  if (tau <= (PRESETS.cheap.tau! + PRESETS.balanced.tau!) / 2) return "Cheapest"
  if (tau <= (PRESETS.balanced.tau! + PRESETS.high.tau!) / 2) return "Balanced"
  return "Best quality"
}

const BARS = [
  { key: "low", label: "Cheapest model", text: "text-[#4ADE80]", fill: "bg-[#22C55E]" },
  { key: "mid", label: "Mid-range model", text: "text-[#60A5FA]", fill: "bg-[#3B82F6]" },
  { key: "high", label: "Most powerful model", text: "text-[#FBBF24]", fill: "bg-[#F59E0B]" },
] as const

export default function QualityDialDemo() {
  const [bar, setBar] = useState(Math.round(benchmark.headline.tau * 100))
  const mix = trafficMix(bar)
  const point = curveAt(bar / 100)

  return (
    <section className="relative px-6 pb-[120px]">
      {/* Soft light behind the card */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute -right-[120px] -top-20 h-[560px] w-[760px] rounded-full"
        style={{ background: "radial-gradient(closest-side, rgba(37,99,235,0.18), rgba(7,7,11,0))" }}
      />

      <div className="relative mx-auto flex max-w-6xl flex-wrap items-center gap-14">
        <div className="min-w-0 flex-[1_1_400px]">
          <SectionHeading
            eyebrow="The quality dial"
            title="You choose the trade-off."
            description="One number decides how sure Triage must be before trusting a cheaper model. Slide it and watch where traffic goes."
          />
        </div>

        <div className="min-w-0 flex-[1_1_460px] rounded-3xl border border-[#1F2130] bg-[#0E0F15] p-8">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <label htmlFor="landing-dial" className="text-[15px] font-semibold">
              Quality bar
            </label>
            <span className="rounded-full bg-[#2563EB]/15 px-3 py-1 text-[13px] font-medium text-[#93C5FD]">
              {dialLabel(bar)} · {bar}%
            </span>
          </div>

          <input
            id="landing-dial"
            type="range"
            min={MIN_BAR}
            max={MAX_BAR}
            step={1}
            value={bar}
            onChange={(e) => setBar(Number(e.target.value))}
            className="mt-6 h-7 w-full cursor-pointer accent-[#3B82F6]"
          />
          <div className="flex justify-between text-xs text-[#8B90A0]">
            <span>Save money</span>
            <span>Best answers</span>
          </div>

          <div className="mt-7 space-y-4">
            {BARS.map((b) => (
              <div key={b.key}>
                <div className="flex justify-between text-[13px]">
                  <span className={`font-medium ${b.text}`}>{b.label}</span>
                  <span className="text-[#C9CDD8]">{mix[b.key]}%</span>
                </div>
                <div className="mt-2 h-2 overflow-hidden rounded-full bg-[#1A1C26]">
                  <div
                    className={`h-full rounded-full transition-[width] duration-300 ${b.fill}`}
                    style={{ width: `${mix[b.key]}%` }}
                  />
                </div>
              </div>
            ))}
          </div>

          <p className="mt-6 text-[13px] text-[#C9CDD8]">
            Keeps <strong className="text-foreground">{formatPercent(point.quality_kept)}</strong> of the strongest
            model's quality at <strong className="text-[#4ADE80]">{formatPercent(point.cost_saved)}</strong> lower cost.
          </p>
          <p className="mt-1.5 text-xs text-[#8B90A0]">
            Real results at each setting, measured on {benchmark.split.n} held-out test questions.
          </p>
        </div>
      </div>
    </section>
  )
}