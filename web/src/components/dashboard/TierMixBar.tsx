import type { TierMix } from "@/lib/dashboard"

const TIERS = [
  { key: "low", label: "Cheapest", color: "#22C55E" },
  { key: "medium", label: "Mid-range", color: "#3B82F6" },
  { key: "high", label: "Most powerful", color: "#F59E0B" },
] as const

// Works with counts or shares: both are turned into percentages that add up to 100
export default function TierMixBar({ mix }: { mix: TierMix }) {
  const total = mix.low + mix.medium + mix.high
  const pct = (v: number) => (total > 0 ? Math.round((v / total) * 100) : 0)
  const summary = TIERS.map((t) => `${t.label} ${pct(mix[t.key])}%`).join(", ")

  return (
    <div>
      <div
        role="img"
        aria-label={`Tier mix: ${summary}`}
        className="mt-[18px] flex h-3.5 gap-0.5 overflow-hidden rounded-full bg-[#1A1C26]"
      >
        {TIERS.map((t) =>
          mix[t.key] > 0 ? (
            <span key={t.key} style={{ width: `${pct(mix[t.key])}%`, background: t.color }} />
          ) : null
        )}
      </div>
      <div className="mt-3.5 flex flex-wrap gap-x-[18px] gap-y-2 text-[13px] text-[#C9CDD8]">
        {TIERS.map((t) => (
          <span key={t.key} className="flex items-center gap-2">
            <span aria-hidden="true" className="h-2.5 w-2.5 rounded-[3px]" style={{ background: t.color }} />
            {t.label} <strong className="text-foreground">{pct(mix[t.key])}%</strong>
          </span>
        ))}
      </div>
    </div>
  )
}