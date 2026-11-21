import {
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"
import type { Benchmark } from "@/lib/dashboard"

// x = cost compared with always-strongest (100% = same cost), y = quality kept
type Dot = { cost: number; quality: number; name: string; color: string }

// Points are named in a legend under the chart, not beside them: several sit close together (cheapest preset,
// always mid-range, perfect router), and labels beside them overlapped and ran off the edges on phones.
function Halo({ cx, cy }: { cx?: number; cy?: number }) {
  return (
    <g>
      <circle cx={cx} cy={cy} r={11} fill="rgba(59,130,246,0.25)" />
      <circle cx={cx} cy={cy} r={6} fill="#3B82F6" stroke="#FFFFFF" strokeWidth={2} />
    </g>
  )
}

function Ring({ cx, cy }: { cx?: number; cy?: number }) {
  return <circle cx={cx} cy={cy} r={6} fill="#0A0A0F" stroke="#60A5FA" strokeWidth={2.5} />
}

function LegendMark({ kind, color }: { kind: "dot" | "halo" | "ring" | "line"; color?: string }) {
  if (kind === "line") return <span aria-hidden="true" className="h-[3px] w-4 rounded-sm bg-[#60A5FA]" />
  return (
    <svg aria-hidden="true" width="16" height="16" viewBox="0 0 16 16" className="shrink-0">
      {kind === "dot" && <circle cx="8" cy="8" r="5" fill={color} />}
      {kind === "halo" && (
        <>
          <circle cx="8" cy="8" r="7.5" fill="rgba(59,130,246,0.25)" />
          <circle cx="8" cy="8" r="4.5" fill="#3B82F6" stroke="#FFFFFF" strokeWidth="1.5" />
        </>
      )}
      {kind === "ring" && <circle cx="8" cy="8" r="5" fill="#0A0A0F" stroke="#60A5FA" strokeWidth="2" />}
    </svg>
  )
}

type TooltipProps = { active?: boolean; payload?: { payload?: Dot }[] }

function PointTooltip({ active, payload }: TooltipProps) {
  const dot = active ? payload?.[0]?.payload : undefined
  if (!dot) return null
  return (
    <div className="rounded-xl border border-[#2A2D3D] bg-[#12131A] px-3 py-2 text-xs">
      <p className="font-semibold text-foreground">{dot.name}</p>
      <p className="mt-0.5 text-[#C9CDD8]">
        Quality kept {dot.quality}% · cost {dot.cost}%
      </p>
    </div>
  )
}

const toDot = (p: { cost_saved: number; quality_kept: number }, name: string, color: string): Dot => ({
  cost: Math.round((1 - p.cost_saved) * 1000) / 10,
  quality: Math.round(p.quality_kept * 1000) / 10,
  name,
  color,
})

export default function CostQualityChart({ benchmark }: { benchmark: Benchmark }) {
  const curve = [...benchmark.curve.points]
    .map((p) => toDot(p, `Setting ${p.tau.toFixed(2)}`, "#60A5FA"))
    .sort((a, b) => a.cost - b.cost)

  const refs: Dot[] = [
    toDot(benchmark.references.always_cheapest, "Always cheapest", "#22C55E"),
    toDot(benchmark.references.always_medium, "Always mid-range", "#8B90A0"),
    toDot(benchmark.references.always_strongest, "Always strongest", "#F59E0B"),
    toDot(benchmark.references.oracle, "Perfect router", "#A78BFA"),
  ]
  const triage: Dot[] = [
    toDot(
      { cost_saved: benchmark.headline.cost_saved, quality_kept: benchmark.headline.quality_kept },
      "Triage · Balanced",
      "#3B82F6"
    ),
    toDot(benchmark.presets.cheap, "Triage · Cheapest", "#3B82F6"),
  ]

  const h = benchmark.headline
  const aria = `Cost versus quality on ${benchmark.split.n} test questions. On the Balanced setting Triage keeps ${(h.quality_kept * 100).toFixed(1)}% of the strongest model's quality at ${(h.cost_saved * 100).toFixed(1)}% lower cost; always using the cheapest model keeps ${(benchmark.references.always_cheapest.quality_kept * 100).toFixed(0)}%.`

  const legend = [
    { kind: "halo" as const, name: "Triage · Balanced (default)" },
    { kind: "ring" as const, name: "Triage · Cheapest" },
    { kind: "line" as const, name: "Triage at every setting" },
    ...refs.map((r) => ({ kind: "dot" as const, name: r.name, color: r.color })),
  ]

  return (
    <div className="mt-3 min-w-0">
      <div className="h-[300px] sm:h-[360px]" role="img" aria-label={aria}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart margin={{ top: 16, right: 16, left: 0, bottom: 4 }}>
            <CartesianGrid stroke="#1A1C26" />
            <XAxis
              type="number"
              dataKey="cost"
              domain={[0, 100]}
              ticks={[0, 25, 50, 75, 100]}
              tickFormatter={(v: number) => `${v}%`}
              tick={{ fill: "#8B90A0", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              type="number"
              dataKey="quality"
              domain={[40, 105]}
              ticks={[50, 60, 70, 80, 90, 100]}
              tickFormatter={(v: number) => `${v}%`}
              tick={{ fill: "#8B90A0", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
              width={44}
            />
            <Tooltip cursor={false} content={<PointTooltip />} />
            <Line data={curve} dataKey="quality" stroke="#60A5FA" strokeWidth={2.5} dot={false} activeDot={false} isAnimationActive={false} />
            <Scatter
              data={refs}
              isAnimationActive={false}
              shape={(props: { cx?: number; cy?: number; payload?: Dot }) => (
                <circle cx={props.cx} cy={props.cy} r={6} fill={props.payload?.color} />
              )}
            />
            <Scatter data={[triage[1]]} isAnimationActive={false} shape={(props: { cx?: number; cy?: number }) => <Ring {...props} />} />
            <Scatter data={[triage[0]]} isAnimationActive={false} shape={(props: { cx?: number; cy?: number }) => <Halo {...props} />} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-1 text-center text-[11px] text-[#8B90A0]">
        Across: cost, compared with always using the strongest model. Up: share of its quality kept.
      </p>
      <ul className="mt-4 flex flex-wrap gap-x-5 gap-y-2 text-[13px] text-[#C9CDD8]" aria-label="Chart key">
        {legend.map((item) => (
          <li key={item.name} className="flex items-center gap-2">
            <LegendMark kind={item.kind} color={"color" in item ? item.color : undefined} />
            {item.name}
          </li>
        ))}
      </ul>
    </div>
  )
}
