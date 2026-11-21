import {
  CartesianGrid,
  ComposedChart,
  LabelList,
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
      { cost_saved: benchmark.headline.cost_saved_vs_strongest, quality_kept: benchmark.headline.quality_kept },
      "Triage · Balanced",
      "#3B82F6"
    ),
    toDot(benchmark.presets.cheap, "Triage · Cheapest", "#3B82F6"),
  ]

  const h = benchmark.headline
  const aria = `Cost versus quality on ${benchmark.split.n} test questions. On the Balanced setting Triage keeps ${(h.quality_kept * 100).toFixed(1)}% of the strongest model's quality at ${(h.cost_saved_vs_strongest * 100).toFixed(1)}% lower cost; always using the cheapest model keeps ${(benchmark.references.always_cheapest.quality_kept * 100).toFixed(0)}%.`

  return (
    <div className="mt-3 h-[380px]" role="img" aria-label={aria}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart margin={{ top: 16, right: 24, left: 8, bottom: 24 }}>
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
            label={{ value: "Cost, compared with always using the strongest model", position: "bottom", offset: 8, fill: "#8B90A0", fontSize: 11 }}
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
          <Tooltip
            cursor={false}
            contentStyle={{ background: "#12131A", border: "1px solid #2A2D3D", borderRadius: 12, fontSize: 12 }}
            labelFormatter={() => ""}
            formatter={(value, key) => [`${value ?? "—"}%`, key === "cost" ? "Cost" : "Quality kept"]}
          />
          <Line data={curve} dataKey="quality" stroke="#60A5FA" strokeWidth={2.5} dot={false} isAnimationActive={false} />
          <Scatter data={refs} shape={(props: { cx?: number; cy?: number; payload?: Dot }) => (
            <circle cx={props.cx} cy={props.cy} r={6} fill={props.payload?.color} />
          )}>
            <LabelList dataKey="name" position="right" offset={10} fill="#C9CDD8" fontSize={12} />
          </Scatter>
          <Scatter data={triage} shape={(props: { cx?: number; cy?: number }) => (
            <g>
              <circle cx={props.cx} cy={props.cy} r={11} fill="rgba(59,130,246,0.25)" />
              <circle cx={props.cx} cy={props.cy} r={6} fill="#3B82F6" stroke="#FFFFFF" strokeWidth={2} />
            </g>
          )}>
            <LabelList dataKey="name" position="top" offset={14} fill="#F5F7FA" fontSize={12} fontWeight={600} />
          </Scatter>
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}