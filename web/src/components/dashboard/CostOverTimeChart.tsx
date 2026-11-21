import { Area, AreaChart, CartesianGrid, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import type { DotItemDotProps } from "recharts/types/util/types"
import type { SeriesPoint } from "@/lib/dashboard"
import { exactUsd, formatBucket, formatUsd } from "@/lib/format"

type Props = {
  series: SeriesPoint[]
  bucket: "day" | "month"
  ownLabel: string // e.g. "Triage" or "You, with Triage"
}

// A small dot only where the bucket had requests, so a single busy day reads as one point, not a hill
function dotWhenBusy(color: string) {
  return function BusyDot({ cx, cy, index, payload, dataKey }: DotItemDotProps) {
    const key = `${String(dataKey)}-${index}`
    if (!payload?.requests || cx == null || cy == null) return <g key={key} />
    return <circle key={key} cx={cx} cy={cy} r={3} fill={color} stroke="#12131A" strokeWidth={1.5} />
  }
}

export default function CostOverTimeChart({ series, bucket, ownLabel }: Props) {
  const data = series.map((p) => ({ ...p, label: formatBucket(p.start, bucket) }))

  return (
    <div>
      <div className="mt-3.5 flex gap-[18px] text-[13px] text-[#C9CDD8]">
        <span className="flex items-center gap-2">
          <span aria-hidden="true" className="h-[3px] w-3.5 rounded-sm bg-[#60A5FA]" />
          {ownLabel}
        </span>
        <span className="flex items-center gap-2">
          <span aria-hidden="true" className="w-3.5 border-t-2 border-dashed border-[#8B90A0]" />
          Always strongest
        </span>
      </div>
      <div className="mt-2 h-64" role="img" aria-label={`Cost over time: ${ownLabel} compared with always using the strongest model`}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="costArea" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0" stopColor="#3B82F6" stopOpacity={0.25} />
                <stop offset="1" stopColor="#3B82F6" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="#1A1C26" vertical={false} />
            <XAxis dataKey="label" tick={{ fill: "#8B90A0", fontSize: 11 }} axisLine={false} tickLine={false} minTickGap={24} />
            <YAxis
              tick={{ fill: "#8B90A0", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
              width={64}
              tickFormatter={(v: number) => formatUsd(v)}
            />
            <Tooltip
              contentStyle={{ background: "#12131A", border: "1px solid #2A2D3D", borderRadius: 12, fontSize: 12 }}
              labelStyle={{ color: "#F5F7FA" }}
              formatter={(value, name) => [
                exactUsd(typeof value === "number" ? value : Number(value)),
                name === "cost_usd" ? ownLabel : "Always strongest",
              ]}
            />
            <Area
              type="linear"
              dataKey="cost_usd"
              stroke="#60A5FA"
              strokeWidth={2.5}
              fill="url(#costArea)"
              dot={dotWhenBusy("#60A5FA")}
              activeDot={{ r: 4 }}
            />
            <Line
              type="linear"
              dataKey="strong_model_cost_usd"
              stroke="#8B90A0"
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={dotWhenBusy("#8B90A0")}
              activeDot={{ r: 4 }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}