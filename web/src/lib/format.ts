// Costs per request are tiny, so show enough digits to be meaningful (used by the chat)
export function formatCost(dollars: number | null): string {
  if (dollars === null) return "-"
  if (dollars === 0) return "$0"
  if (dollars < 0.0001) return `$${dollars.toFixed(6)}`
  if (dollars < 0.01) return `$${dollars.toFixed(5)}`
  return `$${dollars.toFixed(2)}`
}

// Shares (0-1) as percentages.
// Default rounds to whole numbers (0.857 -> "86%"), as the chat expects.
// Pass digits for more detail (0.124, 1 -> "12.4%"), as the dashboard uses.
export function formatPercent(share: number | null | undefined, digits = 0): string {
  if (share == null) return "-"
  const value = share * 100
  if (digits === 0) return `${Math.round(value)}%`
  const fixed = value.toFixed(digits).replace(/\.?0+$/, "")
  return `${fixed}%`
}

// Money: tiny amounts keep their significant digits ($0.00039), bigger ones look normal ($12.40)
export function formatUsd(value: number | null | undefined): string {
  if (value == null) return "—"
  if (value === 0) return "$0"
  const abs = Math.abs(value)
  const sign = value < 0 ? "-" : ""
  if (abs < 0.01) return `${sign}$${Number(abs.toPrecision(2))}`
  if (abs < 1) return `${sign}$${abs.toFixed(4).replace(/0+$/, "").replace(/\.$/, "")}`
  return `${sign}$${abs.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
}

export function formatCount(n: number | null | undefined): string {
  return n == null ? "—" : n.toLocaleString()
}

// Durations: 820 -> "820 ms", 1430 -> "1.4 s"
export function formatMs(ms: number | null | undefined): string {
  if (ms == null) return "—"
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`
}

// Chart labels: "2026-11-04" -> "Nov 4", "2026-11" -> "Nov 2026"
export function formatBucket(start: string, bucket: "day" | "month"): string {
  const date = new Date(bucket === "day" ? `${start}T00:00:00Z` : `${start}-01T00:00:00Z`)
  return bucket === "day"
    ? date.toLocaleDateString([], { month: "short", day: "numeric", timeZone: "UTC" })
    : date.toLocaleDateString([], { month: "short", year: "numeric", timeZone: "UTC" })
}