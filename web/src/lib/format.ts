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

// Money, one rule everywhere (dashboard, charts, tables, chat):
//   under $0.01: 2 significant digits ($0.0015, $0.00004); $0.01 and above: 2 decimals ($0.42, $1,234.50).
// Intl never switches to scientific notation, and a non-zero amount never shows as "$0.00".
// exactUsd is the unrounded amount for hover titles (up to 6 significant digits: $0.001527).
const SMALL = new Intl.NumberFormat("en-US", { maximumSignificantDigits: 2 })
const NORMAL = new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })
const EXACT = new Intl.NumberFormat("en-US", { maximumSignificantDigits: 6 })

function withSign(value: number, digits: string): string {
  return `${value < 0 ? "-" : ""}$${digits}`
}

export function formatUsd(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—"
  if (value === 0) return "$0"
  const abs = Math.abs(value)
  return withSign(value, abs < 0.01 ? SMALL.format(abs) : NORMAL.format(abs))
}

export function exactUsd(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—"
  if (value === 0) return "$0"
  return withSign(value, EXACT.format(Math.abs(value)))
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