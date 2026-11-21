import { exactUsd, formatUsd } from "@/lib/format"

// A dollar amount: the display rule (formatUsd), with the exact amount on hover
export default function Usd({ value, className }: { value: number | null | undefined; className?: string }) {
  if (value == null) return <span className={className}>—</span>
  return (
    <span className={className} title={exactUsd(value)}>
      {formatUsd(value)}
    </span>
  )
}
