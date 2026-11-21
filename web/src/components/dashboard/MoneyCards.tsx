import type { ReactNode } from "react"
import Usd from "@/components/common/Usd"
import { formatPercent } from "@/lib/format"
import StatCard from "./StatCard"

// The two money cards Overview and My usage share, so both read the same way

// What was spent, with "≈ $X per 1,000 requests" for scale (left out when there are no requests)
export function SpentCard({ label, cost, requests, note }: { label: string; cost: number; requests: number; note?: string }) {
  const perThousand = requests > 0 ? (cost / requests) * 1000 : null
  const parts: ReactNode[] = []
  if (note) parts.push(note)
  if (perThousand !== null)
    parts.push(
      <>
        ≈ <Usd value={perThousand} /> per 1,000 requests
      </>
    )
  return (
    <StatCard
      label={label}
      value={<Usd value={cost} />}
      note={parts.length ? parts.map((part, i) => <span key={i}>{i > 0 && " · "}{part}</span>) : undefined}
    />
  )
}

// The saving as a share ("32.3% less"), with the dollar amount in the note
export function SavedCard({ label, share, saved }: { label: string; share: number | null; saved: number }) {
  return (
    <StatCard
      label={label}
      value={share == null ? "—" : `${formatPercent(share, 1)} less`}
      tone="good"
      note={
        <>
          <Usd value={saved} /> vs always the strongest model
        </>
      }
    />
  )
}
