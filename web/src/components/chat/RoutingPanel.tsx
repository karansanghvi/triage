import { formatCost, formatPercent } from "@/lib/format"
import { cn } from "@/lib/utils"
import type { Message, RequestMode, RoutingInfo, Tier } from "@/types/chat"
import { Database, LifeBuoy } from "lucide-react"

type RoutingPanelProps = {
  selected: RoutingInfo | null
  messages: Message[]
}

// What each tier is called in the UI
const TIER_LABEL: Record<Tier, string> = {
  low: "Cheapest model",
  medium: "Mid-range model",
  high: "Most powerful model",
}

const TIER_SHORT: Record<Tier, string> = {
  low: "Cheapest",
  medium: "Mid-range",
  high: "Most powerful",
}

const TIER_TEXT: Record<Tier, string> = {
  low: "text-success bg-success/15",
  medium: "text-primary bg-primary/15",
  high: "text-warning bg-warning/15",
}

const TIER_BAR: Record<Tier, string> = {
  low: "bg-success",
  medium: "bg-primary",
  high: "bg-warning",
}

const MODE_LABEL: Record<RequestMode, string> = {
  demo: "Free demo",
  byok: "Using your own keys",
  api: "API",
  account: "Using your keys",
}

function savedShare(routing: RoutingInfo): number | null {
  if (routing.saved === null || !routing.strongCost) return null
  return routing.saved / routing.strongCost
}

// Turns Triage's numbers into one plain sentence
function explain(routing: RoutingInfo): string {
  const { tier, pLow, pMedium, tau } = routing
  if (tau === null || pLow === null) return routing.reason

  const needed = formatPercent(tau)
  if (tier === "low") {
    return `The cheapest model was ${formatPercent(pLow)} likely to answer well, above the ${needed} needed, so Triage used it.`
  }
  if (tier === "medium" && pMedium !== null) {
    return `The cheapest model was only ${formatPercent(pLow)} likely to answer well, below the ${needed} needed. The mid-range model was ${formatPercent(pMedium)} likely, so Triage used it.`
  }
  return `Neither cheaper model was likely enough to answer well (${needed} needed), so Triage used the most powerful model.`
}

export default function RoutingPanel({ selected, messages }: RoutingPanelProps) {
  return (
    <aside className="hidden w-80 shrink-0 flex-col overflow-hidden rounded-2xl border border-border bg-card xl:flex">
      <div className="border-b border-border px-5 py-4">
        <p className="text-sm font-semibold">How Triage answered</p>
        <p className="mt-0.5 text-xs text-muted-foreground">Triage picks the cheapest model that can answer well.</p>
      </div>

      <div className="min-h-0 flex-1 space-y-7 overflow-y-auto p-5">
        {selected ? <SelectedDetails routing={selected} /> : <EmptyState />}
      </div>

      <SessionTotals messages={messages} />
    </aside>
  )
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center py-10 text-center">
      <p className="text-sm font-medium">No answer selected</p>
      <p className="mt-1 text-xs text-muted-foreground">
        Send a message, then click the line under any answer to see how Triage chose a model for it.
      </p>
    </div>
  )
}

function SelectedDetails({ routing }: { routing: RoutingInfo }) {
  const share = savedShare(routing)
  const costWidth = routing.strongCost ? Math.max((routing.cost / routing.strongCost) * 100, 2) : 100

  return (
    <>
      {/* Answered by */}
      <section>
        <p className="text-xs text-muted-foreground">Answered by</p>
        <div className="mt-2 flex items-center justify-between gap-2">
          <p className="truncate text-base font-semibold">{routing.model}</p>
          <span className={cn("shrink-0 rounded-full px-2.5 py-1 text-xs font-medium", TIER_TEXT[routing.tier])}>
            {TIER_LABEL[routing.tier]}
          </span>
        </div>
        <p className="mt-1 text-xs text-muted-foreground">{MODE_LABEL[routing.mode]}</p>
      </section>

      {/* Savings */}
      <section>
        {share !== null && share > 0 ? (
          <>
            <p className="text-2xl font-semibold text-success">You saved {formatPercent(share)}</p>
            <p className="mt-1 text-xs text-muted-foreground">compared to always using the most powerful model</p>
          </>
        ) : (
          <>
            <p className="text-base font-semibold">No savings on this one</p>
            <p className="mt-1 text-xs text-muted-foreground">This question needed the most powerful model.</p>
          </>
        )}

        <div className="mt-4 space-y-3">
          <CostBar label="This answer" value={formatCost(routing.cost)} width={costWidth} barClass="bg-primary" />
          <CostBar
            label="With the most powerful model"
            value={formatCost(routing.strongCost)}
            width={100}
            barClass="bg-muted-foreground/40"
          />
        </div>
      </section>

      {/* Could a cheaper model handle it? */}
      {routing.tau !== null && (
        <section>
          <p className="text-xs text-muted-foreground">Could a cheaper model handle it?</p>
          <div className="mt-3 space-y-3">
            <ProbabilityBar label="Cheapest model" value={routing.pLow} tau={routing.tau} barClass={TIER_BAR.low} />
            <ProbabilityBar label="Mid-range model" value={routing.pMedium} tau={routing.tau} barClass={TIER_BAR.medium} />
          </div>
          <p className="mt-3 flex items-center gap-2 text-xs text-muted-foreground">
            <span className="inline-block h-3 w-0.5 rounded-full bg-foreground" />
            Needed: {formatPercent(routing.tau)}, set by your quality dial
          </p>
        </section>
      )}

      {/* Why this model */}
      <section>
        <p className="text-xs text-muted-foreground">Why this model</p>
        <p className="mt-2 rounded-xl bg-background px-3 py-2.5 text-xs leading-relaxed">{explain(routing)}</p>
      </section>

      {/* Extras */}
      {(routing.cacheHit || routing.fellBack) && (
        <section className="flex flex-wrap gap-2">
          {routing.cacheHit && (
            <span className="flex items-center gap-1.5 rounded-full bg-accent px-2.5 py-1 text-xs">
              <Database className="h-3 w-3" /> Reused a saved answer
            </span>
          )}
          {routing.fellBack && (
            <span className="flex items-center gap-1.5 rounded-full bg-warning/15 px-2.5 py-1 text-xs text-warning">
              <LifeBuoy className="h-3 w-3" /> First model failed, used a backup
            </span>
          )}
        </section>
      )}
    </>
  )
}

function CostBar({ label, value, width, barClass }: { label: string; value: string; width: number; barClass: string }) {
  return (
    <div>
      <div className="flex justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-medium">{value}</span>
      </div>
      <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-muted">
        <div className={cn("h-full rounded-full transition-all duration-500", barClass)} style={{ width: `${width}%` }} />
      </div>
    </div>
  )
}

function ProbabilityBar({
  label,
  value,
  tau,
  barClass,
}: {
  label: string
  value: number | null
  tau: number
  barClass: string
}) {
  const passed = value !== null && value >= tau

  return (
    <div>
      <div className="flex justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className={cn("font-medium", passed ? "text-foreground" : "text-muted-foreground")}>
          {formatPercent(value)} likely to answer well
        </span>
      </div>
      <div className="relative mt-1.5 h-1.5 rounded-full bg-muted">
        <div
          className={cn("h-full rounded-full transition-all duration-500", passed ? barClass : "bg-muted-foreground/40")}
          style={{ width: `${(value ?? 0) * 100}%` }}
        />
        <div className="absolute -top-1 h-3.5 w-0.5 rounded-full bg-foreground" style={{ left: `${tau * 100}%` }} />
      </div>
    </div>
  )
}

function SessionTotals({ messages }: { messages: Message[] }) {
  const routed: RoutingInfo[] = []
  for (const m of messages) {
    if (m.role === "assistant" && m.routing) routed.push(m.routing)
  }

  const totalCost = routed.reduce((sum, r) => sum + r.cost, 0)
  const totalStrong = routed.reduce((sum, r) => sum + (r.strongCost ?? r.cost), 0)
  const share = totalStrong > 0 ? (totalStrong - totalCost) / totalStrong : null
  const counts: Record<Tier, number> = { low: 0, medium: 0, high: 0 }
  for (const r of routed) counts[r.tier]++

  return (
    <div className="border-t border-border p-5">
      <div className="flex items-center justify-between">
        <p className="text-xs font-medium">This chat</p>
        <p className="text-xs text-muted-foreground">
          {routed.length} {routed.length === 1 ? "answer" : "answers"}
        </p>
      </div>

      {routed.length > 0 ? (
        <>
          <div className="mt-3 flex items-end justify-between gap-3">
            <div>
              <p className="text-sm font-semibold">Cost {formatCost(totalCost)}</p>
              <p className="text-xs text-muted-foreground">instead of {formatCost(totalStrong)}</p>
            </div>
            {share !== null && share >= 0.01 && (
              <p className="text-lg font-semibold text-success">{formatPercent(share)} saved</p>
            )}
          </div>
          <div className="mt-3 flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground">
            {(["low", "medium", "high"] as Tier[]).map((tier) => (
              <span key={tier} className="flex items-center gap-1">
                <span className={cn("h-1.5 w-1.5 rounded-full", TIER_BAR[tier])} />
                {TIER_SHORT[tier]} {counts[tier]}
              </span>
            ))}
          </div>
        </>
      ) : (
        <p className="mt-2 text-xs text-muted-foreground">Nothing yet.</p>
      )}
    </div>
  )
}