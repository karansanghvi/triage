import { useEffect, useState, type ReactNode } from "react"
import { Link, useLocation } from "react-router-dom"
import { ChartNoAxesColumn } from "lucide-react"
import { useRange } from "@/hooks/useRange"
import { getPublicStats, type PublicStats, type TierMix } from "@/lib/dashboard"
import { formatCount, formatMs, formatPercent, formatUsd } from "@/lib/format"
import DashCard from "@/components/dashboard/DashCard"
import StatCard from "@/components/dashboard/StatCard"
import TierMixBar from "@/components/dashboard/TierMixBar"
import CostOverTimeChart from "@/components/dashboard/CostOverTimeChart"
import { TierPill } from "@/components/dashboard/Pills"

type Live = Extract<PublicStats, { enough: true }>
type Waiting = Extract<PublicStats, { enough: false }>
type Result = { key: string; data?: PublicStats; failed?: boolean }

const BOX = "rounded-xl border border-[#1F2130] bg-[#0A0A0F] px-3.5 py-3"
const PRIMARY =
  "inline-flex h-10 items-center gap-1.5 rounded-[10px] bg-[#2563EB] px-4 text-[13px] font-semibold text-white transition-colors hover:bg-[#1D4ED8]"

// Error kinds from the request log, in words; anything else is summed as "Other"
const ERROR_LABELS: Record<string, string> = {
  overloaded: "Provider busy",
  timeout: "Too slow (timeout)",
  rate_limit: "Rate limited",
}

export default function OverviewPage() {
  const [range] = useRange()
  const [attempt, setAttempt] = useState(0)
  const [result, setResult] = useState<Result | null>(null)
  const key = `${range}:${attempt}`

  useEffect(() => {
    // A response for a range (or attempt) that is no longer current is ignored
    let current = true
    getPublicStats(range).then(
      (data) => current && setResult({ key, data }),
      () => current && setResult({ key, failed: true })
    )
    return () => {
      current = false
    }
  }, [range, key])

  if (!result || result.key !== key) return <LoadingView />
  if (result.failed || !result.data) return <ErrorView onRetry={() => setAttempt((a) => a + 1)} />
  if (!result.data.enough) return <WaitingView stats={result.data} />
  return <LiveView stats={result.data} />
}

/* ---------- Loading, error, not enough traffic ---------- */

function Bone({ className }: { className: string }) {
  return <div aria-hidden="true" className={`animate-pulse rounded-md bg-[#1A1C26] ${className}`} />
}

function CardBones({ className = "", children }: { className?: string; children: ReactNode }) {
  return (
    <div className={`min-w-0 rounded-[18px] border border-[#1F2130] bg-[#12131A] px-6 py-[22px] ${className}`}>
      <Bone className="h-6 w-40" />
      <Bone className="mt-1 h-5 w-full max-w-[420px]" />
      {children}
    </div>
  )
}

function LoadingView() {
  return (
    <div className="flex flex-col gap-5" aria-busy="true">
      <p className="sr-only" role="status">
        Loading live stats…
      </p>
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
        {[0, 1, 2, 3].map((i) => (
          <div key={i} className="min-w-0 rounded-[18px] border border-[#1F2130] bg-[#12131A] p-5">
            <Bone className="h-5 w-28" />
            <Bone className="mt-2.5 h-9 w-24" />
            <Bone className="mt-1.5 h-4 w-36" />
          </div>
        ))}
      </div>
      <CardBones>
        <Bone className="mt-3.5 h-5 w-48" />
        <Bone className="mt-2 h-64 w-full" />
      </CardBones>
      <div className="grid gap-5 md:grid-cols-2">
        <CardBones>
          <Bone className="mt-[18px] h-3.5 w-full rounded-full" />
          <Bone className="mt-3.5 h-5 w-64 max-w-full" />
        </CardBones>
        <CardBones>
          <div className="mt-4 grid grid-cols-2 gap-2.5">
            <Bone className="h-[60px] w-full rounded-xl" />
            <Bone className="h-[60px] w-full rounded-xl" />
          </div>
        </CardBones>
      </div>
      <CardBones>
        <Bone className="mt-4 h-40 w-full" />
      </CardBones>
    </div>
  )
}

function ErrorView({ onRetry }: { onRetry: () => void }) {
  return (
    <section
      role="alert"
      className="flex flex-col items-center rounded-[18px] border border-[#1F2130] bg-[#12131A] px-6 py-12 text-center"
    >
      <p className="text-base font-semibold">Couldn't load live stats.</p>
      <p className="mt-1.5 text-[13px] text-muted-foreground">Check your connection, then try again.</p>
      <button type="button" onClick={onRetry} className={`${PRIMARY} mt-5`}>
        Try again
      </button>
    </section>
  )
}

function WaitingView({ stats }: { stats: Waiting }) {
  const { search } = useLocation()
  const share = stats.threshold > 0 ? Math.min(1, stats.qualifying_requests / stats.threshold) : 0

  return (
    <section className="relative overflow-hidden rounded-[18px] border border-[#1F2130] bg-[#12131A] px-6 py-12 sm:px-8">
      <div
        aria-hidden="true"
        className="pointer-events-none absolute -top-40 left-1/2 h-[360px] w-[560px] max-w-full -translate-x-1/2 rounded-full"
        style={{ background: "radial-gradient(closest-side, rgba(37,99,235,0.16), rgba(18,19,26,0))" }}
      />
      <div className="relative mx-auto flex max-w-[460px] flex-col items-center text-center">
        <span className="flex h-12 w-12 items-center justify-center rounded-[14px] border border-[#1F2130] bg-[#0A0A0F] text-[#60A5FA]">
          <ChartNoAxesColumn aria-hidden="true" className="h-6 w-6" />
        </span>
        <h2 className="mt-5 text-xl font-semibold tracking-[-0.01em]">Live stats appear once Triage has enough traffic</h2>
        <p className="mt-2 text-sm leading-relaxed text-[#A7ACBA]">
          To keep the numbers meaningful and nobody's usage identifiable, nothing is shown until {stats.threshold}{" "}
          requests have been routed in this time range.
        </p>
        <div className="mt-6 w-full">
          <div
            role="progressbar"
            aria-label="Routed requests so far"
            aria-valuemin={0}
            aria-valuemax={stats.threshold}
            aria-valuenow={Math.min(stats.qualifying_requests, stats.threshold)}
            className="h-2 overflow-hidden rounded-full bg-[#1A1C26]"
          >
            <div className="h-full rounded-full bg-[#3B82F6]" style={{ width: `${share * 100}%` }} />
          </div>
          <p className="mt-2.5 text-[13px] text-muted-foreground">
            <strong className="text-foreground">{formatCount(stats.qualifying_requests)}</strong> of{" "}
            {formatCount(stats.threshold)} routed requests
          </p>
        </div>
        <Link to={{ pathname: "/dashboard/benchmark", search }} className={`${PRIMARY} mt-7`}>
          See the benchmark
        </Link>
      </div>
    </section>
  )
}

/* ---------- Enough traffic ---------- */

function fullMix(mix: Partial<TierMix>): TierMix {
  return { low: mix.low ?? 0, medium: mix.medium ?? 0, high: mix.high ?? 0 }
}

function errorRows(byKind: Record<string, number>): { label: string; count: number }[] {
  const totals = new Map<string, number>()
  for (const [kind, count] of Object.entries(byKind)) {
    const label = ERROR_LABELS[kind] ?? "Other"
    totals.set(label, (totals.get(label) ?? 0) + count)
  }
  return [...totals.entries()]
    .map(([label, count]) => ({ label, count }))
    .filter((r) => r.count > 0)
    .sort((a, b) => b.count - a.count || (a.label === "Other" ? 1 : b.label === "Other" ? -1 : 0))
}

function LiveView({ stats }: { stats: Live }) {
  const t = stats.totals
  const share = (part: number) => (t.requests > 0 ? part / t.requests : null)
  const errors = errorRows(stats.errors_by_kind)

  return (
    <>
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard label="Requests routed" value={formatCount(t.requests)} note="Small talk not counted" />
        <StatCard label="Money saved" value={formatUsd(t.saved_usd)} tone="good" note="vs always the strongest model" />
        <StatCard label="Saved" value={formatPercent(t.saved_share, 1)} tone="good" note="of the strongest model's cost" />
        <StatCard label="Success rate" value={formatPercent(share(t.ok), 1)} note="Requests answered" />
      </div>

      <DashCard
        title="Cost over time"
        description={`What routed requests cost ${stats.bucket === "day" ? "each day" : "each month"}, against always using the strongest model.`}
      >
        <CostOverTimeChart series={stats.series} bucket={stats.bucket} ownLabel="Triage" />
      </DashCard>

      <div className="grid gap-5 md:grid-cols-2">
        <DashCard title="Where requests went" description="Share of answered requests per model tier.">
          <TierMixBar mix={fullMix(stats.tier_mix)} />
        </DashCard>

        <DashCard title="Reliability" description="How often a backup model answered, and how often nothing did.">
          <div className="mt-4 grid grid-cols-2 gap-2.5">
            <div className={BOX}>
              <p className="text-xs text-muted-foreground">Answered after a fallback</p>
              <p className="mt-1 text-lg font-semibold">{formatPercent(t.fallback_rate, 1)}</p>
            </div>
            <div className={BOX}>
              <p className="text-xs text-muted-foreground">Failed requests</p>
              <p className="mt-1 text-lg font-semibold">{formatPercent(share(t.errors), 1)}</p>
            </div>
          </div>
          {errors.length > 0 && (
            <div className="mt-4">
              <h3 className="text-[13px] font-medium text-[#C9CDD8]">Errors by kind</h3>
              <ul className="mt-2 divide-y divide-[#1A1C26] text-[13px]">
                {errors.map((e) => (
                  <li key={e.label} className="flex items-center justify-between gap-3 py-2">
                    <span className="min-w-0 truncate text-[#C9CDD8]">{e.label}</span>
                    <span className="font-mono text-foreground">{formatCount(e.count)}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </DashCard>
      </div>

      <DashCard
        title="Response time by model"
        description={`Typical is the median; slow is the 95th percentile. Cached answers aren't counted, and models with fewer than ${stats.latency_min_requests} answers aren't shown.`}
      >
        {stats.latency_by_model.length === 0 ? (
          <p className="mt-4 text-[13px] text-muted-foreground">No model has enough answers in this range yet.</p>
        ) : (
          <div className="mt-4 overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="text-xs text-muted-foreground">
                  <th scope="col" className="whitespace-nowrap px-3.5 py-2.5 text-left font-medium">Model</th>
                  <th scope="col" className="whitespace-nowrap px-3.5 py-2.5 text-left font-medium">Tier</th>
                  <th scope="col" className="whitespace-nowrap px-3.5 py-2.5 text-right font-medium">Requests</th>
                  <th scope="col" className="whitespace-nowrap px-3.5 py-2.5 text-right font-medium">Typical</th>
                  <th scope="col" className="whitespace-nowrap px-3.5 py-2.5 text-right font-medium">Slow</th>
                </tr>
              </thead>
              <tbody>
                {stats.latency_by_model.map((m) => (
                  <tr key={m.model} className="border-t border-[#1A1C26] text-sm">
                    <td className="whitespace-nowrap px-3.5 py-3 font-mono text-[13px] text-foreground">{m.model}</td>
                    <td className="px-3.5 py-3">
                      <TierPill tier={m.tier} />
                    </td>
                    <td className="px-3.5 py-3 text-right font-mono text-[13px] text-[#C9CDD8]">{formatCount(m.requests)}</td>
                    <td className="whitespace-nowrap px-3.5 py-3 text-right font-mono text-[13px] text-[#C9CDD8]">
                      {formatMs(m.p50_ms)}
                    </td>
                    <td className="whitespace-nowrap px-3.5 py-3 text-right font-mono text-[13px] text-[#C9CDD8]">
                      {formatMs(m.p95_ms)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </DashCard>

      <p className="text-xs text-muted-foreground">Updated every few minutes.</p>
    </>
  )
}
