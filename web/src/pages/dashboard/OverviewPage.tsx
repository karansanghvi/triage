import { useEffect, useState } from "react"
import { Link, useLocation } from "react-router-dom"
import { ChartNoAxesColumn } from "lucide-react"
import { useRange } from "@/hooks/useRange"
import { getPublicStats, type PublicStats } from "@/lib/dashboard"
import { errorRows } from "@/lib/errorKinds"
import { formatCount, formatMs, formatPercent } from "@/lib/format"
import DashCard from "@/components/dashboard/DashCard"
import StatCard from "@/components/dashboard/StatCard"
import { SavedCard, SpentCard } from "@/components/dashboard/MoneyCards"
import TierMixBar from "@/components/dashboard/TierMixBar"
import CostOverTimeChart from "@/components/dashboard/CostOverTimeChart"
import { TierPill } from "@/components/dashboard/Pills"
import {
  Bone,
  CardBones,
  CenterCard,
  ErrorCard,
  Loading,
  PRIMARY_BUTTON,
  SMALL_BOX,
  StatCardBones,
  fullMix,
} from "@/components/dashboard/states"

type Live = Extract<PublicStats, { enough: true }>
type Waiting = Extract<PublicStats, { enough: false }>
type Result = { key: string; data?: PublicStats; failed?: boolean }

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
  if (result.failed || !result.data)
    return <ErrorCard message="Couldn't load live stats." onRetry={() => setAttempt((a) => a + 1)} />
  if (!result.data.enough) return <WaitingView stats={result.data} />
  return <LiveView stats={result.data} />
}

/* ---------- Loading, error, not enough traffic ---------- */

function LoadingView() {
  return (
    <Loading label="Loading live stats…">
      <StatCardBones />
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
    </Loading>
  )
}

function WaitingView({ stats }: { stats: Waiting }) {
  const { search } = useLocation()
  const share = stats.threshold > 0 ? Math.min(1, stats.qualifying_requests / stats.threshold) : 0

  return (
    <CenterCard icon={<ChartNoAxesColumn aria-hidden="true" />} title="Live stats appear once Triage has enough traffic">
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
      <Link to={{ pathname: "/dashboard/benchmark", search }} className={`${PRIMARY_BUTTON} mt-7`}>
        See the benchmark
      </Link>
    </CenterCard>
  )
}

/* ---------- Enough traffic ---------- */

function LiveView({ stats }: { stats: Live }) {
  const t = stats.totals
  const share = (part: number) => (t.requests > 0 ? part / t.requests : null)
  const errors = errorRows(stats.errors_by_kind)

  return (
    <>
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard label="Requests routed" value={formatCount(t.requests)} note="Small talk not counted" />
        <SpentCard label="Spent" cost={t.cost_usd} requests={t.requests} />
        <SavedCard label="Saved" share={t.saved_share} saved={t.saved_usd} />
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
            <div className={SMALL_BOX}>
              <p className="text-xs text-muted-foreground">Answered after a fallback</p>
              <p className="mt-1 text-lg font-semibold">{formatPercent(t.fallback_rate, 1)}</p>
            </div>
            <div className={SMALL_BOX}>
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
