import { useEffect, useState } from "react"
import { Link, useLocation } from "react-router-dom"
import { Inbox, Loader2, UserRound } from "lucide-react"
import { useAuth } from "@/context/AuthContext"
import { useAccountApi } from "@/hooks/useAccountApi"
import { useRange } from "@/hooks/useRange"
import type { AccountUsage, KeyUsage, RequestItem } from "@/lib/dashboard"
import { errorLabel } from "@/lib/errorKinds"
import { formatCount, formatMs, formatPercent, formatUsd } from "@/lib/format"
import { relativeTime } from "@/lib/time"
import { cn } from "@/lib/utils"
import DashCard from "@/components/dashboard/DashCard"
import StatCard from "@/components/dashboard/StatCard"
import TierMixBar from "@/components/dashboard/TierMixBar"
import CostOverTimeChart from "@/components/dashboard/CostOverTimeChart"
import { StatusPill, TierPill } from "@/components/dashboard/Pills"
import {
  Bone,
  CardBones,
  CenterCard,
  ErrorCard,
  Loading,
  PRIMARY_BUTTON,
  SECONDARY_BUTTON,
  StatCardBones,
  fullMix,
} from "@/components/dashboard/states"

type Result = { key: string; data?: AccountUsage; failed?: boolean }

const TH = "whitespace-nowrap px-3.5 py-2.5 font-medium"
const TD = "whitespace-nowrap px-3.5 py-3"
const NUM = "text-right font-mono text-[13px]"
// Recent requests has eight columns: a little less padding so it fits a desktop card without scrolling
const RTH = "whitespace-nowrap px-2.5 py-2.5 font-medium"
const RTD = "whitespace-nowrap px-2.5 py-3"

export default function UsagePage() {
  const { user, loading } = useAuth()
  if (loading) return <UsageLoading />
  if (!user) return <LoggedOut />
  return <AccountUsageView />
}

/* ---------- Logged out ---------- */

function LoggedOut() {
  const { pathname, search } = useLocation()
  return (
    <CenterCard icon={<UserRound aria-hidden="true" />} title="See your own costs and savings">
      <p className="mt-2 text-sm leading-relaxed text-[#A7ACBA]">
        Create a free account to track every request your apps send through Triage: what it cost, what you saved, and
        which model answered.
      </p>
      <div className="mt-7 flex flex-wrap justify-center gap-3">
        <Link to="/signup" className={PRIMARY_BUTTON}>
          Create an account
        </Link>
        <Link to="/login" state={{ from: `${pathname}${search}` }} className={SECONDARY_BUTTON}>
          Log in
        </Link>
      </div>
    </CenterCard>
  )
}

/* ---------- Logged in ---------- */

function AccountUsageView() {
  const api = useAccountApi()
  const [range] = useRange()
  const [attempt, setAttempt] = useState(0)
  const [result, setResult] = useState<Result | null>(null)
  const key = `${range}:${attempt}`

  useEffect(() => {
    // A response for a range (or attempt) that is no longer current is ignored
    let current = true
    api.getUsage(range).then(
      (data) => current && setResult({ key, data }),
      () => current && setResult({ key, failed: true })
    )
    return () => {
      current = false
    }
  }, [api, range, key])

  let body
  if (!result || result.key !== key) body = <UsageLoading />
  else if (result.failed || !result.data)
    body = <ErrorCard message="Couldn't load your usage." onRetry={() => setAttempt((a) => a + 1)} />
  else if (result.data.totals.requests === 0) body = <EmptyPeriod />
  else body = <UsageView usage={result.data} />

  return (
    <>
      {body}
      {/* Not tied to the range: the newest requests, whenever they were */}
      <RecentRequests />
    </>
  )
}

function UsageLoading() {
  return (
    <Loading label="Loading your usage…">
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
          <Bone className="mt-4 h-28 w-full" />
        </CardBones>
      </div>
      <CardBones>
        <Bone className="mt-4 h-32 w-full" />
      </CardBones>
    </Loading>
  )
}

function EmptyPeriod() {
  return (
    <CenterCard icon={<Inbox aria-hidden="true" />} title="No requests yet in this period">
      <p className="mt-2 text-sm leading-relaxed text-[#A7ACBA]">
        Requests your apps send with your Triage API keys, and website chats on your own provider keys, show up here.
      </p>
      <Link to="/settings/get-started" className={cn(SECONDARY_BUTTON, "mt-7")}>
        Settings → Get started
      </Link>
    </CenterCard>
  )
}

function UsageView({ usage }: { usage: AccountUsage }) {
  const t = usage.totals
  const successRate = t.requests > 0 ? t.ok / t.requests : null
  // Account requests always run on the account's own saved provider keys (no keys: the API refuses them);
  // the free demo is never linked to an account, so it can't appear here.
  const spentNote = "on your own provider keys"
  const savedNote =
    t.saved_share == null
      ? "vs always the strongest model"
      : `${formatPercent(t.saved_share, 1)} vs always the strongest model`

  return (
    <>
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="Your requests"
          value={formatCount(t.requests)}
          note={t.small_talk_requests > 0 ? `${formatCount(t.small_talk_requests)} small talk, not credited` : undefined}
        />
        <StatCard label="You spent" value={formatUsd(t.cost_usd)} note={spentNote} />
        <StatCard label="You saved" value={formatUsd(t.saved_usd)} tone="good" note={savedNote} />
        <StatCard label="Success rate" value={formatPercent(successRate, 1)} note="Requests answered" />
      </div>

      <DashCard
        title="Your cost over time"
        description={`What your requests cost ${usage.bucket === "day" ? "each day" : "each month"}, against always using the strongest model.`}
      >
        <CostOverTimeChart series={usage.series} bucket={usage.bucket} ownLabel="You, with Triage" />
      </DashCard>

      <div className="grid gap-5 md:grid-cols-2">
        <DashCard title="Where your requests went" description="Share of your answered requests per model tier.">
          <TierMixBar mix={fullMix(usage.tier_mix)} />
        </DashCard>

        <DashCard title="By model" description="Which models answered, and what they cost.">
          <div className="mt-4 overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="text-xs text-muted-foreground">
                  <th scope="col" className={cn(TH, "text-left")}>Model</th>
                  <th scope="col" className={cn(TH, "text-right")}>Requests</th>
                  <th scope="col" className={cn(TH, "text-right")}>Cost</th>
                </tr>
              </thead>
              <tbody>
                {usage.by_model.map((m) => (
                  <tr key={m.model ?? "none"} className="border-t border-[#1A1C26] text-sm">
                    <td className={cn(TD, "font-mono text-[13px]", m.model ? "text-foreground" : "text-muted-foreground")}>
                      {m.model ?? "No model (failed)"}
                    </td>
                    <td className={cn(TD, NUM, "text-[#C9CDD8]")}>{formatCount(m.requests)}</td>
                    <td className={cn(TD, NUM, "text-[#C9CDD8]")}>{formatUsd(m.cost_usd)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </DashCard>
      </div>

      <DashCard title="By API key" description="Your keys, and the website chat, in this period.">
        <div className="mt-4 overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr className="text-xs text-muted-foreground">
                <th scope="col" className={cn(TH, "text-left")}>Key</th>
                <th scope="col" className={cn(TH, "text-right")}>Requests</th>
                <th scope="col" className={cn(TH, "text-right")}>Cost</th>
                <th scope="col" className={cn(TH, "text-right")}>Saved</th>
                <th scope="col" className={cn(TH, "text-right")}>Last used</th>
              </tr>
            </thead>
            <tbody>
              {usage.by_key.map((k) => (
                <KeyRow key={k.key_id} k={k} />
              ))}
            </tbody>
          </table>
        </div>
      </DashCard>
    </>
  )
}

function KeyRow({ k }: { k: KeyUsage }) {
  const revoked = k.revoked === true
  const cell = revoked ? "text-muted-foreground" : "text-[#C9CDD8]"
  return (
    <tr className="border-t border-[#1A1C26] text-sm">
      <td className={TD}>
        <span className={cn("font-semibold", revoked ? "text-muted-foreground" : "text-foreground")}>{k.label}</span>
        {k.kind === "api_key" && <span className="ml-2 font-mono text-xs text-muted-foreground">{k.key_id}</span>}
        {revoked && <span className="ml-1.5 text-xs text-muted-foreground">(revoked)</span>}
      </td>
      <td className={cn(TD, NUM, cell)}>{formatCount(k.requests)}</td>
      <td className={cn(TD, NUM, cell)}>{formatUsd(k.cost_usd)}</td>
      <td className={cn(TD, NUM, revoked ? cell : k.saved_usd > 0 ? "text-[#4ADE80]" : cell)}>
        {formatUsd(k.saved_usd)}
      </td>
      <td className={cn(TD, "text-right text-[13px]", cell)}>
        {k.last_used ? (
          <time dateTime={k.last_used} title={new Date(k.last_used).toLocaleString()}>
            {relativeTime(k.last_used)}
          </time>
        ) : (
          "—"
        )}
      </td>
    </tr>
  )
}

/* ---------- Recent requests (paged, not tied to the range) ---------- */

type Recent = {
  status: "loading" | "ready" | "error"
  items: RequestItem[]
  next: string | null
  more: "idle" | "loading" | "error"
}

function RecentRequests() {
  const api = useAccountApi()
  const [attempt, setAttempt] = useState(0)
  const [state, setState] = useState<Recent>({ status: "loading", items: [], next: null, more: "idle" })

  useEffect(() => {
    let current = true
    api.listRequests(null).then(
      (page) => current && setState({ status: "ready", items: page.data, next: page.next_cursor, more: "idle" }),
      () => current && setState((s) => ({ ...s, status: "error" }))
    )
    return () => {
      current = false
    }
  }, [api, attempt])

  function retry() {
    setState({ status: "loading", items: [], next: null, more: "idle" })
    setAttempt((a) => a + 1)
  }

  function loadMore() {
    if (!state.next || state.more === "loading") return
    const cursor = state.next
    setState((s) => ({ ...s, more: "loading" }))
    api.listRequests(cursor).then(
      (page) =>
        // Append only if nothing reset the list meanwhile (the cursor is still the one this page followed)
        setState((s) =>
          s.next === cursor ? { ...s, items: [...s.items, ...page.data], next: page.next_cursor, more: "idle" } : s
        ),
      () => setState((s) => (s.next === cursor ? { ...s, more: "error" } : s))
    )
  }

  if (state.status === "loading")
    return (
      <CardBones>
        <Bone className="mt-4 h-48 w-full" />
      </CardBones>
    )
  if (state.status === "error") return <ErrorCard message="Couldn't load your recent requests." onRetry={retry} />
  if (state.items.length === 0) return null

  return (
    <DashCard
      title="Recent requests"
      description="Prompts and answers are never stored, so only routing details appear here."
    >
      <div className="mt-4 overflow-x-auto">
        <table className="w-full border-collapse">
          <thead>
            <tr className="text-xs text-muted-foreground">
              <th scope="col" className={cn(RTH, "text-left")}>Time</th>
              <th scope="col" className={cn(RTH, "text-left")}>Model</th>
              <th scope="col" className={cn(RTH, "text-left")}>Tier</th>
              <th scope="col" className={cn(RTH, "text-right")}>Cost</th>
              <th scope="col" className={cn(RTH, "text-right")}>Saved</th>
              <th scope="col" className={cn(RTH, "text-right")}>Latency</th>
              <th scope="col" className={cn(RTH, "text-left")}>Key</th>
              <th scope="col" className={cn(RTH, "text-left")}>Status</th>
            </tr>
          </thead>
          <tbody>
            {state.items.map((r) => (
              <RequestRow key={r.request_id} r={r} />
            ))}
          </tbody>
        </table>
      </div>
      {(state.next || state.more === "error") && (
        <div className="mt-4 flex flex-wrap items-center gap-3">
          {state.next && (
            <button
              type="button"
              onClick={loadMore}
              disabled={state.more === "loading"}
              aria-busy={state.more === "loading"}
              className={SECONDARY_BUTTON}
            >
              {state.more === "loading" && <Loader2 aria-hidden="true" className="h-4 w-4 animate-spin" />}
              Load more
            </button>
          )}
          {state.more === "error" && (
            <p role="alert" className="text-[13px] text-[#F87171]">
              Couldn't load more requests. Try again.
            </p>
          )}
        </div>
      )}
    </DashCard>
  )
}

function RequestRow({ r }: { r: RequestItem }) {
  let saved
  if (r.routed_by === "small_talk") saved = <span className="text-muted-foreground">Small talk</span>
  else if (r.saved_usd == null) saved = <span className="text-muted-foreground">—</span>
  else saved = <span className={r.saved_usd > 0 ? "text-[#4ADE80]" : undefined}>{formatUsd(r.saved_usd)}</span>

  return (
    <tr className="border-t border-[#1A1C26] text-sm">
      <td className={cn(RTD, "text-[13px] text-[#C9CDD8]")}>
        <time dateTime={r.timestamp} title={new Date(r.timestamp).toLocaleString()}>
          {relativeTime(r.timestamp)}
        </time>
      </td>
      <td className={cn(RTD, "font-mono text-[13px]", r.served_model ? "text-foreground" : "text-muted-foreground")}>
        {r.served_model ?? r.requested_model}
      </td>
      <td className={RTD}>
        <TierPill tier={r.tier} />
      </td>
      <td className={cn(RTD, NUM, "text-[#C9CDD8]")}>{formatUsd(r.cost_usd)}</td>
      <td className={cn(RTD, NUM, "text-[#C9CDD8]")}>{saved}</td>
      <td className={cn(RTD, NUM, "text-[#C9CDD8]")}>{formatMs(r.latency_ms)}</td>
      <td className={cn(RTD, "max-w-[140px] truncate text-[13px] text-[#C9CDD8]")} title={r.key_label}>
        {r.key_label}
      </td>
      <td className={RTD}>
        <div className="flex flex-col items-start gap-1">
          <StatusPill status={r.status} fellBack={r.fell_back} />
          {r.status === "error" && <span className="text-xs text-muted-foreground">{errorLabel(r.error_kind)}</span>}
        </div>
      </td>
    </tr>
  )
}
