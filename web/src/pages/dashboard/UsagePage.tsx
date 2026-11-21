import { useEffect, useRef, useState } from "react"
import { Link, useLocation } from "react-router-dom"
import { ChevronLeft, ChevronRight, Inbox, Loader2, UserRound } from "lucide-react"
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
// Recent requests has eight columns: a little less padding so it fits a desktop card without scrolling.
// Every row is the same height (h-14, room for a failed row's pill and label), so a page is always 10 rows tall.
const RTH = "whitespace-nowrap px-2.5 py-2.5 font-medium"
const RTD = "whitespace-nowrap px-2.5 py-2"
const PAGE_SIZE = 10
const PAGE_BUTTON =
  "inline-flex h-9 min-w-9 items-center justify-center gap-1 rounded-[10px] border border-[#1F2130] bg-[#0E0F15] px-2.5 text-[13px] font-medium text-[#C9CDD8] transition-colors hover:bg-[#1A1C26] hover:text-foreground disabled:pointer-events-none disabled:opacity-40"
const PAGE_CURRENT = "border-[#2A2D3D] bg-[#1A1C26] text-foreground"

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

type Pager = {
  status: "loading" | "ready" | "error"
  page: number // the page shown, from 1
  rows: RequestItem[]
  total: number
  cursors: (string | null)[] // cursors[i] opens page i + 1 (page 1: no cursor); its length = pages reachable
  pending: number | null // the page being loaded
  pageError: boolean
}

const START: Pager = { status: "loading", page: 1, rows: [], total: 0, cursors: [null], pending: null, pageError: false }

// Cursors after loading page p: the next page's cursor is new information. If it differs from the one kept
// (new requests moved page 1's boundary), the later cursors no longer line up, so they're dropped.
function cursorsAfter(cursors: (string | null)[], p: number, next: string | null): (string | null)[] {
  if (!next) return cursors.slice(0, p)
  if (cursors[p] === next) return cursors
  return [...cursors.slice(0, p), next]
}

// The page numbers to show: every reachable page (compressed around the current one when there are many),
// then the last page from the total, with "gap" for the pages in between
function pageItems(current: number, reachable: number, last: number): (number | "gap")[] {
  const shown = new Set<number>()
  if (reachable <= 7) for (let p = 1; p <= reachable; p++) shown.add(p)
  else [1, current - 1, current, current + 1, reachable].forEach((p) => p >= 1 && p <= reachable && shown.add(p))
  shown.add(last)
  const sorted = [...shown].sort((x, y) => x - y)
  const out: (number | "gap")[] = []
  sorted.forEach((p, i) => {
    if (i > 0 && p - sorted[i - 1] > 1) out.push("gap")
    out.push(p)
  })
  return out
}

function RecentRequests() {
  const api = useAccountApi()
  const [attempt, setAttempt] = useState(0)
  const [state, setState] = useState<Pager>(START)
  const latest = useRef(0) // the newest page request; older answers are ignored

  useEffect(() => {
    let current = true
    const id = ++latest.current
    api.listRequests(null, PAGE_SIZE).then(
      (res) =>
        current &&
        id === latest.current &&
        setState({ ...START, status: "ready", rows: res.data, total: res.total, cursors: cursorsAfter([null], 1, res.next_cursor) }),
      () => current && id === latest.current && setState((s) => ({ ...s, status: "error" }))
    )
    return () => {
      current = false
    }
  }, [api, attempt])

  function retry() {
    setState(START)
    setAttempt((a) => a + 1)
  }

  function goTo(p: number) {
    if (state.pending !== null || p === state.page || p < 1 || p > state.cursors.length) return
    const id = ++latest.current
    setState((s) => ({ ...s, pending: p, pageError: false }))
    api.listRequests(state.cursors[p - 1], PAGE_SIZE).then(
      (res) => {
        if (id !== latest.current) return
        setState((s) => ({
          ...s,
          page: p,
          rows: res.data,
          total: res.total,
          cursors: cursorsAfter(s.cursors, p, res.next_cursor),
          pending: null,
        }))
      },
      () => id === latest.current && setState((s) => ({ ...s, pending: null, pageError: true }))
    )
  }

  if (state.status === "loading")
    return (
      <CardBones>
        <Bone className="mt-4 h-48 w-full" />
      </CardBones>
    )
  if (state.status === "error") return <ErrorCard message="Couldn't load your recent requests." onRetry={retry} />
  if (state.rows.length === 0 && state.page === 1) return null

  const busy = state.pending !== null
  const paged = state.total > PAGE_SIZE
  const reachable = state.cursors.length
  const last = Math.max(reachable, Math.ceil(state.total / PAGE_SIZE), 1)
  const hasNext = state.page < reachable
  const first = (state.page - 1) * PAGE_SIZE + 1
  const filler = paged ? Math.max(0, PAGE_SIZE - state.rows.length) : 0

  return (
    <DashCard
      title="Recent requests"
      description="Prompts and answers are never stored, so only routing details appear here."
    >
      <div className="mt-4 overflow-x-auto">
        <table className="w-full border-collapse" aria-busy={busy}>
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
          <tbody className={cn("transition-opacity", busy && "opacity-50")}>
            {state.rows.map((r) => (
              <RequestRow key={r.request_id} r={r} />
            ))}
            {/* A short last page keeps the table 10 rows tall */}
            {Array.from({ length: filler }, (_, i) => (
              <tr key={`filler-${i}`} aria-hidden="true" className="h-14 border-t border-transparent">
                <td colSpan={8} />
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {paged && (
        <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
          <p className="text-[13px] text-muted-foreground" aria-live="polite">
            Showing {formatCount(first)}–{formatCount(first + state.rows.length - 1)} of {formatCount(state.total)}
          </p>
          <nav aria-label="Recent requests pages" className="flex flex-wrap items-center gap-1.5">
            <button
              type="button"
              aria-label="Previous page"
              disabled={busy || state.page === 1}
              onClick={() => goTo(state.page - 1)}
              className={PAGE_BUTTON}
            >
              <ChevronLeft aria-hidden="true" className="h-4 w-4" />
              <span className="hidden sm:inline">Previous</span>
            </button>
            {pageItems(state.page, reachable, last).map((item, i) =>
              item === "gap" ? (
                <span key={`gap-${i}`} aria-hidden="true" className="px-1 text-[13px] text-muted-foreground">
                  …
                </span>
              ) : (
                <button
                  key={item}
                  type="button"
                  aria-label={`Page ${item}`}
                  aria-current={item === state.page ? "page" : undefined}
                  disabled={busy || item > reachable}
                  title={item > reachable ? "Use Next to reach this page" : undefined}
                  onClick={() => goTo(item)}
                  className={cn(PAGE_BUTTON, item === state.page && PAGE_CURRENT)}
                >
                  {state.pending === item ? <Loader2 aria-hidden="true" className="h-4 w-4 animate-spin" /> : item}
                </button>
              )
            )}
            <button
              type="button"
              aria-label="Next page"
              disabled={busy || !hasNext}
              onClick={() => goTo(state.page + 1)}
              className={PAGE_BUTTON}
            >
              <span className="hidden sm:inline">Next</span>
              <ChevronRight aria-hidden="true" className="h-4 w-4" />
            </button>
          </nav>
        </div>
      )}
      {state.pageError && (
        <p role="alert" className="mt-3 text-[13px] text-[#F87171]">
          Couldn't load that page. Try again.
        </p>
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
    <tr className="h-14 border-t border-[#1A1C26] text-sm">
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
        <div className="flex flex-col items-start gap-0.5">
          <StatusPill status={r.status} fellBack={r.fell_back} />
          {r.status === "error" && <span className="text-xs text-muted-foreground">{errorLabel(r.error_kind)}</span>}
        </div>
      </td>
    </tr>
  )
}
