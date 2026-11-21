import { request } from "@/lib/auth"

/* ---------- Shared ---------- */

export type Range = "7d" | "30d" | "all"

export const RANGES: { value: Range; label: string }[] = [
  { value: "7d", label: "7 days" },
  { value: "30d", label: "30 days" },
  { value: "all", label: "All time" },
]

export type TierMix = { low: number; medium: number; high: number }

/* ---------- Benchmark (web/src/content/benchmark.json) ---------- */

export type BenchmarkPoint = {
  tau?: number
  quality_kept: number // 0-1 share of the strongest model's quality
  cost_saved: number // 0-1 share saved vs always-strongest
  cost_per_1k?: number
  correct?: number
  post_hoc?: boolean // true if it comes from the reporting-only curve
}

export type Benchmark = {
  schema: number
  classifier: string
  split: { name: string; n: number; scored: string }
  headline: {
    tau: number
    quality_kept: number
    cost_saved: number
    cost_per_1k_usd: number
    correct: number
    n: number
    mix: TierMix
    baseline: string
    provenance: string
  }
  vs_always_cheapest: {
    cost_multiple: number
    quality_kept_points_gained: number
    note: string
  }
  references: {
    always_cheapest: BenchmarkPoint
    always_medium: BenchmarkPoint
    always_strongest: BenchmarkPoint
    oracle: BenchmarkPoint
  }
  presets: { cheap: BenchmarkPoint; balanced: BenchmarkPoint; high: BenchmarkPoint }
  tier_mix_default: TierMix
  curve: {
    provenance: string
    points: { tau: number; quality_kept: number; cost_saved: number; mix?: TierMix }[]
  }
  classifiers: {
    method: string
    source: string
    rows: {
      name: string
      auc_cheap: number
      auc_medium: number
      brier_cheap?: number
      brier_medium?: number
      chosen: boolean
    }[]
  }
  judge: {
    n: number
    agreement: number
    passed_wrong_answers: number
    failed_correct_answers: number
    pass_means_right?: number
    passed_wrong_answers_when_reference_right?: number
    split: string
    source: string
  }
  dataset: {
    prompts: number
    labeled: number
    train_labeled: number
    fit: number
    validation: number
    test_labeled: number
    answers_collected: Record<string, string>
    answers_judged: Record<string, string>
    sources: Record<string, unknown>
    lineup: Record<string, unknown>
    prices: string
  }
  caveats: string[]
}

/* ---------- Live stats (Overview and My usage) ---------- */

// GET /v1/stats/public's totals (app/api/public_stats.py TOTAL_FIELDS)
export type Totals = {
  requests: number
  ok: number
  errors: number
  priced_requests: number
  cost_usd: number
  strong_model_cost_usd: number
  saved_usd: number
  saved_share: number | null // null when no request was priced against the strongest model
  cache_hits: number
  fallbacks: number
  fallback_rate: number | null // null when no request reached a model
}

export type SeriesPoint = {
  start: string // "2026-11-01" (day) or "2026-11" (month)
  requests: number
  cost_usd: number
  strong_model_cost_usd: number
  saved_usd: number
}

export type LatencyRow = {
  model: string
  provider: string
  tier: "low" | "medium" | "high" | "unknown"
  requests: number
  p50_ms: number
  p95_ms: number
}

type LiveStats = {
  range: Range
  bucket: "day" | "month"
  generated_at: string
  totals: Totals
  series: SeriesPoint[]
  tier_mix: Partial<TierMix> // shares of answered requests; a tier nobody reached is left out
  errors_by_kind: Record<string, number>
  latency_by_model: LatencyRow[]
  saved_is_estimate: boolean
  baseline: string
}

// Public Overview: either "not enough traffic yet" (only the count) or the full numbers
export type PublicStats =
  | { enough: false; range: Range; threshold: number; qualifying_requests: number; generated_at: string }
  | ({
      enough: true
      threshold: number
      qualifying_requests: number
      latency_min_requests: number // models with fewer answered requests are left out of latency_by_model
      small_talk_excluded: true
    } & LiveStats)

export type ModelUsage = {
  model: string
  provider: string
  requests: number
  cost_usd: number
  saved_usd: number
}

export type KeyUsage = {
  key_id: string
  label: string // "Website chat" for the website, else the key's name
  revoked: boolean
  requests: number
  cost_usd: number
  saved_usd: number
  last_used: string | null
}

export type AccountUsage = LiveStats & {
  by_model: ModelUsage[]
  by_key: KeyUsage[]
}

export type RequestItem = {
  request_id: string
  timestamp: string
  requested_model: string
  served_model: string | null
  tier: "low" | "medium" | "high" | null
  routed_by: "classifier" | "small_talk" | "requested_model" | null
  status: "ok" | "error"
  error_kind: string | null
  stream: boolean
  cache_hit: boolean | null
  fell_back: boolean | null
  cost_usd: number | null
  saved_usd: number | null
  latency_ms: number | null
  key_label: string
}

export type RequestPage = { data: RequestItem[]; next_cursor: string | null }

/* ---------- Fetching ---------- */

export function getPublicStats(range: Range) {
  return request<PublicStats>(`/v1/stats/public?range=${range}`, "GET")
}

export function getAccountUsage(token: string, range: Range) {
  return request<AccountUsage>(`/v1/account/usage?range=${range}`, "GET", undefined, token)
}

export function listAccountRequests(token: string, cursor?: string | null, limit = 50) {
  const params = new URLSearchParams({ limit: String(limit) })
  if (cursor) params.set("cursor", cursor)
  return request<RequestPage>(`/v1/account/requests?${params}`, "GET", undefined, token)
}