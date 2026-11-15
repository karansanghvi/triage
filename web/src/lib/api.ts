import type { DemoStatus, Message, RoutingInfo, Tier, RequestMode } from "@/types/chat"
import { TriageError } from "@/lib/errors"
import { qualityToTau } from "@/lib/quality"

const BASE_URL = import.meta.env.VITE_TRIAGE_URL ?? "http://localhost:8000"
const DEFAULT_MAX_PROMPT_CHARS = 8000

// ---------- What Triage sends back (snake_case, as in the API) ----------

type ApiTriage = {
  request_id: string
  model: string
  tier: Tier
  p_low: number | null
  p_medium: number | null
  tau: number | null
  reason: string
  cost: number
  strong_cost: number | null
  saved: number | null
  cache: string
  fallback: string
  mode: RequestMode
}

type ApiChatResponse = {
  choices: { message: { content: string } }[]
  triage: ApiTriage
}

type ApiError = {
  error?: { code?: string; message?: string }
}

type ApiDemoStatus = {
  enabled: boolean
  paused?: boolean
  visitor?: { limit: number; used: number; remaining: number }
  limits?: { max_prompt_chars: number; max_tokens: number }
  resets_at?: string
}

export type DemoHeaders = {
  remaining: number
  limit: number
  resetsAt: string | null
}

export type ChatResult = {
  content: string
  routing: RoutingInfo
  demo: DemoHeaders | null
}

// ---------- Helpers ----------

type ApiMessage = { role: "user" | "assistant"; content: string }

// Only real, finished turns go to Triage: no "thinking" placeholders, no errors
function toApiMessages(messages: Message[]): ApiMessage[] {
  const result: ApiMessage[] = []
  for (const m of messages) {
    if (m.role === "user") result.push({ role: "user", content: m.content })
    else if (m.status === "done") result.push({ role: "assistant", content: m.content })
  }
  return result
}

// The demo counts every message toward its prompt limit, so drop the oldest turns first.
// The latest message is always kept.
function trimToLimit(messages: ApiMessage[], maxChars: number): ApiMessage[] {
  const trimmed = [...messages]
  const total = () => trimmed.reduce((sum, m) => sum + m.content.length, 0)
  while (trimmed.length > 1 && total() > maxChars) {
    trimmed.shift()
  }
  // A conversation should start with the user, not with an orphaned answer
  while (trimmed.length > 1 && trimmed[0].role === "assistant") {
    trimmed.shift()
  }
  return trimmed
}

function readDemoHeaders(headers: Headers): DemoHeaders | null {
  const remaining = headers.get("X-Triage-Demo-Remaining")
  const limit = headers.get("X-Triage-Demo-Limit")
  if (remaining === null || limit === null) return null
  return {
    remaining: Number(remaining),
    limit: Number(limit),
    resetsAt: headers.get("X-Triage-Demo-Resets"),
  }
}

function toRoutingInfo(t: ApiTriage): RoutingInfo {
  return {
    requestId: t.request_id,
    model: t.model,
    tier: t.tier,
    pLow: t.p_low,
    pMedium: t.p_medium,
    tau: t.tau,
    reason: t.reason,
    cost: t.cost,
    strongCost: t.strong_cost,
    saved: t.saved,
    cacheHit: t.cache === "hit",
    fellBack: t.fallback !== "none",
    mode: t.mode,
  }
}

// ---------- Public functions ----------

export async function sendChat(
  messages: Message[],
  quality: number,
  maxPromptChars: number = DEFAULT_MAX_PROMPT_CHARS
): Promise<ChatResult> {
  const body = {
    model: "auto",
    messages: trimToLimit(toApiMessages(messages), maxPromptChars),
  }

  let response: Response
  try {
    response = await fetch(`${BASE_URL}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Triage-Quality": qualityToTau(quality).toFixed(2),
      },
      body: JSON.stringify(body),
    })
  } catch {
    throw new TriageError("network_error", "Couldn't reach Triage. Check your connection and try again.")
  }

  if (!response.ok) {
    const data = (await response.json().catch(() => ({}))) as ApiError
    const retryAfter = response.headers.get("Retry-After")
    throw new TriageError(
      data.error?.code ?? `http_${response.status}`,
      data.error?.message ?? "Something went wrong. Please try again.",
      retryAfter ? Number(retryAfter) : undefined
    )
  }

  const data = (await response.json()) as ApiChatResponse
  return {
    content: data.choices[0]?.message.content ?? "",
    routing: toRoutingInfo(data.triage),
    demo: readDemoHeaders(response.headers),
  }
}

export async function getDemoStatus(): Promise<DemoStatus | null> {
  try {
    const response = await fetch(`${BASE_URL}/v1/demo/status`)
    if (!response.ok) return null
    const data = (await response.json()) as ApiDemoStatus
    if (!data.enabled || !data.visitor) return null
    return {
      enabled: true,
      paused: data.paused ?? false,
      used: data.visitor.used,
      remaining: data.visitor.remaining,
      limit: data.visitor.limit,
      resetsAt: data.resets_at ?? "",
    }
  } catch {
    return null
  }
}