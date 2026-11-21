import type { DemoStatus, Message, RoutingInfo, Tier, RequestMode, RoutedBy } from "@/types/chat"
import { TriageError } from "@/lib/errors"
import { qualityToTau } from "@/lib/quality"
import { LANGUAGE_INSTRUCTION, withLanguageInstruction, type ApiMessage } from "@/lib/chatMessages"
import { readChatStream, StreamFormatError } from "@/lib/sse"

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
  cost: number | null         // null on a stream's first chunk: known only at the end
  strong_cost: number | null
  saved: number | null
  cache: string
  fallback: string
  mode: RequestMode
  routed_by?: RoutedBy | null
}

// What GET /v1/requests/{id} says about a finished (or stopped) request
type ApiRequestCost = {
  status?: string
  price_as_cost?: number | null
  strong_cost?: number | null
  saved?: number | null
  tokens_estimated?: boolean | null
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
  truncated: boolean
}

// ---------- Helpers ----------

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
    routedBy: t.routed_by ?? undefined,
  }
}

// The routing from the response headers: for a stream whose chunks carried no "triage" object (a server from
// before streamed responses had one). Cost isn't in the headers of a stream, so it stays unknown.
function routingFromHeaders(headers: Headers): RoutingInfo | null {
  const requestId = headers.get("X-Triage-Request-Id")
  const model = headers.get("X-Triage-Model")
  const tier = headers.get("X-Triage-Tier") as Tier | null
  if (!requestId || !model || !tier) return null
  const num = (name: string) => {
    const value = headers.get(name)
    return value === null || value === "" ? null : Number(value)
  }
  return {
    requestId,
    model,
    tier,
    pLow: num("X-Triage-P-Low"),
    pMedium: num("X-Triage-P-Medium"),
    tau: num("X-Triage-Tau"),
    reason: headers.get("X-Triage-Reason") ?? "",
    cost: null,
    strongCost: null,
    saved: null,
    cacheHit: headers.get("X-Triage-Cache") === "hit",
    fellBack: (headers.get("X-Triage-Fallback") ?? "none") !== "none",
    mode: (headers.get("X-Triage-Mode") ?? "demo") as RequestMode,
    routedBy: (headers.get("X-Triage-Routed-By") as RoutedBy | null) ?? undefined,
  }
}

// ---------- Public functions ----------

export type SendChatOptions = {
  maxPromptChars?: number
  sessionToken?: string | null   // given: account mode (the account's saved keys); not given: demo, as for guests
}

const TOO_LARGE_MESSAGE = "This conversation is too long to send. Start a new chat or shorten your message."


export type StreamHandlers = {
  signal?: AbortSignal
  onStart?: (routing: RoutingInfo, demo: DemoHeaders | null) => void   // routing known; cost not yet
  onDelta?: (text: string) => void                                    // the next piece of the answer
}

// Streams one answer. Resolves with the complete answer and its routing (cost included) once the stream ends.
// Rejects with a TriageError before or during the stream (a failure part-way included), or, if the signal
// aborts it, with the abort's DOMException: the caller keeps what onDelta delivered.
export async function streamChat(
  messages: Message[],
  quality: number,
  { maxPromptChars = DEFAULT_MAX_PROMPT_CHARS, sessionToken = null }: SendChatOptions = {},
  { signal, onStart, onDelta }: StreamHandlers = {}
): Promise<ChatResult> {
  // Only the demo has a prompt limit: account mode sends the whole conversation. The demo counts every message,
  // the language instruction too, so the history gets what is left.
  const history = toApiMessages(messages)
  const body = {
    model: "auto",
    stream: true,
    messages: withLanguageInstruction(
      sessionToken ? history : trimToLimit(history, maxPromptChars - LANGUAGE_INSTRUCTION.length)
    ),
  }

  let response: Response
  try {
    response = await fetch(`${BASE_URL}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Triage-Quality": qualityToTau(quality).toFixed(2),
        ...(sessionToken ? { Authorization: `Bearer ${sessionToken}` } : {}),
      },
      body: JSON.stringify(body),
      signal,
    })
  } catch (err) {
    if (signal?.aborted) throw err
    throw new TriageError("network_error", "Couldn't reach Triage. Check your connection and try again.")
  }

  if (!response.ok) {
    const data = (await response.json().catch(() => ({}))) as ApiError
    const retryAfter = response.headers.get("Retry-After")
    // An ended or unknown session gets the same 401 as a bad API key here: it means the session is over
    if (sessionToken && response.status === 401 && data.error?.code === "invalid_api_key") {
      throw new TriageError("invalid_session", "Please log in again.")
    }
    throw new TriageError(
      data.error?.code ?? `http_${response.status}`,
      data.error?.message ?? (response.status === 413 ? TOO_LARGE_MESSAGE : "Something went wrong. Please try again."),
      retryAfter ? Number(retryAfter) : undefined
    )
  }

  const demo = readDemoHeaders(response.headers)
  if (!response.body) throw new TriageError("network_error", "Couldn't read the answer. Please try again.")

  let outcome
  let startSent = false
  try {
    outcome = await readChatStream(response.body, {
      onStartTriage: (triage) => {
        startSent = true
        onStart?.(toRoutingInfo(triage as ApiTriage), demo)
      },
      onDelta: (piece) => {
        // No "triage" on the first chunk (an older server): the headers have the routing, so Stop still works
        if (!startSent) {
          startSent = true
          const routing = routingFromHeaders(response.headers)
          if (routing) onStart?.(routing, demo)
        }
        onDelta?.(piece)
      },
    })
  } catch (err) {
    if (signal?.aborted) throw err
    if (err instanceof StreamFormatError) throw new TriageError("stream_unreadable", err.message)
    throw new TriageError("network_error", "The connection dropped while the answer was arriving. Please try again.")
  }

  if (outcome.kind === "error") {
    throw new TriageError(outcome.code ?? "stream_interrupted", outcome.message ?? "The answer stopped part-way.")
  }
  // Only a connection that closed before the server finished the answer (no finish_reason chunk, no [DONE])
  if (outcome.kind === "interrupted") {
    throw new TriageError("stream_interrupted", "The answer stopped part-way. Please try again.")
  }

  // Finished. The finish chunk's "triage" has the cost; without it, keep the answer with the routing known
  // from the first chunk or the headers, and the cost unknown
  const routing = outcome.finalTriage
    ? toRoutingInfo(outcome.finalTriage as ApiTriage)
    : outcome.startTriage
      ? { ...toRoutingInfo(outcome.startTriage as ApiTriage), cost: null, strongCost: null, saved: null }
      : routingFromHeaders(response.headers)
  if (!routing) {
    throw new TriageError("stream_unreadable", "The answer arrived without its routing details. Please try again.")
  }
  return {
    content: outcome.content,
    routing,
    demo,
    truncated: outcome.finishReason === "length",
  }
}

// A stopped answer's estimated cost, which Triage logs once it has closed the model's stream: a moment after the
// stop, so this asks a few times. The demo's requests can be read without a key; an account's with its session.
// null: not available (the request was never logged, or nothing was priced).
export async function fetchStoppedCost(
  requestId: string,
  sessionToken: string | null
): Promise<{ cost: number; strongCost: number | null; saved: number | null } | null> {
  for (const wait of [250, 500, 1000, 1500, 2500]) {
    await new Promise((resolve) => setTimeout(resolve, wait))
    try {
      const response = await fetch(`${BASE_URL}/v1/requests/${encodeURIComponent(requestId)}`, {
        headers: sessionToken ? { Authorization: `Bearer ${sessionToken}` } : {},
      })
      if (response.status === 404) continue            // not logged yet
      if (!response.ok) return null
      const row = (await response.json()) as ApiRequestCost
      if (row.price_as_cost == null) return null
      return { cost: row.price_as_cost, strongCost: row.strong_cost ?? null, saved: row.saved ?? null }
    } catch {
      continue
    }
  }
  return null
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