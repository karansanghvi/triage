// ---------- Routing: how Triage handled one answer ----------

export type Tier = "low" | "medium" | "high"

export type RequestMode = "api" | "byok" | "demo" | "account"

// What decided the tier: the classifier, the small-talk rule (a greeting: the cheapest model), or a named model
export type RoutedBy = "classifier" | "small_talk" | "requested_model"

export type RoutingInfo = {
  requestId: string
  model: string               // e.g. "groq-small"
  tier: Tier
  pLow: number | null         // predicted chance the cheap tier answers well (0 to 1)
  pMedium: number | null      // same for the medium tier
  tau: number | null          // the quality bar a tier must reach
  reason: string              // Triage's explanation
  cost: number | null         // what this answer cost, in dollars; null: a stopped answer whose cost isn't known
  costEstimated?: boolean     // the cost is an estimate (a stopped answer: from the text produced so far)
  strongCost: number | null   // what the strongest model would have cost
  saved: number | null        // strongCost - cost
  cacheHit: boolean
  fellBack: boolean           // a backup model answered
  mode: RequestMode
  routedBy?: RoutedBy         // missing on answers saved before it existed
}

// ---------- Messages in the conversation ----------

export type MessageStatus = "pending" | "done" | "error"

export type ChatError = {
  code: string                // e.g. "demo_limit_reached"
  message: string
  retryAfter?: number         // seconds, when Triage says when to try again
  provider?: string           // display name of the provider that rejected a saved key, e.g. "Groq"
}

export type UserMessage = {
  id: string
  role: "user"
  content: string
}

export type AssistantMessage = {
  id: string
  role: "assistant"
  content: string
  status: MessageStatus
  routing?: RoutingInfo
  error?: ChatError
  truncated?: boolean         // the answer stopped at the length limit
  stopped?: boolean           // the visitor pressed Stop: content is the part that arrived
}

export type Message = UserMessage | AssistantMessage

// ---------- Demo status (sidebar card) ----------

export type DemoStatus = {
  enabled: boolean
  paused: boolean
  used: number
  remaining: number
  limit: number
  resetsAt: string            // ISO time, e.g. "2026-11-15T00:00:00Z"
}
