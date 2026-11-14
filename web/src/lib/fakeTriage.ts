import type { Message, RoutingInfo, Tier } from "@/types/chat"
import { TriageError } from "@/lib/errors"

type ChatResult = {
  content: string
  routing: RoutingInfo
}

// Made-up prices per 1,000 tokens, roughly the real ratios between tiers
const MODELS: Record<Tier, { name: string; pricePer1k: number }> = {
  low: { name: "groq-allam", pricePer1k: 0.0002 },
  medium: { name: "groq-small", pricePer1k: 0.0004 },
  high: { name: "groq-large", pricePer1k: 0.0008 },
}

const FAKE_ANSWERS: Record<Tier, string> = {
  low: "That's **12**. 15% of 80 is 0.15 × 80 = 12.",
  medium:
    "A **hash map** stores key-value pairs. It runs each key through a hash function to get an index into an array, " +
    "so lookups, inserts and deletes take about **O(1)** time on average.\n\n- Collisions are handled by chaining or probing\n- It resizes when it gets too full",
  high:
    "**Proof by contradiction.**\n\n1. Assume √2 = a/b in lowest terms.\n2. Then 2b² = a², so a is even: a = 2k.\n" +
    "3. Then 2b² = 4k², so b² = 2k² and b is even too.\n4. Both even contradicts lowest terms, so √2 is irrational. ∎",
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

// quality 0..100 on the dial -> tau 0.50..0.95 (the real mapping comes from the backend check)
function qualityToTau(quality: number): number {
  return 0.5 + (quality / 100) * 0.45
}

export async function sendFakeChat(messages: Message[], quality: number): Promise<ChatResult> {
  await sleep(800 + Math.random() * 700)

  const last = messages[messages.length - 1]
  const question = last?.content ?? ""

  // Type "/error" or "/limit" to test the error UI
  if (question.includes("/limit")) {
    throw new TriageError(
      "demo_limit_reached",
      "You've used today's 10 free demo requests. Add your own provider key to keep going.",
      3600
    )
  }
  if (question.includes("/error")) {
    throw new TriageError("server_error", "Something went wrong. Please try again.")
  }

  // Longer questions look "harder": lower confidence that cheap tiers can handle them
  const difficulty = Math.min(question.length / 120, 1)
  const pLow = Number((0.9 - difficulty * 0.6).toFixed(3))
  const pMedium = Number((0.97 - difficulty * 0.25).toFixed(3))
  const tau = Number(qualityToTau(quality).toFixed(3))

  let tier: Tier
  let reason: string
  if (pLow >= tau) {
    tier = "low"
    reason = `Cheap tier is confident enough (${pLow} ≥ ${tau})`
  } else if (pMedium >= tau) {
    tier = "medium"
    reason = `Cheap ${pLow} < ${tau}; medium ${pMedium} ≥ ${tau}`
  } else {
    tier = "high"
    reason = `Cheap ${pLow} and medium ${pMedium} both < ${tau}, using strong`
  }

  const tokens = 80 + Math.round(question.length / 4) + 120
  const cost = (tokens / 1000) * MODELS[tier].pricePer1k
  const strongCost = (tokens / 1000) * MODELS.high.pricePer1k

  return {
    content: FAKE_ANSWERS[tier],
    routing: {
      requestId: crypto.randomUUID(),
      model: MODELS[tier].name,
      tier,
      pLow,
      pMedium,
      tau,
      reason,
      cost,
      strongCost,
      saved: strongCost - cost,
      cacheHit: false,
      fellBack: false,
      mode: "demo",
    },
  }
}