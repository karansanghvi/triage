// The API's host (Render's free plan) puts it to sleep after 15 idle minutes, and waking it takes about two minutes.
// This pings GET /healthz in the background so the server is usually awake by the time someone sends a message,
// and tells the chat when it's still waking, so it can say so instead of hanging.
//
// Awake means /healthz answered with Triage's own JSON {"status": "ok"}. Anything else counts as still waking:
// no answer, a 503 while the model loads, or the host's "waking up" page (HTML, and without CORS headers, so the
// browser refuses it). Pure logic, no React: tests run it with a fake fetch and clock.

export type WakeState = "unknown" | "awake" | "waking" | "unreachable"

export const WAKE_ESTIMATE_SECONDS = 120     // about two minutes: the host's spin-up plus the API's own startup
export const LONG_PROMPT_CHARS = 3000        // at or above: reading the message can take a while on the free server

export type WakeOptions = {
  fetch?: (url: string, init: RequestInit) => Promise<Response>
  now?: () => number
  sleep?: (ms: number) => Promise<void>
  attemptTimeoutMs?: number      // one ping
  retryMs?: number               // between pings
  slowAfterMs?: number           // no answer by then: say it's waking
  giveUpMs?: number              // stop trying
  trustAwakeMs?: number          // an "awake" this recent needs no new ping (the host sleeps after 15 minutes)
}

export type Waker = {
  state: () => WakeState
  wakingSince: () => number | null           // when waking was first noticed (now() units), while waking
  subscribe: (listener: () => void) => () => void
  ensureAwake: (signal?: AbortSignal) => Promise<boolean>   // true once awake; false if it gave up or was aborted
}

export function createWaker(healthUrl: string, options: WakeOptions = {}): Waker {
  const {
    fetch: fetchFn = (url, init) => globalThis.fetch(url, init),
    now = Date.now,
    sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
    attemptTimeoutMs = 5000,
    retryMs = 3000,
    slowAfterMs = 1500,
    giveUpMs = 4 * 60_000,
    trustAwakeMs = 10 * 60_000,
  } = options

  let state: WakeState = "unknown"
  let since: number | null = null
  let lastAwake = -Infinity
  let running: Promise<boolean> | null = null
  const listeners = new Set<() => void>()

  const set = (next: WakeState) => {
    if (next === "waking" && state !== "waking") since = now()
    if (next !== "waking") since = null
    if (next !== state) {
      state = next
      listeners.forEach((l) => l())
    }
  }

  async function ping(): Promise<boolean> {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), attemptTimeoutMs)
    try {
      const response = await fetchFn(healthUrl, { signal: controller.signal, cache: "no-store" })
      if (!response.ok) return false
      const body = (await response.json().catch(() => null)) as { status?: string } | null
      return body?.status === "ok"
    } catch {
      return false
    } finally {
      clearTimeout(timer)
    }
  }

  async function run(): Promise<boolean> {
    const started = now()
    const slow = setTimeout(() => {
      if (state !== "awake") set("waking")
    }, slowAfterMs)
    try {
      for (;;) {
        if (await ping()) {
          lastAwake = now()
          set("awake")
          return true
        }
        if (now() - started >= giveUpMs) {
          set("unreachable")
          return false
        }
        set("waking")
        await sleep(retryMs)
      }
    } finally {
      clearTimeout(slow)
    }
  }

  function ensureAwake(signal?: AbortSignal): Promise<boolean> {
    if (state === "awake" && now() - lastAwake < trustAwakeMs) return Promise.resolve(true)
    if (!running) running = run().finally(() => { running = null })
    if (!signal) return running
    if (signal.aborted) return Promise.resolve(false)
    const current = running
    return new Promise((resolve) => {
      const onAbort = () => resolve(false)
      signal.addEventListener("abort", onAbort, { once: true })
      current.then((ok) => {
        signal.removeEventListener("abort", onAbort)
        resolve(ok)
      })
    })
  }

  return {
    state: () => state,
    wakingSince: () => since,
    subscribe: (listener) => {
      listeners.add(listener)
      return () => listeners.delete(listener)
    },
    ensureAwake,
  }
}

// The user text a request sends (what the router reads); long enough, and the chat says it may take a while
export function isLongPrompt(messages: { role: string; content: string }[]): boolean {
  return messages.filter((m) => m.role === "user").reduce((n, m) => n + m.content.length, 0) >= LONG_PROMPT_CHARS
}
