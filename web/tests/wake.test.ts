// Run with: npm test. The background /healthz ping that tells the chat when the free server is still waking up.
import { test } from "node:test"
import assert from "node:assert/strict"
import { createWaker, isLongPrompt, LONG_PROMPT_CHARS } from "../src/lib/wake.ts"

const OK = () => new Response(JSON.stringify({ status: "ok" }), { status: 200, headers: { "Content-Type": "application/json" } })
const LOADING_PAGE = () => new Response("<html>Service waking up...</html>", { status: 200, headers: { "Content-Type": "text/html" } })
const STARTING = () => new Response(JSON.stringify({ status: "unavailable" }), { status: 503 })

// A clock that only moves when the waker sleeps, and a fetch that answers from a script
function setup(script: Array<() => Response | Promise<Response>>, opts = {}) {
  let t = 0
  const calls: string[] = []
  const waker = createWaker("https://api.test/healthz", {
    now: () => t,
    sleep: async (ms) => { t += ms },
    fetch: async (url) => {
      calls.push(url)
      const next = script.shift()
      if (!next) throw new TypeError("no more answers scripted")
      return next()
    },
    slowAfterMs: 1_000_000,          // the "waking" timer is wall-clock; these tests drive state by failures
    ...opts,
  })
  return { waker, calls, advance: (ms: number) => { t += ms } }
}

test("an awake server: one ping, then trusted for a while", async () => {
  const { waker, calls, advance } = setup([OK, OK])
  assert.equal(await waker.ensureAwake(), true)
  assert.equal(waker.state(), "awake")
  assert.equal(await waker.ensureAwake(), true)
  assert.equal(calls.length, 1)                       // no second ping inside the trust window
  advance(11 * 60_000)
  assert.equal(await waker.ensureAwake(), true)       // the host may have slept since: ping again
  assert.equal(calls.length, 2)
})

test("a sleeping server: waking until /healthz says ok", async () => {
  const states: string[] = []
  const failing = () => Promise.reject(new TypeError("Failed to fetch"))     // e.g. the loading page, blocked by CORS
  const { waker } = setup([failing, LOADING_PAGE, STARTING, OK])
  waker.subscribe(() => states.push(waker.state()))
  assert.equal(await waker.ensureAwake(), true)
  assert.deepEqual(states, ["waking", "awake"])
  assert.equal(waker.wakingSince(), null)
})

test("the host's loading page and a starting API don't count as awake", async () => {
  const { waker } = setup([LOADING_PAGE, STARTING, OK])
  await waker.ensureAwake()
  assert.equal(waker.state(), "awake")
})

test("it gives up after the limit and says so", async () => {
  const failing = () => Promise.reject(new TypeError("Failed to fetch"))
  const { waker } = setup(Array(200).fill(failing), { giveUpMs: 30_000, retryMs: 3000 })
  assert.equal(await waker.ensureAwake(), false)
  assert.equal(waker.state(), "unreachable")
})

test("callers share one ping loop, and an aborted wait returns at once", async () => {
  let release: (r: Response) => void = () => {}
  const { waker, calls } = setup([() => new Promise<Response>((r) => { release = r })])
  const controller = new AbortController()
  const a = waker.ensureAwake()
  const b = waker.ensureAwake(controller.signal)
  controller.abort()
  assert.equal(await b, false)                        // Stop while waking
  release(OK())
  assert.equal(await a, true)
  assert.equal(calls.length, 1)
})

test("a long prompt is the user text, at the threshold or above", () => {
  const user = (n: number) => ({ role: "user", content: "x".repeat(n) })
  assert.equal(isLongPrompt([user(LONG_PROMPT_CHARS - 1)]), false)
  assert.equal(isLongPrompt([user(2000), { role: "assistant", content: "y".repeat(5000) }, user(1000)]), true)
})
