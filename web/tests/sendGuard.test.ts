// Run with: npm test. One chat request at a time (src/lib/sendGuard.ts), and with the wake-up flow
// (src/lib/wake.ts): however long the server takes to wake, a message is POSTed exactly once.
import { test } from "node:test"
import assert from "node:assert/strict"
import { createSendGuard } from "../src/lib/sendGuard.ts"
import { createWaker } from "../src/lib/wake.ts"

function deferred<T = void>() {
  let resolve!: (value: T) => void
  let reject!: (err: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

test("a second send in the same moment is refused and runs nothing", async () => {
  const guard = createSendGuard()
  const gate = deferred()
  let runs = 0
  const task = async () => {
    runs++
    await gate.promise
  }
  const first = guard.run(task)
  const second = guard.run(task)                 // Enter and a click, before React re-renders
  assert.ok(first !== null)
  assert.equal(second, null)
  assert.equal(runs, 1)
  assert.equal(guard.busy(), true)
  gate.resolve()
  await first
  assert.equal(guard.busy(), false)
})

test("the guard is free again after the request ends, so the next message can be sent", async () => {
  const guard = createSendGuard()
  let runs = 0
  await guard.run(async () => { runs++ })
  await guard.run(async () => { runs++ })
  assert.equal(runs, 2)
})

test("a failed or stopped request frees the guard too", async () => {
  const guard = createSendGuard()
  await assert.rejects(guard.run(async () => { throw new DOMException("stopped", "AbortError") })!)
  assert.equal(guard.busy(), false)
  assert.throws(() => guard.run(() => { throw new Error("before any await") }))
  assert.equal(guard.busy(), false)
  assert.ok(guard.run(async () => {}) !== null)
})

// The chat's send, as ChatPage does it: wait until the server is awake, then POST the message once
function chatSetup(healthAnswers: boolean[]) {
  let t = 0
  const requests: { method: string; url: string }[] = []
  const waker = createWaker("https://api.test/healthz", {
    now: () => t,
    sleep: async (ms) => { t += ms },
    slowAfterMs: 1_000_000,
    fetch: async (url, init) => {
      requests.push({ method: init.method ?? "GET", url })
      const ok = healthAnswers.shift()
      if (ok === undefined) throw new TypeError("no more answers scripted")
      if (!ok) throw new TypeError("Failed to fetch")             // asleep: the host's page, without CORS
      return new Response(JSON.stringify({ status: "ok" }), { status: 200 })
    },
  })
  const guard = createSendGuard()
  const send = (content: string) =>
    guard.run(async () => {
      if (!(await waker.ensureAwake())) throw new Error("unreachable")
      requests.push({ method: "POST", url: `https://api.test/v1/chat/completions#${content}` })
    })
  return { send, requests, guard, waker }
}

test("while the server wakes up, only /healthz is pinged; the message is POSTed once it's awake", async () => {
  const { send, requests } = chatSetup([false, false, false, true])
  const first = send("hello")
  const again = send("hello")                       // pressed again while "Waking up the server" shows
  assert.equal(again, null)
  await first
  const posts = requests.filter((r) => r.method === "POST")
  assert.deepEqual(posts.map((r) => r.url), ["https://api.test/v1/chat/completions#hello"])
  assert.ok(requests.slice(0, -1).every((r) => r.method === "GET" && r.url.endsWith("/healthz")))
  assert.equal(requests.at(-1)!.method, "POST")
})

test("the background wake-up ping never sends a chat message", async () => {
  const { requests, waker } = chatSetup([false, true])
  await waker.ensureAwake()                        // what main.tsx starts when the site loads
  assert.ok(requests.length === 2 && requests.every((r) => r.method === "GET" && r.url.endsWith("/healthz")))
})

test("a send waiting on the site's own wake-up ping shares it and still POSTs once", async () => {
  const { send, requests, waker } = chatSetup([false, false, true])
  const background = waker.ensureAwake()
  await send("hi")
  await background
  assert.equal(requests.filter((r) => r.method === "POST").length, 1)
  assert.equal(requests.filter((r) => r.url.endsWith("/healthz")).length, 3)      // one shared ping loop
})

test("if the server never wakes, nothing is POSTed and the guard is free for Try again", async () => {
  const { send, requests, guard } = chatSetup(Array(200).fill(false))
  await assert.rejects(send("hi")!)
  assert.equal(requests.filter((r) => r.method === "POST").length, 0)
  assert.equal(guard.busy(), false)
})
