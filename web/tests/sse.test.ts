// Run with: npm test. A mock server sends a streamed answer split at awkward byte positions, as a real network can.
import { test, after } from "node:test"
import assert from "node:assert/strict"
import { createServer, type Server } from "node:http"
import type { AddressInfo } from "node:net"
import { readChatStream, SseParser, StreamFormatError } from "../src/lib/sse.ts"

// ---------- A long answer, as Triage streams it ----------

const TRIAGE_START = { request_id: "chatcmpl-test", model: "groq-large", tier: "high", cost: null }
const TRIAGE_FINAL = { ...TRIAGE_START, cost: 0.00084105 }

// Markdown table rows with multi-byte characters (é, —, emoji, Arabic, CJK) and <br> in cells
function longAnswerPieces(): string[] {
  const pieces: string[] = ["| Term | Meaning — détails |\n", "|---|---|\n"]
  for (let i = 0; i < 400; i++) {
    pieces.push(`| Row ${i} 🤖 | A large language model<br>predicts the next token — naïve café ${i} مرحبا 模型 |\n`)
  }
  pieces.push("\nThat's the summary. ✅")
  return pieces
}

const chunk = (body: object) => `data: ${JSON.stringify(body)}\n\n`
const delta = (content: string) =>
  chunk({ object: "chat.completion.chunk", choices: [{ index: 0, delta: { content }, finish_reason: null }] })

function sseBody(pieces: string[], { final = true, done = true, triage = true, eol = "\n" } = {}): string {
  let text = chunk({ choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }],
    ...(triage ? { triage: TRIAGE_START } : {}) })
  for (const p of pieces) text += delta(p)
  if (final) {
    text += chunk({ choices: [{ index: 0, delta: {}, finish_reason: "stop" }], ...(triage ? { triage: TRIAGE_FINAL } : {}) })
  }
  if (done) text += "data: [DONE]\n\n"
  return eol === "\n" ? text : text.replace(/\n/g, eol)
}

// Cuts the bytes at positions that break things a naive reader gets wrong: inside "data:", inside the JSON,
// between the two newlines of an event's end, and inside multi-byte UTF-8 characters
function awkwardSplits(bytes: Buffer): Buffer[] {
  const cuts = new Set<number>()
  const text = bytes.toString("latin1")                 // one char per byte, so indexes are byte positions
  for (let i = 0; i < bytes.length; i++) {
    const b = bytes[i]
    if (b >= 0x80 && b < 0xc0) cuts.add(i)              // a UTF-8 continuation byte: cut mid-character
  }
  for (const m of text.matchAll(/data:/g)) cuts.add(m.index + 3)      // "dat|a:"
  for (const m of text.matchAll(/\n\n/g)) cuts.add(m.index + 1)       // "\n|\n"
  for (const m of text.matchAll(/\r\n/g)) cuts.add(m.index + 1)       // "\r|\n"
  for (const m of text.matchAll(/"content"/g)) cuts.add(m.index + 5)  // mid-JSON
  for (let i = 7; i < bytes.length; i += 97) cuts.add(i)              // and every so often anywhere
  const sorted = [...cuts].filter((c) => c > 0 && c < bytes.length).sort((a, b) => a - b)
  const parts: Buffer[] = []
  let start = 0
  for (const c of sorted) {
    parts.push(bytes.subarray(start, c))
    start = c
  }
  parts.push(bytes.subarray(start))
  return parts
}

// ---------- The mock server ----------

const routes = new Map<string, () => { parts: Buffer[]; abort?: boolean }>()
const server: Server = createServer(async (req, res) => {
  const route = routes.get(req.url ?? "")
  if (!route) {
    res.writeHead(404).end()
    return
  }
  const { parts, abort } = route()
  res.writeHead(200, { "Content-Type": "text/event-stream" })
  for (let i = 0; i < parts.length; i++) {
    res.write(parts[i])
    if (i % 50 === 0) await new Promise((r) => setTimeout(r, 1))      // let some parts arrive as separate reads
  }
  if (abort) res.destroy()                              // the connection drops mid-answer
  else res.end()
})
await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve))
const base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`
after(() => server.close())

async function stream(path: string) {
  const response = await fetch(base + path)
  let deltas = ""
  let startTriage: unknown = null
  const outcome = await readChatStream(response.body!, {
    onDelta: (t) => (deltas += t),
    onStartTriage: (t) => (startTriage = t),
  })
  return { outcome, deltas, startTriage }
}

// ---------- Tests ----------

test("a long answer split at awkward byte positions arrives complete and intact", async () => {
  const pieces = longAnswerPieces()
  const bytes = Buffer.from(sseBody(pieces), "utf8")
  const parts = awkwardSplits(bytes)
  assert.ok(parts.length > 1000, `split into ${parts.length} parts`)
  routes.set("/long", () => ({ parts }))

  const { outcome, deltas, startTriage } = await stream("/long")
  assert.equal(outcome.kind, "complete")
  if (outcome.kind !== "complete") return
  assert.equal(outcome.content, pieces.join(""))
  assert.equal(deltas, pieces.join(""))
  assert.equal(outcome.finishReason, "stop")
  assert.deepEqual(outcome.finalTriage, TRIAGE_FINAL)
  assert.deepEqual(startTriage, TRIAGE_START)
  assert.ok(!outcome.content.includes("�"), "no broken UTF-8 characters")
})

test("one byte per write works too", async () => {
  const pieces = longAnswerPieces().slice(0, 40)
  const bytes = Buffer.from(sseBody(pieces), "utf8")
  routes.set("/bytes", () => ({ parts: [...bytes].map((b) => Buffer.from([b])) }))
  const { outcome } = await stream("/bytes")
  assert.equal(outcome.kind, "complete")
  if (outcome.kind === "complete") assert.equal(outcome.content, pieces.join(""))
})

test("CRLF line endings, split between \\r and \\n", async () => {
  const pieces = longAnswerPieces().slice(0, 60)
  const bytes = Buffer.from(sseBody(pieces, { eol: "\r\n" }), "utf8")
  routes.set("/crlf", () => ({ parts: awkwardSplits(bytes) }))
  const { outcome } = await stream("/crlf")
  assert.equal(outcome.kind, "complete")
  if (outcome.kind === "complete") {
    assert.equal(outcome.content, pieces.join(""))
    assert.deepEqual(outcome.finalTriage, TRIAGE_FINAL)
  }
})

test("a finished answer without a triage object on its chunks is still complete", async () => {
  const pieces = longAnswerPieces().slice(0, 30)
  routes.set("/no-triage", () => ({ parts: [Buffer.from(sseBody(pieces, { triage: false }), "utf8")] }))
  const { outcome } = await stream("/no-triage")
  assert.equal(outcome.kind, "complete")
  if (outcome.kind === "complete") {
    assert.equal(outcome.content, pieces.join(""))
    assert.equal(outcome.finalTriage, null)
  }
})

test("[DONE] alone, or the finish chunk alone, counts as finished", async () => {
  const pieces = ["Hello", " world"]
  routes.set("/done-only", () => ({ parts: [Buffer.from(sseBody(pieces, { final: false }), "utf8")] }))
  routes.set("/finish-only", () => ({ parts: [Buffer.from(sseBody(pieces, { done: false }), "utf8")] }))
  assert.equal((await stream("/done-only")).outcome.kind, "complete")
  assert.equal((await stream("/finish-only")).outcome.kind, "complete")
})

test("a final event without its closing blank line still counts", async () => {
  const text = sseBody(["Hi"]).replace(/\n\n$/, "")           // ends "data: [DONE]" with no newline
  routes.set("/no-trailing", () => ({ parts: [Buffer.from(text, "utf8")] }))
  assert.equal((await stream("/no-trailing")).outcome.kind, "complete")
})

test("a connection that closes before the end is interrupted, not complete", async () => {
  const pieces = longAnswerPieces().slice(0, 50)
  const text = sseBody(pieces, { final: false, done: false })
  routes.set("/cut", () => ({ parts: [Buffer.from(text, "utf8")] }))
  const { outcome } = await stream("/cut")
  assert.equal(outcome.kind, "interrupted")
  assert.equal(outcome.content, pieces.join(""))
})

test("a connection that drops mid-event rejects or ends interrupted, never complete", async () => {
  const bytes = Buffer.from(sseBody(longAnswerPieces().slice(0, 50), { final: false, done: false }), "utf8")
  routes.set("/drop", () => ({ parts: [bytes.subarray(0, bytes.length - 40)], abort: true }))
  try {
    const { outcome } = await stream("/drop")
    assert.equal(outcome.kind, "interrupted")
  } catch (err) {
    assert.ok(!(err instanceof StreamFormatError), "a dropped connection is not a format error")
  }
})

test("an error event part-way is reported with its code and message", async () => {
  const text = chunk({ choices: [{ index: 0, delta: { content: "" }, finish_reason: null }], triage: TRIAGE_START }) +
    delta("Partial") + chunk({ error: { code: "stream_interrupted", message: "The model failed part-way." } })
  routes.set("/error", () => ({ parts: [Buffer.from(text, "utf8")] }))
  const { outcome } = await stream("/error")
  assert.deepEqual(outcome, {
    kind: "error", content: "Partial", code: "stream_interrupted", message: "The model failed part-way.",
    startTriage: TRIAGE_START,
  })
})

test("an event that isn't JSON is a format error", async () => {
  routes.set("/bad-json", () => ({ parts: [Buffer.from("data: {not json\n\n", "utf8")] }))
  await assert.rejects(stream("/bad-json"), StreamFormatError)
})

test("parser: comments, 'data:' without a space, and multi-line data", () => {
  const parser = new SseParser()
  assert.deepEqual(parser.push(": keep-alive\n\ndata:{\"a\":1}\n\ndata: line1\ndata: line2\n\n"), [
    '{"a":1}',
    "line1\nline2",
  ])
  assert.deepEqual(parser.push("data: half"), [])
  assert.deepEqual(parser.push("way\n"), [])
  assert.deepEqual(parser.push("\n"), ["halfway"])
})
