// Run with: npm test (node --test; Node strips the TypeScript types itself)
import { test } from "node:test"
import assert from "node:assert/strict"
import { LANGUAGE_INSTRUCTION, withLanguageInstruction, type ApiMessage } from "../src/lib/chatMessages.ts"

test("adds the language instruction as the first message", () => {
  const messages: ApiMessage[] = [{ role: "user", content: "hello" }]
  assert.deepEqual(withLanguageInstruction(messages), [
    { role: "system", content: LANGUAGE_INSTRUCTION },
    { role: "user", content: "hello" },
  ])
})

test("the instruction is the same sentence as the backend's groq-allam prefix", () => {
  assert.equal(LANGUAGE_INSTRUCTION, "Always reply in the same language as the user's latest message.")
})

test("adds it on every request: a longer conversation too", () => {
  const messages: ApiMessage[] = [
    { role: "user", content: "hi" },
    { role: "assistant", content: "Hello!" },
    { role: "user", content: "Explain recursion" },
  ]
  const sent = withLanguageInstruction(messages)
  assert.equal(sent.length, 4)
  assert.deepEqual(sent[0], { role: "system", content: LANGUAGE_INSTRUCTION })
  assert.deepEqual(sent.slice(1), messages)
})

test("leaves a conversation that already has a system message alone", () => {
  const messages: ApiMessage[] = [
    { role: "system", content: "You are terse." },
    { role: "user", content: "hello" },
  ]
  assert.equal(withLanguageInstruction(messages), messages)
})

test("doesn't change its input", () => {
  const messages: ApiMessage[] = [{ role: "user", content: "hello" }]
  withLanguageInstruction(messages)
  assert.deepEqual(messages, [{ role: "user", content: "hello" }])
})
