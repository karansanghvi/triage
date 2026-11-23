// Run with: npm test. The website's password and wait-time helpers.
import { test } from "node:test"
import assert from "node:assert/strict"
import { PASSWORD_MAX, PASSWORD_MIN, passwordLength, passwordLengthProblem } from "../src/lib/passwordRules.ts"
import { minutesFrom } from "../src/lib/time.ts"

test("the limits are the server's: 10 to 128 characters", () => {
  assert.equal(PASSWORD_MIN, 10)
  assert.equal(PASSWORD_MAX, 128)
})

test("a password is counted by characters, like the server (Python's len), not UTF-16 units", () => {
  assert.equal(passwordLength("🔥🔥🔥🔥🔥🔥"), 6)          // .length would say 12
  assert.equal("🔥🔥🔥🔥🔥🔥".length, 12)
  assert.equal(passwordLength("sunny☀️day🌈"), 11)       // ☀️ is two code points (sun + variation selector)
  assert.equal(passwordLength("ordinary"), 8)
})

test("emoji passwords get the same verdict as the server would give", () => {
  assert.equal(passwordLengthProblem("🔥".repeat(6)), "short")      // passed before: 12 UTF-16 units
  assert.equal(passwordLengthProblem("🔥".repeat(10)), null)
  assert.equal(passwordLengthProblem("🔥".repeat(100)), null)       // was refused before: 200 units
  assert.equal(passwordLengthProblem("🔥".repeat(129)), "long")
  assert.equal(passwordLengthProblem("a".repeat(9)), "short")
  assert.equal(passwordLengthProblem("a".repeat(10)), null)
  assert.equal(passwordLengthProblem("a".repeat(128)), null)
  assert.equal(passwordLengthProblem("a".repeat(129)), "long")
})

test("wait times read naturally", () => {
  assert.equal(minutesFrom(1), "1 minute")
  assert.equal(minutesFrom(59), "1 minute")
  assert.equal(minutesFrom(60), "1 minute")
  assert.equal(minutesFrom(61), "2 minutes")
  assert.equal(minutesFrom(900), "15 minutes")
  assert.equal(minutesFrom(undefined), "a few minutes")
  assert.equal(minutesFrom(0), "a few minutes")
  assert.equal(minutesFrom(Number.NaN), "a few minutes")
})
