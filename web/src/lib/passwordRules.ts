// The server's password rule (10 to 128 characters), counted the way the server counts: by Unicode code points,
// like Python's len(). JavaScript's .length counts UTF-16 units, so an emoji would count as 2 and a password the
// server refuses could pass here. No imports from "@/", so the Node tests can load it directly.

export const PASSWORD_MIN = 10
export const PASSWORD_MAX = 128

export function passwordLength(password: string): number {
  return [...password].length
}

// "short", "long", or null when the length is fine
export function passwordLengthProblem(password: string): "short" | "long" | null {
  const length = passwordLength(password)
  if (length < PASSWORD_MIN) return "short"
  if (length > PASSWORD_MAX) return "long"
  return null
}
