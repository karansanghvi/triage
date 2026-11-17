import { TriageError } from "@/lib/errors"

const BASE_URL = import.meta.env.VITE_TRIAGE_URL ?? "http://localhost:8000"

export type User = {
  id: string
  email: string
  display_name: string | null
  preferred_provider: string | null
  created_at: string
}

type SessionResponse = {
  user: User
  session_token: string
  expires_at: string
}

type ApiError = { error?: { code?: string; message?: string } }

async function request<T>(path: string, method: string, body?: unknown, token?: string): Promise<T> {
  const headers: Record<string, string> = { "Content-Type": "application/json" }
  if (token) headers.Authorization = `Bearer ${token}`

  let response: Response
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
    })
  } catch {
    throw new TriageError("network_error", "Couldn't reach Triage. Check your connection and try again.")
  }

  if (response.status === 204) return undefined as T

  const data = (await response.json().catch(() => ({}))) as ApiError & T
  if (!response.ok) {
    const retryAfter = response.headers.get("Retry-After")
    throw new TriageError(
      data.error?.code ?? `http_${response.status}`,
      data.error?.message ?? "Something went wrong. Please try again.",
      retryAfter ? Number(retryAfter) : undefined
    )
  }
  return data
}

export function signup(email: string, password: string, displayName?: string) {
  return request<SessionResponse>("/v1/auth/signup", "POST", {
    email,
    password,
    ...(displayName ? { display_name: displayName } : {}),
  })
}

export function login(email: string, password: string) {
  return request<SessionResponse>("/v1/auth/login", "POST", { email, password })
}

export function logout(token: string) {
  return request<void>("/v1/auth/logout", "POST", undefined, token)
}

export function getMe(token: string) {
  return request<User>("/v1/auth/me", "GET", undefined, token)
}