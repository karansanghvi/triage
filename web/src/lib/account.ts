import { request, type User } from "@/lib/auth"

// ---------- Shapes the backend returns ----------

export type Provider = "openai" | "anthropic" | "groq" | "gemini"

export const PROVIDERS: { id: Provider; name: string; letter: string; placeholder: string }[] = [
  { id: "openai", name: "OpenAI", letter: "O", placeholder: "sk-..." },
  { id: "anthropic", name: "Anthropic", letter: "A", placeholder: "sk-ant-..." },
  { id: "groq", name: "Groq", letter: "G", placeholder: "gsk_..." },
  { id: "gemini", name: "Gemini", letter: "G", placeholder: "AIza..." },
]

export type ProviderKey = {
  provider: Provider
  last4: string
  status: "valid" | "invalid" | "unverified"
  status_detail: string | null
  verified_at: string | null
  created_at: string
  updated_at: string
}

export type ApiKey = {
  id: string
  name: string
  public_id: string
  created_at: string
  last_used_at: string | null
  revoked_at: string | null
}

// Only returned once, right after creating a key
export type CreatedApiKey = ApiKey & {
  key: string
  note?: string
}

// ---------- Provider keys ----------

export function listProviderKeys(token: string) {
  return request<ProviderKey[]>("/v1/account/provider-keys", "GET", undefined, token)
}

export function saveProviderKey(token: string, provider: Provider, key: string) {
  return request<ProviderKey>(`/v1/account/provider-keys/${provider}`, "PUT", { key }, token)
}

export function recheckProviderKey(token: string, provider: Provider) {
  return request<ProviderKey>(`/v1/account/provider-keys/${provider}/verify`, "POST", undefined, token)
}

export function removeProviderKey(token: string, provider: Provider) {
  return request<void>(`/v1/account/provider-keys/${provider}`, "DELETE", undefined, token)
}

// ---------- Triage API keys ----------

export function listApiKeys(token: string) {
  return request<ApiKey[]>("/v1/account/api-keys", "GET", undefined, token)
}

export function createApiKey(token: string, name: string) {
  return request<CreatedApiKey>("/v1/account/api-keys", "POST", { name }, token)
}

export function revokeApiKey(token: string, id: string) {
  return request<void>(`/v1/account/api-keys/${id}`, "DELETE", undefined, token)
}

// ---------- Account ----------

export type AccountChanges = {
  display_name?: string | null
  preferred_provider?: Provider | null
}

export function updateAccount(token: string, changes: AccountChanges) {
  return request<User | undefined>("/v1/account", "PATCH", changes, token)
}

export function changePassword(token: string, currentPassword: string, newPassword: string) {
  return request<void>(
    "/v1/account/password",
    "POST",
    { current_password: currentPassword, new_password: newPassword },
    token
  )
}

export function deleteAccount(token: string, password: string) {
  return request<void>("/v1/account", "DELETE", { password }, token)
}