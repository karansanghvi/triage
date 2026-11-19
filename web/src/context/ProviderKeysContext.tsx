import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react"
import { useAuth } from "@/context/AuthContext"
import { useAccountApi } from "@/hooks/useAccountApi"
import { PROVIDERS, type Provider, type ProviderKey } from "@/lib/account"

// The logged-in user's saved provider keys, shared by the sidebar and the Provider keys settings page,
// so a key saved, re-checked or removed in Settings updates the sidebar at once.

type ProviderKeysContextValue = {
  keys: ProviderKey[] | null       // null: not loaded (logged out, loading, or the load failed)
  loading: boolean                 // logged in and the first load hasn't finished
  error: boolean                   // the last load failed
  hasUsableKey: boolean            // at least one key that is "valid" or "unverified"
  allRejected: boolean             // at least one key saved, none usable
  usableProviders: string[]        // display names, e.g. ["Anthropic", "Groq"]
  refresh: () => Promise<void>
  upsertKey: (key: ProviderKey) => void
  removeKey: (provider: Provider) => void
}

const ProviderKeysContext = createContext<ProviderKeysContextValue | null>(null)

function usable(key: ProviderKey): boolean {
  return key.status === "valid" || key.status === "unverified"
}

export function ProviderKeysProvider({ children }: { children: ReactNode }) {
  const { user } = useAuth()
  const api = useAccountApi()
  const userId = user?.id ?? null

  // Tagged with the user they belong to: after logout (or another login) they no longer count, with no reset
  const [loaded, setLoaded] = useState<{ userId: string; keys: ProviderKey[] | null; error: boolean } | null>(null)
  const current = loaded && loaded.userId === userId ? loaded : null
  const keys = current?.keys ?? null
  const error = current?.error ?? false

  // The latest api and user, for refresh() without re-running the load effect when they change identity
  const apiRef = useRef(api)
  const userRef = useRef(userId)
  useEffect(() => {
    apiRef.current = api
    userRef.current = userId
  })

  // Load once per logged-in user
  useEffect(() => {
    if (!userId) return
    let cancelled = false
    apiRef.current
      .listProviderKeys()
      .then((list) => {
        if (!cancelled) setLoaded({ userId, keys: list, error: false })
      })
      .catch(() => {
        if (!cancelled) setLoaded({ userId, keys: null, error: true })
      })
    return () => {
      cancelled = true
      setLoaded(null)          // logged out (or another user): forget the keys
    }
  }, [userId])

  // Reload in the background: the current keys stay shown until the new list arrives
  const refresh = useCallback(async () => {
    const forUser = userRef.current
    if (!forUser) return
    try {
      const list = await apiRef.current.listProviderKeys()
      if (userRef.current === forUser) setLoaded({ userId: forUser, keys: list, error: false })
    } catch {
      if (userRef.current === forUser) {
        setLoaded((prev) => ({ userId: forUser, keys: prev?.userId === forUser ? prev.keys : null, error: true }))
      }
    }
  }, [])

  // Local updates after a save, re-check or remove in Settings
  const update = useCallback((change: (keys: ProviderKey[]) => ProviderKey[]) => {
    const forUser = userRef.current
    if (!forUser) return
    setLoaded((prev) => {
      const keys = prev?.userId === forUser ? (prev.keys ?? []) : []
      return { userId: forUser, keys: change(keys), error: false }
    })
  }, [])

  const upsertKey = useCallback(
    (key: ProviderKey) => update((keys) => [...keys.filter((k) => k.provider !== key.provider), key]),
    [update]
  )

  const removeKey = useCallback(
    (provider: Provider) => update((keys) => keys.filter((k) => k.provider !== provider)),
    [update]
  )

  const value = useMemo<ProviderKeysContextValue>(() => {
    const usableIds = new Set((keys ?? []).filter(usable).map((k) => k.provider))
    return {
      keys,
      loading: !!userId && keys === null && !error,
      error,
      hasUsableKey: usableIds.size > 0,
      allRejected: (keys ?? []).length > 0 && usableIds.size === 0,
      usableProviders: PROVIDERS.filter((p) => usableIds.has(p.id)).map((p) => p.name),
      refresh,
      upsertKey,
      removeKey,
    }
  }, [keys, userId, error, refresh, upsertKey, removeKey])

  return <ProviderKeysContext.Provider value={value}>{children}</ProviderKeysContext.Provider>
}

export function useProviderKeys(): ProviderKeysContextValue {
  const value = useContext(ProviderKeysContext)
  if (!value) throw new Error("useProviderKeys must be used inside ProviderKeysProvider")
  return value
}
