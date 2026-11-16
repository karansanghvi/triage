/* eslint-disable react-hooks/set-state-in-effect */
import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react"
import * as authApi from "@/lib/auth"
import type { User } from "@/lib/auth"
import { TriageError } from "@/lib/errors"

const TOKEN_KEY = "triage.session"

type AuthContextValue = {
  user: User | null
  token: string | null
  loading: boolean // true while checking a saved session on page load
  signup: (email: string, password: string, displayName?: string) => Promise<void>
  login: (email: string, password: string) => Promise<void>
  logout: () => Promise<void>
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(() => localStorage.getItem(TOKEN_KEY))
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState<boolean>(() => !!localStorage.getItem(TOKEN_KEY))

  // On page load (or a new token), ask the backend who we are
  useEffect(() => {
    if (!token) {
      setUser(null)
      setLoading(false)
      return
    }
    let cancelled = false
    authApi
      .getMe(token)
      .then((me) => {
        if (!cancelled) setUser(me)
      })
      .catch((err) => {
        // An expired or revoked session: forget it. A network error: keep it and try again later.
        if (!cancelled && err instanceof TriageError && err.code !== "network_error") {
          localStorage.removeItem(TOKEN_KEY)
          setToken(null)
          setUser(null)
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [token])

  const saveSession = useCallback((newToken: string, newUser: User) => {
    localStorage.setItem(TOKEN_KEY, newToken)
    setUser(newUser)
    setToken(newToken)
  }, [])

  const signup = useCallback(
    async (email: string, password: string, displayName?: string) => {
      const res = await authApi.signup(email, password, displayName)
      saveSession(res.session_token, res.user)
    },
    [saveSession]
  )

  const login = useCallback(
    async (email: string, password: string) => {
      const res = await authApi.login(email, password)
      saveSession(res.session_token, res.user)
    },
    [saveSession]
  )

  const logout = useCallback(async () => {
    const current = token
    localStorage.removeItem(TOKEN_KEY)
    setToken(null)
    setUser(null)
    if (current) {
      try {
        await authApi.logout(current) // revoke it on the server too
      } catch {
        // Already gone or offline: it's removed from this browser either way
      }
    }
  }, [token])

  return (
    <AuthContext.Provider value={{ user, token, loading, signup, login, logout }}>{children}</AuthContext.Provider>
  )
}

export function useAuth(): AuthContextValue {
  const value = useContext(AuthContext)
  if (!value) throw new Error("useAuth must be used inside AuthProvider")
  return value
}