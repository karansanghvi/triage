import { useMemo } from "react"
import { useAuth } from "@/context/AuthContext"
import * as account from "@/lib/account"
import type { AccountChanges, Provider } from "@/lib/account"
import { TriageError } from "@/lib/errors"
import * as dashboard from "@/lib/dashboard"
import type { Range } from "@/lib/dashboard"

export function useAccountApi() {
  const { token, expireSession, updateUser, refreshUser } = useAuth()

  return useMemo(() => {
    // Every call: attach the session, and log out locally if it has expired
    async function call<T>(fn: (token: string) => Promise<T>): Promise<T> {
      if (!token) throw new TriageError("invalid_session", "Please log in again.")
      try {
        return await fn(token)
      } catch (err) {
        if (err instanceof TriageError && err.code === "invalid_session") expireSession()
        throw err
      }
    }

    return {
      listProviderKeys: () => call((t) => account.listProviderKeys(t)),
      saveProviderKey: (provider: Provider, key: string) => call((t) => account.saveProviderKey(t, provider, key)),
      recheckProviderKey: (provider: Provider) => call((t) => account.recheckProviderKey(t, provider)),
      removeProviderKey: (provider: Provider) => call((t) => account.removeProviderKey(t, provider)),

      listApiKeys: () => call((t) => account.listApiKeys(t)),
      createApiKey: (name: string) => call((t) => account.createApiKey(t, name)),
      revokeApiKey: (id: string) => call((t) => account.revokeApiKey(t, id)),

      getUsage: (range: Range) => call((t) => dashboard.getAccountUsage(t, range)),
      listRequests: (cursor?: string | null, limit?: number) =>
        call((t) => dashboard.listAccountRequests(t, cursor, limit)),

      // After a change, update the user everywhere (sidebar, chat greeting)
      updateAccount: async (changes: AccountChanges) => {
        const updated = await call((t) => account.updateAccount(t, changes))
        if (updated && "id" in updated) updateUser(updated)
        else await refreshUser()
      },
      changePassword: (current: string, next: string) => call((t) => account.changePassword(t, current, next)),
      deleteAccount: (password: string) => call((t) => account.deleteAccount(t, password)),
    }
  }, [token, expireSession, updateUser, refreshUser])
}