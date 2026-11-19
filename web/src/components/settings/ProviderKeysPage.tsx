import { useState } from "react"
import { Check, ChevronDown, Loader2 } from "lucide-react"
import SettingsCard from "@/components/settings/SettingsCard"
import ProviderKeyRow from "@/components/settings/ProviderKeyRow"
import { useAccountApi } from "@/hooks/useAccountApi"
import { useAuth } from "@/context/AuthContext"
import { useProviderKeys } from "@/context/ProviderKeysContext"
import { PROVIDERS, type Provider } from "@/lib/account"
import { TriageError } from "@/lib/errors"

export default function ProviderKeysPage() {
  const api = useAccountApi()
  const { user } = useAuth()
  // Shared with the sidebar: changes here show there at once
  const { keys, error: loadError, upsertKey: upsert, removeKey: remove } = useProviderKeys()

  const [prefSaving, setPrefSaving] = useState(false)
  const [prefSaved, setPrefSaved] = useState(false)
  const [prefError, setPrefError] = useState<string | null>(null)

  // Only providers with a usable key can be preferred
  const usable = PROVIDERS.filter((p) => keys?.some((k) => k.provider === p.id && k.status !== "invalid"))
  const current = user?.preferred_provider ?? ""
  const currentHasNoKey = current !== "" && !usable.some((p) => p.id === current)

  async function handlePreference(value: string) {
    setPrefError(null)
    setPrefSaved(false)
    setPrefSaving(true)
    try {
      await api.updateAccount({ preferred_provider: value === "" ? null : (value as Provider) })
      setPrefSaved(true)
      setTimeout(() => setPrefSaved(false), 2000)
    } catch (err) {
      setPrefError(err instanceof TriageError ? err.message : "Couldn't save your preference. Please try again.")
    } finally {
      setPrefSaving(false)
    }
  }

  return (
    <SettingsCard
      title="Provider keys"
      description="Triage uses these to answer your requests. Stored encrypted, never logged, and only the last 4 characters are ever shown."
    >
      {loadError ? (
        <p className="px-6 py-6 text-sm text-[#FCA5A5]">Couldn't load your keys. Refresh the page to try again.</p>
      ) : !keys ? (
        <p className="flex items-center gap-2 px-6 py-6 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" /> Loading your keys…
        </p>
      ) : (
        <>
          {PROVIDERS.map((provider) => (
            <ProviderKeyRow
              key={provider.id}
              provider={provider}
              saved={keys.find((k) => k.provider === provider.id)}
              onSaved={upsert}
              onRemoved={remove}
            />
          ))}

          {/* Preferred provider */}
          <div className="flex flex-wrap items-center justify-between gap-4 border-t border-[#1F2130] bg-[#0F1016] px-6 py-5">
            <div className="min-w-0 flex-[1_1_360px]">
              <label htmlFor="prefer" className="text-sm font-medium">
                Preferred provider
              </label>
              <p className="mt-1 text-[13px] leading-relaxed text-muted-foreground">
                Triage still picks how powerful a model each request needs. This provider answers first; your other
                keys are backups.
              </p>
              {prefError && <p role="alert" className="mt-2 text-xs text-[#FCA5A5]">{prefError}</p>}
            </div>
            <div className="flex items-center gap-2">
              {prefSaving && <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" aria-label="Saving" />}
              {prefSaved && <Check className="h-4 w-4 text-[#4ADE80]" aria-label="Saved" />}
              <div className="relative w-[220px]">
                <select
                  id="prefer"
                  value={current}
                  disabled={prefSaving || usable.length === 0}
                  onChange={(e) => handlePreference(e.target.value)}
                  className="h-11 w-full cursor-pointer appearance-none rounded-xl border border-[#1F2130] bg-[#0A0A0F] pl-3.5 pr-10 text-sm outline-none focus:border-[#3B82F6] disabled:cursor-not-allowed disabled:opacity-60"
                >
                  <option value="">No preference</option>
                  {usable.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.name}
                    </option>
                  ))}
                  {currentHasNoKey && (
                    <option value={current}>
                      {PROVIDERS.find((p) => p.id === current)?.name ?? current} (no key)
                    </option>
                  )}
                </select>
                <ChevronDown className="pointer-events-none absolute right-3.5 top-3.5 h-4 w-4 text-muted-foreground" />
              </div>
            </div>
          </div>
        </>
      )}
    </SettingsCard>
  )
}