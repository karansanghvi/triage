import { useAccountApi } from "@/hooks/useAccountApi";
import type { Provider, ProviderKey } from "@/lib/account";
import { TriageError } from "@/lib/errors";
import { minutesFrom, relativeTime } from "@/lib/time";
import { useState, type FormEvent } from "react";
import StatusBadge from "./StatusBadge";
import { Loader2, RefreshCw, Trash2 } from "lucide-react";
import ConfirmDialog from "../common/ConfirmDialog";

type ProviderKeyRowProps = {
  provider: { id: Provider; name: string; letter: string; placeholder: string }
  saved: ProviderKey | undefined
  onSaved: (key: ProviderKey) => void
  onRemoved: (provider: Provider) => void
}

function describe(key: ProviderKey): { text: string; tone: "muted" | "warn" | "bad" } {
  if (key.status === "valid") {
    return { text: `Key accepted · checked ${key.verified_at ? relativeTime(key.verified_at) : "recently"}`, tone: "muted" }
  }
  if (key.status === "unverified") {
    return { text: key.status_detail ?? "Couldn't check it yet. Try Re-check later.", tone: "warn" }
  }
  return { text: key.status_detail ?? "The provider rejected this key. Replace it with a working one.", tone: "bad" }
}

function errorText(err: unknown, providerName: string): string {
  if (err instanceof TriageError && err.code === "invalid_provider_key") return err.message
  if (err instanceof TriageError && err.retryAfter) {
    return `Too many key changes recently. Try again in ${minutesFrom(err.retryAfter)}.`
  }
  if (err instanceof TriageError) return err.message
  return `Couldn't reach ${providerName}. Please try again.`
}

export default function ProviderKeyRow({ provider, saved, onSaved, onRemoved }: ProviderKeyRowProps) {

    const api = useAccountApi()
    const [editing, setEditing] = useState(false)
    const [value, setValue] = useState("")
    const [busy, setBusy] = useState<"save" | "recheck" | null>(null)
    const [error, setError] = useState<string | null>(null)

    const [confirmingRemove, setConfirmingRemove] = useState(false)
    const [removing, setRemoving] = useState(false)
    const [removeError, setRemoveError] = useState<string | null>(null)

    const showInput = !saved || editing

    async function handleSave(e: FormEvent) {
        e.preventDefault()
        const key = value.trim()
        if (!key) {
            setError(`Paste your ${provider.name} key first.`)
            return
        }
        setError(null)
        setBusy("save")
        try {
            const result = await api.saveProviderKey(provider.id, key)
            onSaved(result)
            setValue("")
            setEditing(false)
        } catch (err) {
            setError(errorText(err, provider.name))
        } finally {
            setBusy(null)
        }
    }

    async function handleRecheck() {
        setError(null)
            setBusy("recheck")
        try {
            onSaved(await api.recheckProviderKey(provider.id))
        } catch (err) {
            setError(errorText(err, provider.name))
        } finally {
            setBusy(null)
        }
    }

    async function handleRemove() {
        setRemoving(true)
        setRemoveError(null)
        try {
            await api.removeProviderKey(provider.id)
            onRemoved(provider.id)
            setConfirmingRemove(false)
        } catch (err) {
            setRemoveError(errorText(err, provider.name))
        } finally {
            setRemoving(false)
        }
    }

    const info = saved ? describe(saved) : null
    const inputId = `pk-${provider.id}`

  return (
    <>
      <div className="flex flex-wrap items-start gap-4 border-b border-[#1F2130] px-6 py-[18px] last:border-b-0">
        {/* Provider name and status */}
        <div className="min-w-0 flex-[0_0_180px]">
            <p className="text-sm font-medium">{provider.name}</p>
            <div className="mt-1">
                <StatusBadge kind={saved ? saved.status : "missing"} />
            </div>
        </div>

        <div className="min-w-0 flex-[1_1_320px]">
            {showInput ? (
                // Paste a new key
                <form onSubmit={handleSave} noValidate className="flex flex-wrap gap-2">
                    <label htmlFor={inputId} className="sr-only">
                        {provider.name} API key
                    </label>
                    <input
                        id={inputId}
                        type="password"
                        autoComplete="off"
                        spellCheck={false}
                        value={value}
                        onChange={(e) => setValue(e.target.value)}
                        placeholder={`Paste your ${provider.name} key (${provider.placeholder})`}
                        disabled={busy === "save"}
                        className="h-11 min-w-0 flex-[1_1_220px] rounded-xl border border-[#1F2130] bg-[#0A0A0F] px-3.5 font-mono text-sm outline-none transition-colors placeholder:font-sans placeholder:text-[#7C8192] focus:border-[#3B82F6] focus:ring-4 focus:ring-[#3B82F6]/15 disabled:opacity-60"
                    />
                    <button
                        type="submit"
                        disabled={busy === "save"}
                        className="inline-flex h-11 items-center gap-2 rounded-[10px] bg-[#2563EB] px-4 text-[13px] font-semibold text-white transition-colors hover:bg-[#1D4ED8] disabled:opacity-60"
                    >
                        {busy === "save" && <Loader2 className="h-4 w-4 animate-spin" />}
                        {busy === "save" ? `Checking with ${provider.name}…` : "Save"}
                    </button>
                    {editing && (
                        <button
                            type="button"
                            onClick={() => {
                                setEditing(false)
                                setValue("")
                                setError(null)
                            }}
                            className="h-11 rounded-[10px] px-3 text-[13px] font-medium text-muted-foreground hover:bg-accent hover:text-foreground"
                        >
                            Cancel
                        </button>
                    )}
                </form>
            ) : (
                // A saved key 
                saved &&
                info && (
                    <div className="flex flex-wrap items-center justify-between gap-3">
                        <div className="min-w-0">
                            <p className="font-mono text-sm tracking-[0.04em] text-[#C9CDD8]">••••••••{saved.last4}</p>
                            <p
                                className={
                                    info.tone === "warn"
                                    ? "mt-1 text-xs text-[#FBBF24]"
                                    : info.tone === "bad"
                                        ? "mt-1 text-xs text-[#FCA5A5]"
                                        : "mt-1 text-xs text-muted-foreground"
                                }
                            >
                                {info.text}
                            </p>
                        </div>
                        <div className="flex gap-2">
                            <button
                                type="button"
                                onClick={handleRecheck}
                                disabled={busy === "recheck"}
                                className="inline-flex h-10 items-center gap-1.5 rounded-[10px] border border-[#2A2D3D] bg-[#1A1C26] px-3.5 text-[13px] font-medium transition-colors hover:border-[#3B4058] disabled:opacity-60"
                            >
                                <RefreshCw className={busy === "recheck" ? "h-[15px] w-[15px] animate-spin" : "h-[15px] w-[15px]"} />
                                {busy === "recheck" ? "Checking..." : "Re-check"}
                            </button>
                            <button
                                type="button"
                                onClick={() => {
                                    setEditing(true)
                                    setError(null)
                                }}
                                className="h-10 rounded-[10px] border border-[#2A2D3D] bg-[#1A1C26] px-3.5 text-[13px] font-medium transition-colors hover:border-[#3B4058]"
                            >
                                Replace
                            </button>
                            <button
                                type="button"
                                onClick={() => {
                                    setRemoveError(null)
                                    setConfirmingRemove(true)
                                }}
                                aria-label={`Remove ${provider.name} key`}
                                className="flex h-10 w-10 items-center justify-center rounded-[10px] border border-[#1F2130] text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
                            >
                                <Trash2 className="h-[15px] w-[15px]" />
                            </button>
                        </div>
                    </div>
                )
            )}

            {error && (
                <p role="alert" className="mt-2 text-xs text-[#FCA5A5]">
                    {error}
                </p>
            )}
        </div>
        
        <ConfirmDialog
            open={confirmingRemove}
            tone="danger"
            title={`Remove your ${provider.name} key?`}
            description={`Triage stops using ${provider.name} for your requests. You can only add a key again any time.`}
            confirmLabel="Remove key"
            busy={removing}
            error={removeError}
            onConfirm={handleRemove}
            onCancel={() => setConfirmingRemove(false)}
        />
      </div>
    </>
  )
}
