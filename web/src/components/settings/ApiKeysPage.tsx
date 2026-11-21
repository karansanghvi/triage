import { useAccountApi } from "@/hooks/useAccountApi";
import SettingsCard from "./SettingsCard";
import { useEffect, useState } from "react";
import type { ApiKey } from "@/lib/account";
import { TriageError } from "@/lib/errors";
import { KeyRound, Loader2, Plus } from "lucide-react";
import { cn } from "@/lib/utils";
import { relativeTime, shortDate } from "@/lib/time";
import StatusBadge from "./StatusBadge";
import CreateKeyDialog from "./CreateKeyDialog";
import ConfirmDialog from "../common/ConfirmDialog";

// Active keys first (newest first), revoked keys after
function sortKeys(keys: ApiKey[]): ApiKey[] {
  return [...keys].sort((a, b) => {
    if (!!a.revoked_at !== !!b.revoked_at) return a.revoked_at ? 1 : -1
    return b.created_at.localeCompare(a.created_at)
  })
}

export default function ApiKeysPage() {

    const api = useAccountApi()
    const [keys, setKeys] = useState<ApiKey[] | null>(null)
    const [loadError, setLoadError] = useState(false)
    const [creating, setCreating] = useState(false)

    const [toRevoke, setToRevoke] = useState<ApiKey | null>(null)
    const [revoking, setRevoking] = useState(false)
    const [revokeError, setRevokeError] = useState<string | null>(null)

    useEffect(() => {
        let cancelled = false
        api
            .listApiKeys()
            .then((list) => {
                if (!cancelled) setKeys(sortKeys(list))
            })
            .catch(() => {
                if (!cancelled) setLoadError(true)
            })
        return () => {
            cancelled = true
        }
    }, [api])

    async function handleRevoke() {
        if (!toRevoke) return
        setRevoking(true)
        setRevokeError(null)
        try {
            await api.revokeApiKey(toRevoke.id)
            const now = new Date().toISOString()
            setKeys((prev) => sortKeys((prev ?? []).map((k) => (k.id === toRevoke.id ? { ...k, revoked_at: now } : k))))
            setToRevoke(null)
        } catch (err) {
            setRevokeError(err instanceof TriageError ? err.message : "Couldn't revoke the key. Please try again.")
        }
    }

  return (
    <>
      <SettingsCard 
        title="API keys" 
        description="Keys your apps use to call Triage. Each key is shown only once, when you create it."
        action={
            <button
                type="button"
                onClick={() => setCreating(true)}
                className="inline-flex h-10 items-center gap-1.5 rounded-[10px] bg-[#2563EB] px-4 text-[13px] font-semibold text-white transition-colors hover:bg-[#1D4ED8] max-md:min-h-11 pointer-coarse:min-h-11"
            >
                <Plus className="h-4 w-4" />
                Create key
            </button>
        }
      >
        {loadError ? (
            <p className="px-6 py-6 text-sm text-[#FCA5A5]">Couldn't load your keys. Refresh the page to try again.</p>
        ) : !keys ? (
            <p className="flex items-center gap-2 px-6 py-6 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" /> Loading your keys…
            </p>
        ) : keys.length === 0 ? (
            <div className="flex flex-col items-center px-6 py-12 text-center">
                <span className="flex h-11 w-11 items-center justify-center rounded-xl bg-[#1A1C26] text-[#60A5FA]">
                    <KeyRound className="h-5 w-5" />
                </span>
                <p className="mt-4 text-sm font-medium">No keys yet</p>
                <p className="mt-1 max-w-xs text-[13px] text-muted-foreground">
                    Create one to connect your app to Triage
                </p>
            </div>
        ) : (
            <div className="overflow-x-auto">
                <table className="w-full min-w-[620px] border-collapse">
                    <thead>
                        <tr className="text-left text-xs font-medium text-muted-foreground">
                            <th scope="col" className="px-6 py-2.5 font-medium">Name</th>
                            <th scope="col" className="px-6 py-2.5 font-medium">Key</th>
                            <th scope="col" className="px-6 py-2.5 font-medium">Created</th>
                            <th scope="col" className="px-6 py-2.5 font-medium">Last used</th>
                            <th scope="col" className="px-6 py-2.5">
                                <span className="sr-only">Actions</span>
                            </th>
                        </tr>
                    </thead>
                    <tbody>
                        {keys.map((key) => (
                            <tr key={key.id} className={cn("border-t border-[#1F2130]", key.revoked_at && "opacity-55")}>
                                <td className="px-6 py-3.5 text-sm font-medium">{key.name}</td>
                                <td className="px-3 py-3.5 font-mono text-[13px] text-[#C9CDD8]">{key.public_id}…</td>
                                <td className="px-3 py-3.5 text-[13px] text-muted-foreground">{shortDate(key.created_at)}</td>
                                <td className="px-3 py-3.5 text-[13px] text-muted-foreground">
                                    {key.last_used_at ? relativeTime(key.last_used_at) : "Never"}
                                </td>
                                <td className="px-6 py-3.5 text-right">
                                    {key.revoked_at ? (
                                        <StatusBadge kind="revoked" />
                                    ) : (
                                        <button
                                        type="button"
                                        onClick={() => {
                                            setRevokeError(null)
                                            setToRevoke(key)
                                        }}
                                        className="h-9 rounded-[10px] border border-[#F87171]/30 px-3 text-[13px] font-medium text-[#F87171] transition-colors hover:bg-[#EF4444]/10"
                                        >
                                        Revoke
                                        </button>
                                    )}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        )}
      </SettingsCard>

      <CreateKeyDialog
        open={creating}
        onClose={() => setCreating(false)}
        onCreated={(key) => setKeys((prev) => sortKeys([key, ...(prev ?? [])]))}
      />

      <ConfirmDialog
        open={!!toRevoke}
        tone="danger"
        title="Revoke this key?"
        description={
          <>
            Apps using <span className="font-medium text-foreground">{toRevoke?.name}</span> ({toRevoke?.public_id}…)
            stop working immediately. This can't be undone.
          </>
        }
        confirmLabel="Revoke key"
        busy={revoking}
        error={revokeError}
        onConfirm={handleRevoke}
        onCancel={() => setToRevoke(null)}
      />
    </>
  )
}
