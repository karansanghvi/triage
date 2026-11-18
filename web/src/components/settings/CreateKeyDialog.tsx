/* eslint-disable @typescript-eslint/no-unused-vars */
/* eslint-disable react-hooks/set-state-in-effect */
import { useEffect, useState, type FormEvent } from "react"
import { AlertTriangle, Check, X } from "lucide-react"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import CopyButton from "@/components/common/CopyButton"
import { useAccountApi } from "@/hooks/useAccountApi"
import type { ApiKey } from "@/lib/account"
import { TriageError } from "@/lib/errors"

type CreateKeyDialogProps = {
  open: boolean
  onClose: () => void
  onCreated: (key: ApiKey) => void
}

export default function CreateKeyDialog({ open, onClose, onCreated }: CreateKeyDialogProps) {
  const api = useAccountApi()
  const [name, setName] = useState("")
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [created, setCreated] = useState<{ name: string; key: string } | null>(null)

  // Start fresh every time the popup opens
  useEffect(() => {
    if (open) {
      setName("")
      setError(null)
      setCreated(null)
      setSubmitting(false)
    }
  }, [open])

  async function handleCreate(e: FormEvent) {
    e.preventDefault()
    const trimmed = name.trim()
    if (!trimmed) {
      setError('Give the key a name, like "my-app".')
      return
    }
    setError(null)
    setSubmitting(true)
    try {
      const result = await api.createApiKey(trimmed)
      const { key, note: _note, ...listed } = result
      onCreated(listed) // add it to the table (without the secret)
      setCreated({ name: result.name, key }) // show the secret once, here
    } catch (err) {
      if (err instanceof TriageError && err.code === "too_many_api_keys") {
        setError("You've reached the limit of 20 active keys. Revoke one you no longer use first.")
      } else if (err instanceof TriageError && err.retryAfter) {
        setError(`You've created a lot of keys recently. Try again in ${Math.ceil(err.retryAfter / 60)} minutes.`)
      } else if (err instanceof TriageError) {
        setError(err.message)
      } else {
        setError("Couldn't create the key. Please try again.")
      }
    } finally {
      setSubmitting(false)
    }
  }

  // Step 1 can be closed freely; step 2 only with "Done", so nobody loses the key by accident
  function handleOpenChange(isOpen: boolean) {
    if (!isOpen && !created && !submitting) onClose()
  }

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent
        showCloseButton={false}
        className="grid-cols-[minmax(0,1fr)] gap-0 overflow-hidden p-0 sm:max-w-[480px] sm:rounded-[20px]"
      >
        {!created ? (
          /* ---------- Step 1: name the key ---------- */
          <form onSubmit={handleCreate} noValidate className="min-w-0">
            <div className="relative min-w-0 p-7 pb-2">
              <button
                type="button"
                onClick={onClose}
                aria-label="Close"
                className="absolute right-3.5 top-3.5 flex h-10 w-10 items-center justify-center rounded-[10px] text-[#8B90A0] hover:bg-accent hover:text-foreground"
              >
                <X className="h-[18px] w-[18px]" />
              </button>
              <DialogHeader className="text-left">
                <DialogTitle className="text-xl font-semibold">Create API key</DialogTitle>
                <DialogDescription className="mr-10 mt-1.5 text-sm leading-relaxed text-[#A7ACBA]">
                  Use a separate key for each app, so you can revoke one without breaking the others.
                </DialogDescription>
              </DialogHeader>

              <label htmlFor="ck-name" className="mt-5 block text-sm font-medium">
                Name
              </label>
              <input
                id="ck-name"
                autoFocus
                maxLength={60}
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="my-app"
                aria-describedby="ck-hint"
                className="mt-2 h-12 w-full rounded-xl border border-[#1F2130] bg-[#0A0A0F] px-3.5 text-[15px] outline-none transition-colors placeholder:text-[#7C8192] focus:border-[#3B82F6] focus:ring-4 focus:ring-[#3B82F6]/15"
              />
              <p id="ck-hint" className="mt-2 text-xs text-[#8B90A0]">
                For example "production" or "my-app". Only you see it.
              </p>
              {error && (
                <p
                  role="alert"
                  className="mt-4 rounded-xl border border-[#F87171]/35 bg-[#EF4444]/8 px-3.5 py-2.5 text-[13px] text-[#FCA5A5]"
                >
                  {error}
                </p>
              )}
            </div>
            <div className="mt-5 flex justify-end gap-2.5 border-t border-[#1F2130] px-7 py-4">
              <button
                type="button"
                onClick={onClose}
                className="h-11 rounded-xl border border-[#2A2D3D] px-[18px] text-sm font-medium transition-colors hover:bg-accent"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={submitting}
                className="h-11 rounded-xl bg-[#2563EB] px-[18px] text-sm font-semibold text-white transition-colors hover:bg-[#1D4ED8] disabled:opacity-60"
              >
                {submitting ? "Creating…" : "Create key"}
              </button>
            </div>
          </form>
        ) : (
          /* ---------- Step 2: show the key once ---------- */
          <div className="min-w-0">
            <div className="min-w-0 p-7 pb-2">
              <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-[#22C55E]/14 text-[#4ADE80]">
                <Check className="h-[22px] w-[22px]" strokeWidth={2.5} />
              </div>
              <DialogHeader className="text-left">
                <DialogTitle className="mt-[18px] text-xl font-semibold">Save your API key</DialogTitle>
                <DialogDescription className="mt-1.5 text-sm leading-relaxed text-[#A7ACBA]">
                  This is the only time you'll see <span className="font-medium text-foreground">{created.name}</span>'s
                  key. Store it somewhere safe, like your app's environment variables.
                </DialogDescription>
              </DialogHeader>

              {/* The key wraps onto two lines instead of pushing the button out */}
              <div className="mt-5 flex min-w-0 items-start gap-2">
                <code className="min-w-0 flex-1 break-all rounded-xl border border-[#2A2D3D] bg-[#0A0A0F] px-3.5 py-3 font-mono text-[13px] leading-relaxed">
                  {created.key}
                </code>
                <CopyButton text={created.key} label="Copy" variant="primary" className="h-12" />
              </div>

              <p className="mt-4 flex items-start gap-2 rounded-xl border border-[#F59E0B]/25 bg-[#F59E0B]/8 px-3.5 py-3 text-[13px] leading-relaxed text-[#FCD34D]">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                Never share it or put it in front-end code. If you lose it, revoke it and create a new one.
              </p>
            </div>
            <div className="mt-5 flex justify-end border-t border-[#1F2130] px-7 py-4">
              <button
                type="button"
                onClick={onClose}
                className="h-11 rounded-xl bg-[#2563EB] px-[18px] text-sm font-semibold text-white transition-colors hover:bg-[#1D4ED8]"
              >
                Done
              </button>
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}