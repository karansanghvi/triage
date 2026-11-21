import { useEffect, useRef, useState, type FormEvent } from "react"
import { useNavigate } from "react-router-dom"
import { Check, Loader2 } from "lucide-react"
import { useAuth } from "@/context/AuthContext"
import { useChats } from "@/context/ChatsContext"
import { useAccountApi } from "@/hooks/useAccountApi"
import { TriageError } from "@/lib/errors"
import { cn } from "@/lib/utils"
import { PasswordInput, fieldBorder, inputClass } from "../auth/fields"
import ConfirmDialog from "../common/ConfirmDialog"
import SettingsCard from "./SettingsCard"

const NAME_MAX = 80
const PASSWORD_MIN = 10
const PASSWORD_MAX = 128

const saveButton =
  "mt-2 inline-flex h-12 shrink-0 items-center justify-center gap-2 rounded-[10px] bg-[#2563EB] px-4 text-[13px] font-semibold text-white transition-colors hover:bg-[#1D4ED8] disabled:cursor-not-allowed disabled:opacity-60"
const labelClass = "block text-sm font-medium"
const errorClass = "mt-2 text-xs text-[#FCA5A5]"

// The same wording for a wrong password, a rate limit, or anything else the backend says
function passwordError(err: unknown, fallback: string): string {
  if (err instanceof TriageError && err.code === "incorrect_password") return "That password isn't right."
  if (err instanceof TriageError && err.retryAfter) {
    return `Too many attempts. Try again in ${Math.ceil(err.retryAfter / 60)} minutes.`
  }
  if (err instanceof TriageError) return err.message
  return fallback
}

function chars(text: string): number {
  return [...text].length
}

// ---------- Profile ----------

function ProfileCard() {
  const { user } = useAuth()
  const api = useAccountApi()
  const current = user?.display_name ?? ""
  const [name, setName] = useState(current)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const savedTimer = useRef<number | undefined>(undefined)

  useEffect(() => () => window.clearTimeout(savedTimer.current), [])

  const changed = name.trim() !== current

  async function handleSave(e: FormEvent) {
    e.preventDefault()
    const next = name.trim()
    if (!next) {
      setError("Enter a name.")
      return
    }
    setError(null)
    setSaving(true)
    try {
      await api.updateAccount({ display_name: next })
      setName(next)
      setSaved(true)
      window.clearTimeout(savedTimer.current)
      savedTimer.current = window.setTimeout(() => setSaved(false), 2000)
    } catch (err) {
      setError(err instanceof TriageError ? err.message : "Couldn't save your name. Please try again.")
    } finally {
      setSaving(false)
    }
  }

  return (
    <SettingsCard title="Profile" description="How Triage greets you.">
      <form onSubmit={handleSave} noValidate className="px-6 py-5">
        <label htmlFor="account-name" className={labelClass}>
          Display name
        </label>
        <div className="flex flex-wrap items-start gap-2">
          <input
            id="account-name"
            maxLength={NAME_MAX}
            autoComplete="name"
            value={name}
            onChange={(e) => {
              setName(e.target.value)
              setSaved(false)
              setError(null)
            }}
            aria-invalid={!!error}
            aria-describedby={error ? "account-name-error" : undefined}
            disabled={saving}
            className={cn(inputClass, fieldBorder(!!error), "min-w-0 flex-[1_1_240px] disabled:opacity-60")}
          />
          <button type="submit" disabled={!changed || saving} className={saveButton}>
            {saving && <Loader2 className="h-4 w-4 animate-spin" />}
            {saving ? "Saving…" : "Save"}
          </button>
        </div>
        {error && (
          <p id="account-name-error" role="alert" className={errorClass}>
            {error}
          </p>
        )}
        {saved && !error && (
          <p className="mt-2 flex items-center gap-1.5 text-xs text-[#4ADE80]">
            <Check className="h-3.5 w-3.5" strokeWidth={2.5} />
            Saved
          </p>
        )}

        <p className="mt-5 text-sm font-medium">Email</p>
        <p className="mt-1 text-sm text-[#A7ACBA]">{user?.email}</p>
      </form>
    </SettingsCard>
  )
}

// ---------- Password ----------

function PasswordCard() {
  const api = useAccountApi()
  const [currentPassword, setCurrentPassword] = useState("")
  const [newPassword, setNewPassword] = useState("")
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<{ field: "current" | "new" | null; message: string } | null>(null)
  const [done, setDone] = useState(false)

  function check(): { field: "current" | "new"; message: string } | null {
    if (!currentPassword) return { field: "current", message: "Enter your current password." }
    const length = chars(newPassword)
    if (length < PASSWORD_MIN || length > PASSWORD_MAX) {
      return { field: "new", message: `Your new password must be ${PASSWORD_MIN} to ${PASSWORD_MAX} characters.` }
    }
    if (newPassword === currentPassword) {
      return { field: "new", message: "Your new password must be different from the current one." }
    }
    return null
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setDone(false)
    const problem = check()
    if (problem) {
      setError(problem)
      return
    }
    setError(null)
    setBusy(true)
    try {
      await api.changePassword(currentPassword, newPassword)
      setCurrentPassword("")
      setNewPassword("")
      setDone(true)
    } catch (err) {
      const wrong = err instanceof TriageError && err.code === "incorrect_password"
      setError({ field: wrong ? "current" : null, message: passwordError(err, "Couldn't change your password. Please try again.") })
    } finally {
      setBusy(false)
    }
  }

  return (
    <SettingsCard title="Password" description="Changing it logs you out on every other device.">
      <form onSubmit={handleSubmit} noValidate className="px-6 py-5">
        <label htmlFor="account-current-password" className={labelClass}>
          Current password
        </label>
        <PasswordInput
          id="account-current-password"
          autoComplete="current-password"
          value={currentPassword}
          onChange={(e) => {
            setCurrentPassword(e.target.value)
            setDone(false)
          }}
          invalid={error?.field === "current"}
          aria-invalid={error?.field === "current"}
          disabled={busy}
        />

        <label htmlFor="account-new-password" className={cn(labelClass, "mt-4")}>
          New password
        </label>
        <PasswordInput
          id="account-new-password"
          autoComplete="new-password"
          value={newPassword}
          onChange={(e) => {
            setNewPassword(e.target.value)
            setDone(false)
          }}
          invalid={error?.field === "new"}
          aria-invalid={error?.field === "new"}
          aria-describedby="account-new-password-hint"
          disabled={busy}
        />
        <p id="account-new-password-hint" className="mt-2 text-xs text-[#8B90A0]">
          {PASSWORD_MIN} to {PASSWORD_MAX} characters.
        </p>

        {error && (
          <p role="alert" className={errorClass}>
            {error.message}
          </p>
        )}
        <p role="status" className="text-xs text-[#4ADE80] empty:hidden">
          {done ? "Password changed. Other devices have been logged out." : ""}
        </p>

        <button type="submit" disabled={busy} className={cn(saveButton, "mt-4")}>
          {busy && <Loader2 className="h-4 w-4 animate-spin" />}
          {busy ? "Changing…" : "Change password"}
        </button>
      </form>
    </SettingsCard>
  )
}

// ---------- Data ----------

function DataCard() {
  const { chats, clearAll } = useChats()
  const [confirming, setConfirming] = useState(false)
  const count = chats.length

  return (
    <SettingsCard title="Data" description="Your chats are saved in this browser only.">
      <div className="flex flex-wrap items-center justify-between gap-3 px-6 py-5">
        <p className="text-sm text-[#C9CDD8]">
          {count === 0
            ? "No chats saved on this device"
            : `${count} ${count === 1 ? "chat" : "chats"} saved on this device`}
        </p>
        <button
          type="button"
          onClick={() => setConfirming(true)}
          disabled={count === 0}
          className="h-10 rounded-[10px] border border-[#2A2D3D] bg-[#1A1C26] px-3.5 text-[13px] font-medium max-md:min-h-11 pointer-coarse:min-h-11 transition-colors hover:border-[#3B4058] disabled:cursor-not-allowed disabled:opacity-50"
        >
          Clear all chats
        </button>
      </div>

      <ConfirmDialog
        open={confirming}
        tone="danger"
        title="Clear all chats?"
        description="Every saved chat in this browser is deleted. This can't be undone."
        confirmLabel="Clear all chats"
        onConfirm={() => {
          clearAll()
          setConfirming(false)
        }}
        onCancel={() => setConfirming(false)}
      />
    </SettingsCard>
  )
}

// ---------- Delete account ----------

function DeleteAccountCard() {
  const api = useAccountApi()
  const { expireSession } = useAuth()
  const { clearAll } = useChats()
  const navigate = useNavigate()
  const [open, setOpen] = useState(false)
  const [password, setPassword] = useState("")
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  function openDialog() {
    setPassword("")
    setError(null)
    setOpen(true)
  }

  async function handleDelete() {
    if (!password || busy) return
    setBusy(true)
    setError(null)
    try {
      await api.deleteAccount(password)
      clearAll()                  // the chats belonged to that account
      expireSession()
      navigate("/", { replace: true })
    } catch (err) {
      setError(passwordError(err, "Couldn't delete your account. Please try again."))
      setBusy(false)
    }
  }

  return (
    <SettingsCard
      tone="danger"
      title="Delete account"
      description="Permanently deletes your account, provider keys and API keys. Apps using your keys stop working."
    >
      <div className="px-6 py-5">
        <button
          type="button"
          onClick={openDialog}
          className="h-10 rounded-[10px] border border-[#F87171]/40 px-4 text-[13px] font-semibold max-md:min-h-11 pointer-coarse:min-h-11 text-[#FCA5A5] transition-colors hover:border-[#F87171]/70 hover:bg-[#EF4444]/10"
        >
          Delete account
        </button>
      </div>

      <ConfirmDialog
        open={open}
        tone="danger"
        title="Delete your account?"
        description="Your account, provider keys and API keys are deleted for good. Apps using your keys stop working."
        confirmLabel="Delete account"
        busy={busy}
        confirmDisabled={!password}
        error={error}
        onConfirm={handleDelete}
        onCancel={() => setOpen(false)}
      >
        <label htmlFor="account-delete-password" className={labelClass}>
          Enter your password to confirm
        </label>
        <PasswordInput
          id="account-delete-password"
          autoFocus
          autoComplete="current-password"
          value={password}
          onChange={(e) => {
            setPassword(e.target.value)
            setError(null)
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault()
              handleDelete()
            }
          }}
          invalid={!!error}
          aria-invalid={!!error}
          disabled={busy}
        />
      </ConfirmDialog>
    </SettingsCard>
  )
}

export default function AccountPage() {
  return (
    <>
      <ProfileCard />
      <PasswordCard />
      <DataCard />
      <DeleteAccountCard />
    </>
  )
}
