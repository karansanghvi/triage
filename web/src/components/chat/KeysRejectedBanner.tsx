import { KeyRound } from "lucide-react"
import { Link } from "react-router-dom"

// A logged-in user on the free demo because every saved key was rejected
export default function KeysRejectedBanner() {
  return (
    <div className="mb-3 flex items-center gap-4 rounded-2xl border border-warning/30 bg-warning/10 p-4">
      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-warning/15">
        <KeyRound className="h-4 w-4 text-warning" />
      </div>
      <p className="min-w-0 flex-1 text-sm font-medium">You're on the free demo because your saved keys were rejected.</p>
      <Link
        to="/settings/provider-keys"
        className="flex shrink-0 items-center gap-2 rounded-lg bg-primary px-3 py-2 text-xs font-medium text-primary-foreground hover:bg-primary/90"
      >
        <KeyRound className="h-3.5 w-3.5" />
        Fix your keys
      </Link>
    </div>
  )
}
