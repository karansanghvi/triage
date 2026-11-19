import { Link, NavLink, useLocation, useNavigate } from "react-router-dom"
import { KeyRound, LayoutDashboard, LogOut, MessageSquare, Settings, Split, type LucideIcon } from "lucide-react"
import { cn } from "@/lib/utils"
import { useDisplayName } from "@/hooks/useDisplayName"
import { useAuth } from "@/context/AuthContext"
import { useDemoStatus } from "@/context/DemoStatusContext"
import { useProviderKeys } from "@/context/ProviderKeysContext"
import ChatHistory from "./ChatHistory"

type NavItem = {
  to: string
  label: string
  icon: LucideIcon
}

const navItems: NavItem[] = [
  { to: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { to: "/chat", label: "Chat", icon: MessageSquare },
  { to: "/settings", label: "Settings", icon: Settings },
]

export default function Sidebar() {
  const { pathname } = useLocation()
  const onChatPage = pathname.startsWith("/chat")
  const { user, loading } = useAuth()

  return (
    <aside className="flex h-full w-64 flex-col overflow-hidden border-r border-border bg-card p-4">
      {/* Logo */}
      <Link to="/" className="mb-8 flex items-center gap-3 px-2">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary">
          <Split className="h-5 w-5 text-primary-foreground" />
        </div>
        <span className="text-xl font-semibold">Triage</span>
      </Link>

      {/* Navigation */}
      <nav className="flex flex-col gap-1">
        {navItems.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                isActive
                  ? "bg-primary/15 text-primary"
                  : "text-muted-foreground hover:bg-accent hover:text-foreground"
              )
            }
          >
            <Icon className="h-4 w-4" />
            {label}
          </NavLink>
        ))}
      </nav>

      {/* Recent chats only on the Chat page; elsewhere, empty space keeps the bottom pinned */}
      {onChatPage ? <ChatHistory /> : <div className="flex-1" />}

      <div className="flex flex-col gap-3 pt-4">
        {loading ? null : user ? <AccountBottom /> : <GuestBottom />}
      </div>
    </aside>
  )
}

/* ---------- Logged in: the demo card or "Using your keys", then the account row ---------- */

function AccountBottom() {
  const { loading, error, hasUsableKey, allRejected } = useProviderKeys()

  // While loading (or if the keys couldn't be loaded), nothing extra: no flash of the wrong card
  const known = !loading && !error
  return (
    <>
      {known && (hasUsableKey ? <OwnKeysCard /> : <DemoCard forAccount allRejected={allRejected} />)}
      <AccountRow />
    </>
  )
}

function OwnKeysCard() {
  const { usableProviders } = useProviderKeys()
  return (
    <div className="rounded-xl border border-border bg-background p-4">
      <div className="flex items-center justify-between gap-3">
        <p className="flex items-center gap-2 whitespace-nowrap text-sm font-medium">
          <span aria-hidden="true" className="h-2 w-2 shrink-0 rounded-full bg-[#4ADE80] shadow-[0_0_6px_rgba(74,222,128,0.6)]" />
          Using your keys
        </p>
        <Link
          to="/settings/provider-keys"
          aria-label="Manage provider keys"
          className="shrink-0 text-xs text-muted-foreground transition-colors hover:text-foreground"
        >
          Manage
        </Link>
      </div>
      <ul className="mt-2 flex flex-wrap gap-1.5">
        {usableProviders.map((name) => (
          <li
            key={name}
            className="rounded-full border border-[#2A2D3D] bg-[#1A1C26] px-2 py-0.5 text-[11px] text-[#C9CDD8]"
          >
            {name}
          </li>
        ))}
      </ul>
    </div>
  )
}

/* ---------- Logged in: name tag with Log out ---------- */

function AccountRow() {
  const { user, logout } = useAuth()
  const name = useDisplayName()
  const navigate = useNavigate()

  async function handleLogout() {
    await logout()
    navigate("/", { replace: true })
  }

  return (
    <div className="flex items-center gap-2 rounded-lg px-1 py-1">
      <Link to="/settings/account" className="flex min-w-0 flex-1 items-center gap-3 rounded-lg px-1 py-1 transition-colors hover:bg-accent">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-primary/15 text-sm font-semibold text-primary">
          {name ? name[0].toUpperCase() : "?"}
        </div>
        <div className="min-w-0">
          <p className="truncate text-sm font-medium">{name}</p>
          <p className="truncate text-xs text-muted-foreground">{user?.email}</p>
        </div>
      </Link>
      <button
        onClick={handleLogout}
        title="Log out"
        aria-label="Log out"
        className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
      >
        <LogOut className="h-4 w-4" />
      </button>
    </div>
  )
}

/* ---------- Not logged in: demo card + sign up / log in ---------- */

function GuestBottom() {
  return (
    <>
      <DemoCard />
      <div className="flex gap-2">
        <Link
          to="/login"
          className="flex h-9 flex-1 items-center justify-center rounded-lg border border-border text-sm font-medium transition-colors hover:bg-accent"
        >
          Log in
        </Link>
        <Link
          to="/signup"
          className="flex h-9 flex-1 items-center justify-center rounded-lg bg-primary text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
        >
          Sign up
        </Link>
      </div>
    </>
  )
}

function formatResetTime(resetsAt: string): string | null {
  if (!resetsAt) return null
  const date = new Date(resetsAt)
  if (Number.isNaN(date.getTime())) return null
  return date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })
}

type DemoCardProps = {
  forAccount?: boolean       // logged in without a working key: point to the key settings, not sign-up
  allRejected?: boolean      // their saved keys were all rejected
}

function DemoCard({ forAccount = false, allRejected = false }: DemoCardProps) {
  const { status, loading } = useDemoStatus()

  // Demo is off or Triage is unreachable: don't show a card with made-up numbers
  if (!loading && !status) return null

  const limit = status?.limit ?? 10
  const remaining = status?.remaining ?? limit
  const percentUsed = ((limit - remaining) / limit) * 100
  const resetTime = status ? formatResetTime(status.resetsAt) : null
  const exhausted = remaining === 0 || status?.paused

  let line: string
  if (loading) line = "Checking your free requests…"
  else if (status?.paused) line = "The free demo is paused for today"
  else if (remaining === 0) line = "No free requests left today"
  else line = `${remaining} of ${limit} requests left today`

  return (
    <div className="rounded-xl border border-border bg-background p-4">
      <p className="text-sm font-medium">Free demo</p>
      <p className="mt-1 text-xs text-muted-foreground">{line}</p>

      <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-muted">
        <div
          className={cn("h-full rounded-full transition-all duration-500", exhausted ? "bg-warning" : "bg-primary")}
          style={{ width: `${loading ? 0 : percentUsed}%` }}
        />
      </div>

      {resetTime && exhausted && <p className="mt-2 text-xs text-muted-foreground">Resets at {resetTime}</p>}

      {forAccount && allRejected && (
        <p className="mt-3 text-xs text-warning">
          Your saved keys were rejected.{" "}
          <Link to="/settings/provider-keys" className="font-medium underline underline-offset-2 hover:no-underline">
            Fix your keys
          </Link>
        </p>
      )}

      <Link
        to={forAccount ? "/settings/provider-keys" : "/signup"}
        className="mt-4 flex items-center justify-center gap-2 rounded-lg bg-primary/15 px-3 py-2 text-sm font-medium text-primary transition-colors hover:bg-primary/25"
      >
        <KeyRound className="h-4 w-4" />
        {forAccount ? "Add your keys" : "Use your own keys"}
      </Link>
    </div>
  )
}