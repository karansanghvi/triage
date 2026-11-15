import { Link, NavLink } from "react-router-dom"
import { LayoutDashboard, MessageSquare, Settings, Split, KeyRound, type LucideIcon } from "lucide-react"
import { cn } from "@/lib/utils"
import { useVisitorName } from "@/hooks/useVisitorName"
import { useDemoStatus } from "@/context/DemoStatusContext"

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
  const { name } = useVisitorName()

  return (
    <aside className="flex h-full w-64 flex-col overflow-y-auto border-r border-border bg-card p-4">
      {/* Logo */}
      <div className="mb-8 flex items-center gap-3 px-2">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary">
          <Split className="h-5 w-5 text-primary-foreground" />
        </div>
        <span className="text-xl font-semibold">Triage</span>
      </div>

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

      <div className="mt-auto flex flex-col gap-3 pt-6">
        <DemoCard />

        {/* Profile row: click to change the name in Settings */}
        <Link
          to="/settings"
          className="flex items-center gap-3 rounded-lg px-2 py-2 transition-colors hover:bg-accent"
        >
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-primary/15 text-sm font-semibold text-primary">
            {name ? name[0].toUpperCase() : "G"}
          </div>
          <p className="truncate text-sm font-medium">{name || "Guest"}</p>
        </Link>
      </div>
    </aside>
  )
}

function formatResetTime(resetsAt: string): string | null {
  if (!resetsAt) return null
  const date = new Date(resetsAt)
  if (Number.isNaN(date.getTime())) return null
  return date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })
}

function DemoCard() {
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
          className={cn(
            "h-full rounded-full transition-all duration-500",
            exhausted ? "bg-warning" : "bg-primary"
          )}
          style={{ width: `${loading ? 0 : percentUsed}%` }}
        />
      </div>

      {resetTime && exhausted && <p className="mt-2 text-xs text-muted-foreground">Resets at {resetTime}</p>}

      <Link
        to="/settings"
        className="mt-4 flex items-center justify-center gap-2 rounded-lg bg-primary px-3 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
      >
        <KeyRound className="h-4 w-4" />
        Add your keys
      </Link>
    </div>
  )
}