import { Link, NavLink } from "react-router-dom"
import { LayoutDashboard, MessageSquare, Settings, Split, KeyRound, type LucideIcon } from "lucide-react"
import { cn } from "@/lib/utils"
import { useVisitorName } from "@/hooks/useVisitorName"

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

// Fake values for now; later these come from GET /v1/demo/status
const demo = { used: 3, limit: 10 }

export default function Sidebar() {
  const { name } = useVisitorName()
  const remaining = demo.limit - demo.used
  const percentUsed = (demo.used / demo.limit) * 100

  return (
    <aside className="flex w-64 flex-col border-r border-border bg-card p-4">
      {/* Logo */}
      <div className="mb-10 flex items-center gap-3 px-2">
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

      <div className="mt-auto flex flex-col gap-3">
        {/* Demo status card */}
        <div className="rounded-xl border border-border bg-background p-4">
          <p className="text-sm font-medium">Free demo</p>
          <p className="mt-1 text-xs text-muted-foreground">
            {remaining} of {demo.limit} requests left today
          </p>

          <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-muted">
            <div
              className="h-full rounded-full bg-primary transition-all"
              style={{ width: `${percentUsed}%` }}
            />
          </div>

          <Link
            to="/settings"
            className="mt-4 flex items-center justify-center gap-2 rounded-lg bg-primary px-3 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
          >
            <KeyRound className="h-4 w-4" />
            Add your keys
          </Link>
        </div>

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