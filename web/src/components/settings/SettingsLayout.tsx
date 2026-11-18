import { cn } from "@/lib/utils";
import { KeyRound, Layers, UserRound, Zap, type LucideIcon } from "lucide-react";
import { NavLink, Outlet } from "react-router-dom";

type MenuItem = { to: string, label: string, icon: LucideIcon }

const MENU: MenuItem[] = [
    { to: "/settings/get-started", label: "Get Started", icon: Zap },
    { to: "/settings/api-keys", label: "API Keys", icon: KeyRound },
    { to: "/settings/provider-keys", label: "Provider Keys", icon: Layers },
    { to: "/settings/account", label: "Account", icon: UserRound }
]

export default function SettingsLayout() {
  return (
    <>
      <div className="mx-auto max-w-[1100px] pb-16">
        <header>
            <h1 className="text-[30px] font-semibold tracking-[-0.01em]">Settings</h1>
            <p className="mt-1.5 text-sm text-muted-foreground">Connect your apps, manage keys and your account.</p>
        </header>

        <div className="mt-7 flex flex-wrap items-start gap-8">
            {/* Left menu */}
            <nav aria-label="Settings" className="flex w-full flex-row gap-1 overflow-x-auto md:w-[210px] md:flex-col">
                {MENU.map(({ to, label, icon: Icon }) => (
                    <NavLink
                        key={to}
                        to={to}
                        className={({ isActive }) =>
                            cn(
                                "flex shrink-0 items-center gap-2.5 rounded-[10px] px-3 py-2.5 text-sm font-medium transition-colors",
                                isActive ? "bg-[#1A1C26] text-foreground" : "text-muted-foreground hover:bg-accent/60 hover:text-foreground"
                            )
                        }
                    >
                        <Icon className="h-4 w-4" />
                        {label}
                    </NavLink>
                ))}
            </nav>

            {/* The selected view */}
            <div className="flex min-w-0 flex-[1_1_560px] flex-col gap-[22px]">
                <Outlet />
            </div>
        </div>
      </div>
    </>
  )
}
