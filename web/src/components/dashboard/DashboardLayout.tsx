import { cn } from "@/lib/utils";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import RangeSelector from "./RangeSelector";

const TABS = [
  { to: "/dashboard/benchmark", label: "Benchmark" },
  { to: "/dashboard/overview", label: "Overview" },
  { to: "/dashboard/usage", label: "My usage" },
]

const SUBTITLES: Record<string, string> = {
  "/dashboard/benchmark": "How well Triage routes, measured on held-out test questions.",
  "/dashboard/overview": "Live routing across everyone using Triage. Nobody's details are shown.",
  "/dashboard/usage": "Your own requests, costs and savings.",
}

export default function DashboardLayout() {

  const { pathname, search } = useLocation()
  const showRange = pathname !== "/dashboard/benchmark"

  return (
    <>
      <div className="relative mx-auto max-w-[1180px] overflow-x-clip pb-16">
        {/* Soft blue light, top right (kept inside the page so nothing scrolls sideways) */}
        <div
          aria-hidden="true"
          className="pointer-events-none absolute -top-56 right-0 -z-10 h-[520px] w-[760px] max-w-full rounded-full"
          style={{ background: "radial-gradient(closest-side, rgba(37,99,235,0.14), rgba(10,10,15,0))" }}
        />

        <header>
          <h1 className="text-[30px] font-semibold tracking-[-0.01em]">Dashboard</h1>
          <p className="mt-1.5 text-sm text-muted-foreground">{SUBTITLES[pathname] ?? ""}</p>
        </header>

        <div className="mt-6 flex flex-wrap items-center justify-between gap-4">
          <nav aria-label="Dashboard" className="flex gap-1.5 overflow-x-auto rounded-[14px] border border-[#1A1C26] bg-[#0E0F15] p-1">
            {TABS.map(({ to, label }) => (
              <NavLink
                key={to}
                // Keep the chosen range when switching tabs 
                to={{ pathname: to, search }}
                className={({ isActive }) =>
                  cn(
                    "shrink-0 rounded-[10px] border px-4 py-2 text-sm font-medium transition-colors max-md:min-h-11 pointer-coarse:min-h-11",
                    isActive
                      ? "border-[#2A2D3D] bg-[#1A1C26] text-foreground"
                      : "border-transparent text-muted-foreground hover:text-foreground"
                  )
                }
              >
                {label}
              </NavLink>
            ))}
          </nav>
          {showRange && <RangeSelector />}
        </div>

        <div className="mt-6 flex flex-col gap-5">
          <Outlet />
        </div>
      </div>
    </>
  )
}
