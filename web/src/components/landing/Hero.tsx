import { Link } from "react-router-dom"
import { ArrowRight } from "lucide-react"
import { useAuth } from "@/context/AuthContext"
import { cn } from "@/lib/utils"
import RoutingCard from "./RoutingCard";

function RouteLines() {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 1440 720"
      preserveAspectRatio="none"
      className="pointer-events-none absolute -top-10 left-0 h-[760px] w-full"
    >
      <defs>
        <linearGradient id="routeFade" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stopColor="#3B82F6" stopOpacity="0" />
          <stop offset="0.35" stopColor="#3B82F6" stopOpacity="0.28" />
          <stop offset="0.75" stopColor="#60A5FA" stopOpacity="0.22" />
          <stop offset="1" stopColor="#60A5FA" stopOpacity="0" />
        </linearGradient>
      </defs>

      {/* One path in, branching into three */}
      <path d="M-40 470 C 260 470, 420 470, 640 470 C 860 470, 960 250, 1480 190" fill="none" stroke="url(#routeFade)" strokeWidth="1.4" />
      <path d="M640 470 C 880 470, 1080 480, 1480 470" fill="none" stroke="url(#routeFade)" strokeWidth="1.4" />
      <path d="M640 470 C 860 470, 980 660, 1480 720" fill="none" stroke="url(#routeFade)" strokeWidth="1.4" />

      {/* A slow pulse travelling along the main path */}
      <path
        d="M-40 470 C 260 470, 420 470, 640 470 C 860 470, 960 250, 1480 190"
        fill="none"
        stroke="#60A5FA"
        strokeOpacity="0.35"
        strokeWidth="1.6"
        className="triage-drift"
      />

      {/* Two extra faint lines for depth */}
      <path d="M-40 300 C 300 300, 520 320, 760 360 C 980 400, 1180 330, 1480 320" fill="none" stroke="url(#routeFade)" strokeWidth="1" opacity="0.6" />
      <path d="M-40 620 C 320 610, 560 590, 820 560 C 1060 532, 1240 600, 1480 600" fill="none" stroke="url(#routeFade)" strokeWidth="1" opacity="0.5" />
    </svg>
  )
}

export default function Hero() {
  const { user, loading } = useAuth()
  return (
    <section className="relative px-6 pb-24 pt-14">
      <RouteLines />

      {/* Soft blue light behind the hero */}
      <div
        aria-hidden="true"
        className="triage-pulse pointer-events-none absolute -top-[120px] left-1/2 h-[700px] w-[900px] -translate-x-[30%] rounded-full"
        style={{
          background:
            "radial-gradient(closest-side, rgba(37,99,235,0.35), rgba(37,99,235,0.08) 55%, rgba(7,7,11,0) 100%)",
        }}
      />

      <div className="relative mx-auto flex max-w-6xl flex-wrap items-center gap-14">
        {/* Text */}
        <div className="min-w-0 flex-[1_1_480px]">
          <span className="inline-flex items-center gap-2 rounded-full border border-[#60A5FA]/35 bg-[#2563EB]/10 px-3.5 py-1.5 text-[13px] font-medium text-[#93C5FD]">
            <span className="h-1.5 w-1.5 rounded-full bg-[#60A5FA]" />
            OpenAI-compatible LLM router
          </span>

          <h1 className="mt-6 text-5xl font-bold leading-[1.05] tracking-[-0.03em] sm:text-6xl lg:text-[64px]">
            Every prompt, routed to the <span className="text-[#60A5FA]">right model.</span>
          </h1>

          <p className="mt-6 max-w-[540px] text-lg leading-relaxed text-[#A7ACBA]">
            Triage predicts how hard each prompt is and sends it to the cheapest model that can answer it well. Lower
            costs, near-identical quality.
          </p>

          {/* Logged in: back to work. While a saved session is checked, the space is kept but nothing shows, so
              the logged-out buttons never flash for a returning user */}
          <div className={cn("mt-9 flex flex-wrap gap-3", loading && "invisible")}>
            <Link
              to="/chat"
              className="group inline-flex min-h-[52px] items-center gap-2.5 rounded-[14px] bg-[#2563EB] px-[26px] text-base font-semibold text-white shadow-[0_10px_40px_rgba(37,99,235,0.4)] transition-colors hover:bg-[#1D4ED8]"
            >
              {user ? "Open chat" : "Try the demo"}
              <ArrowRight className="h-[18px] w-[18px] transition-transform group-hover:translate-x-1" />
            </Link>
            <Link
              to={user ? "/settings/get-started" : "/signup"}
              className="inline-flex min-h-[52px] items-center rounded-[14px] border border-[#2A2D3D] bg-[#12131A]/60 px-[26px] text-base font-medium transition-colors hover:border-[#3B4058] hover:bg-[#12131A]"
            >
              {user ? "Connect your app" : "Create an account"}
            </Link>
          </div>

          {!user && (
            <p className={cn("mt-[18px] text-[13px] text-[#8B90A0]", loading && "invisible")}>
              No signup for the demo · 10 free requests a day
            </p>
          )}
        </div>

        {/* Routing card: built in the next step */}
        <div className="flex min-w-0 flex-[1_1_440px] justify-center">
          <div className="flex h-[460px] w-full max-w-[500px] items-center justify-center rounded-[28px] border border-dashed border-[#1F2130] text-sm text-[#8B90A0]">
            <RoutingCard />
          </div>
        </div>
      </div>
    </section>
  )
}