import { Link } from "react-router-dom"
import { useAuth } from "@/context/AuthContext"
import { cn } from "@/lib/utils"

export default function FinalCTA() {
  const { user, loading } = useAuth()

  return (
    <section className="relative overflow-hidden px-6 pb-[120px] pt-10">
      <div
        aria-hidden="true"
        className="pointer-events-none absolute left-1/2 top-0 h-[500px] w-[900px] -translate-x-1/2 rounded-full"
        style={{ background: "radial-gradient(closest-side, rgba(37,99,235,0.28), rgba(7,7,11,0))" }}
      />

      <div className="relative mx-auto max-w-[760px] text-center">
        <h2 className="text-4xl font-bold leading-[1.1] tracking-[-0.03em] sm:text-[52px]">Start routing smarter.</h2>
        <p className="mx-auto mt-[18px] max-w-[520px] text-[17px] leading-relaxed text-[#A7ACBA]">
          {user
            ? "Pick up where you left off, or connect another app."
            : "Try it in seconds with no signup, or create an account and connect your app today."}
        </p>
        <div className={cn("mt-9 flex flex-wrap justify-center gap-3", loading && "invisible")}>
          <Link
            to="/chat"
            className="inline-flex min-h-[52px] items-center rounded-[14px] bg-[#2563EB] px-[26px] text-base font-semibold text-white shadow-[0_10px_40px_rgba(37,99,235,0.4)] transition-colors hover:bg-[#1D4ED8]"
          >
            {user ? "Open chat" : "Try the demo"}
          </Link>
          <Link
            to={user ? "/settings/get-started" : "/signup"}
            className="inline-flex min-h-[52px] items-center rounded-[14px] border border-[#2A2D3D] px-[26px] text-base font-medium transition-colors hover:border-[#3B4058] hover:bg-[#12131A]"
          >
            {user ? "Connect your app" : "Create an account"}
          </Link>
        </div>
        <p className="mt-6 text-sm text-[#8B90A0]">
          Building with it?{" "}
          <Link
            to="/docs"
            className="rounded-[4px] font-medium text-[#60A5FA] underline-offset-2 outline-none transition-colors hover:text-[#93C5FD] hover:underline focus-visible:ring-2 focus-visible:ring-[#60A5FA] focus-visible:ring-offset-2 focus-visible:ring-offset-[#07070B] pointer-coarse:inline-flex pointer-coarse:min-h-11 pointer-coarse:items-center"
          >
            Read the docs <span aria-hidden="true">→</span>
          </Link>
        </p>
      </div>
    </section>
  )
}