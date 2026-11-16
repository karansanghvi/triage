import type { ReactNode } from "react"
import { Link } from "react-router-dom"
import { Split } from "lucide-react"

type AuthLayoutProps = {
  title: string
  subtitle: string
  children: ReactNode
  footer?: ReactNode
}

export default function AuthLayout({ title, subtitle, children, footer }: AuthLayoutProps) {
  return (
    <div className="relative flex min-h-screen flex-col items-center justify-center overflow-hidden bg-[#07070B] px-5 py-12 text-foreground">
      {/* Route lines */}
      <svg aria-hidden="true" viewBox="0 0 1440 900" preserveAspectRatio="none" className="pointer-events-none absolute inset-0 h-full w-full">
        <defs>
          <linearGradient id="authFade" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0" stopColor="#3B82F6" stopOpacity="0" />
            <stop offset="0.3" stopColor="#3B82F6" stopOpacity="0.26" />
            <stop offset="0.7" stopColor="#60A5FA" stopOpacity="0.22" />
            <stop offset="1" stopColor="#60A5FA" stopOpacity="0" />
          </linearGradient>
        </defs>
        <path d="M-40 520 C 260 520, 420 520, 600 520 C 820 520, 940 300, 1480 240" fill="none" stroke="url(#authFade)" strokeWidth="1.4" />
        <path d="M600 520 C 860 520, 1080 530, 1480 520" fill="none" stroke="url(#authFade)" strokeWidth="1.4" />
        <path d="M600 520 C 820 520, 960 730, 1480 800" fill="none" stroke="url(#authFade)" strokeWidth="1.4" />
        <path d="M-40 300 C 300 300, 520 330, 760 380 C 980 420, 1180 340, 1480 330" fill="none" stroke="url(#authFade)" strokeWidth="1" opacity="0.55" />
      </svg>

      {/* Light behind the card */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute left-1/2 top-1/2 h-[760px] w-[900px] -translate-x-1/2 -translate-y-1/2 rounded-full"
        style={{ background: "radial-gradient(closest-side, rgba(37,99,235,0.30), rgba(37,99,235,0.07) 55%, rgba(7,7,11,0))" }}
      />

      <Link to="/" className="relative inline-flex items-center gap-2.5">
        <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-[#2563EB] shadow-[0_0_40px_rgba(37,99,235,0.5)]">
          <Split className="h-[22px] w-[22px] text-white" />
        </span>
        <span className="text-[22px] font-semibold">Triage</span>
      </Link>

      <main className="relative mt-7 w-full max-w-[440px] rounded-3xl border border-[#1F2130] bg-[#0E0F15]/90 px-6 py-10 shadow-[0_30px_120px_rgba(37,99,235,0.18)] sm:px-9">
        <h1 className="text-center text-[28px] font-bold tracking-[-0.02em]">{title}</h1>
        <p className="mt-2 text-center text-[15px] leading-relaxed text-[#A7ACBA]">{subtitle}</p>
        {children}
      </main>

      {footer && <div className="relative mt-6 text-center">{footer}</div>}
    </div>
  )
}