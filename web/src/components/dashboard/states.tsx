import type { ReactNode } from "react"
import type { TierMix } from "@/lib/dashboard"
import { cn } from "@/lib/utils"

// Shared by the live tabs (Overview, My usage): buttons, skeleton pieces, the error card and the centered card

export const PRIMARY_BUTTON =
  "inline-flex h-10 items-center justify-center gap-1.5 rounded-[10px] bg-[#2563EB] px-4 max-md:min-h-11 pointer-coarse:min-h-11 text-[13px] font-semibold text-white transition-colors hover:bg-[#1D4ED8] disabled:pointer-events-none disabled:opacity-60"
export const SECONDARY_BUTTON =
  "inline-flex h-10 items-center justify-center gap-1.5 rounded-[10px] border border-[#2A2D3D] max-md:min-h-11 pointer-coarse:min-h-11 bg-[#0E0F15] px-4 text-[13px] font-semibold text-foreground transition-colors hover:bg-[#1A1C26] disabled:pointer-events-none disabled:opacity-60"
export const SMALL_BOX = "rounded-xl border border-[#1F2130] bg-[#0A0A0F] px-3.5 py-3"

// A tier nobody reached is left out of tier_mix; the bar wants all three
export function fullMix(mix: Partial<TierMix>): TierMix {
  return { low: mix.low ?? 0, medium: mix.medium ?? 0, high: mix.high ?? 0 }
}

export function Bone({ className }: { className: string }) {
  return <div aria-hidden="true" className={cn("animate-pulse rounded-md bg-[#1A1C26]", className)} />
}

// A DashCard-sized placeholder: title and description lines, then the given body
export function CardBones({ children }: { children: ReactNode }) {
  return (
    <div className="min-w-0 rounded-[18px] border border-[#1F2130] bg-[#12131A] px-6 py-[22px]">
      <Bone className="h-6 w-40" />
      <Bone className="mt-1 h-5 w-full max-w-[420px]" />
      {children}
    </div>
  )
}

// Four StatCard-sized placeholders
export function StatCardBones() {
  return (
    <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
      {[0, 1, 2, 3].map((i) => (
        <div key={i} className="min-w-0 rounded-[18px] border border-[#1F2130] bg-[#12131A] p-5">
          <Bone className="h-5 w-28" />
          <Bone className="mt-2.5 h-9 w-24" />
          <Bone className="mt-1.5 h-4 w-36" />
        </div>
      ))}
    </div>
  )
}

export function Loading({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-5" aria-busy="true">
      <p className="sr-only" role="status">
        {label}
      </p>
      {children}
    </div>
  )
}

export function ErrorCard({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <section
      role="alert"
      className="flex flex-col items-center rounded-[18px] border border-[#1F2130] bg-[#12131A] px-6 py-12 text-center"
    >
      <p className="text-base font-semibold">{message}</p>
      <p className="mt-1.5 text-[13px] text-muted-foreground">Check your connection, then try again.</p>
      <button type="button" onClick={onRetry} className={cn(PRIMARY_BUTTON, "mt-5")}>
        Try again
      </button>
    </section>
  )
}

// The centered card with a soft glow and an icon: Overview's waiting card, My usage's logged-out and empty states
export function CenterCard({ icon, title, children }: { icon: ReactNode; title: string; children: ReactNode }) {
  return (
    <section className="relative overflow-hidden rounded-[18px] border border-[#1F2130] bg-[#12131A] px-6 py-12 sm:px-8">
      <div
        aria-hidden="true"
        className="pointer-events-none absolute -top-40 left-1/2 h-[360px] w-[560px] max-w-full -translate-x-1/2 rounded-full"
        style={{ background: "radial-gradient(closest-side, rgba(37,99,235,0.16), rgba(18,19,26,0))" }}
      />
      <div className="relative mx-auto flex max-w-[460px] flex-col items-center text-center">
        <span className="flex h-12 w-12 items-center justify-center rounded-[14px] border border-[#1F2130] bg-[#0A0A0F] text-[#60A5FA] [&_svg]:h-6 [&_svg]:w-6">
          {icon}
        </span>
        <h2 className="mt-5 text-xl font-semibold tracking-[-0.01em]">{title}</h2>
        {children}
      </div>
    </section>
  )
}
