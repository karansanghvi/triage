import { cn } from "@/lib/utils"

const TIER_STYLE = {
  low: { label: "Cheapest", cls: "bg-[#22C55E]/12 text-[#4ADE80]", dot: "bg-[#4ADE80]" },
  medium: { label: "Mid-range", cls: "bg-[#3B82F6]/14 text-[#60A5FA]", dot: "bg-[#60A5FA]" },
  high: { label: "Most powerful", cls: "bg-[#F59E0B]/12 text-[#FBBF24]", dot: "bg-[#FBBF24]" },
} as const

function Pill({ label, cls, dot }: { label: string; cls: string; dot: string }) {
  return (
    <span className={cn("inline-flex items-center gap-1.5 whitespace-nowrap rounded-full px-2.5 py-0.5 text-xs font-medium", cls)}>
      <span aria-hidden="true" className={cn("h-1.5 w-1.5 rounded-full", dot)} />
      {label}
    </span>
  )
}

export function TierPill({ tier }: { tier: string | null }) {
  const style = tier && tier in TIER_STYLE ? TIER_STYLE[tier as keyof typeof TIER_STYLE] : null
  if (!style) return <span className="text-muted-foreground">—</span>
  return <Pill {...style} />
}

export function StatusPill({ status, fellBack }: { status: "ok" | "error" | "stopped"; fellBack: boolean | null }) {
  if (status === "error") return <Pill label="Failed" cls="bg-[#EF4444]/12 text-[#F87171]" dot="bg-[#F87171]" />
  if (status === "stopped") return <Pill label="Stopped" cls="bg-[#8B90A0]/14 text-[#C9CDD8]" dot="bg-[#8B90A0]" />
  if (fellBack) return <Pill label="Fell back" cls="bg-[#F59E0B]/12 text-[#FBBF24]" dot="bg-[#FBBF24]" />
  return <Pill label="OK" cls="bg-[#22C55E]/12 text-[#4ADE80]" dot="bg-[#4ADE80]" />
}