import { cn } from "@/lib/utils"

export type BadgeKind = "valid" | "unverified" | "invalid" | "missing" | "revoked"

const STYLES: Record<BadgeKind, { label: string; text: string; bg: string; dot: string }> = {
  valid: { label: "Verified", text: "text-[#4ADE80]", bg: "bg-[#22C55E]/12", dot: "bg-[#4ADE80]" },
  unverified: { label: "Unverified", text: "text-[#FBBF24]", bg: "bg-[#F59E0B]/12", dot: "bg-[#FBBF24]" },
  invalid: { label: "Rejected", text: "text-[#F87171]", bg: "bg-[#EF4444]/12", dot: "bg-[#F87171]" },
  missing: { label: "Not added", text: "text-[#8B90A0]", bg: "bg-[#1A1C26]", dot: "bg-[#8B90A0]" },
  revoked: { label: "Revoked", text: "text-[#8B90A0]", bg: "bg-[#1A1C26]", dot: "bg-[#8B90A0]" },
}

export default function StatusBadge({ kind }: { kind: BadgeKind }) {
  const s = STYLES[kind]
  return (
    <span className={cn("inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium", s.text, s.bg)}>
      <span className={cn("h-1.5 w-1.5 rounded-full", s.dot)} />
      {s.label}
    </span>
  )
}