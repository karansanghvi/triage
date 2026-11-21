import type { ReactNode } from "react"
import { cn } from "@/lib/utils"

type StatCardProps = {
  label: string
  value: ReactNode
  note?: ReactNode
  tone?: "default" | "good"
}

export default function StatCard({ label, value, note, tone = "default" }: StatCardProps) {
  return (
    <div className="min-w-0 rounded-[18px] border border-[#1F2130] bg-[#12131A] p-5">
      <p className="text-[13px] text-muted-foreground">{label}</p>
      <p className={cn("mt-2.5 truncate text-3xl font-bold tracking-[-0.02em]", tone === "good" && "text-[#4ADE80]")}>
        {value}
      </p>
      {note && <p className="mt-1.5 text-xs text-muted-foreground">{note}</p>}
    </div>
  )
}