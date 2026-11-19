import type { ReactNode } from "react"
import { cn } from "@/lib/utils"

type SettingsCardProps = {
  title: string
  description?: ReactNode
  action?: ReactNode           // e.g. a "Create key" button in the header
  children?: ReactNode
  tone?: "default" | "danger"
  className?: string
}

export default function SettingsCard({ title, description, action, children, tone = "default", className }: SettingsCardProps) {
  return (
    <section
      className={cn(
        "overflow-hidden rounded-[18px] border bg-[#12131A]",
        tone === "danger" ? "border-[#F87171]/25" : "border-[#1F2130]",
        className
      )}
    >
      <div className="flex flex-wrap items-end justify-between gap-3 border-b border-[#1F2130] px-6 pb-[18px] pt-[22px]">
        <div className="min-w-0 flex-[1_1_320px]">
          <h2 className={cn("text-base font-semibold", tone === "danger" && "text-[#FCA5A5]")}>{title}</h2>
          {description && <p className="mt-1 text-[13px] leading-relaxed text-[#8B90A0]">{description}</p>}
        </div>
        {action}
      </div>
      {children}
    </section>
  )
}