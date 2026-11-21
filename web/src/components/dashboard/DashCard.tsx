import { cn } from "@/lib/utils";
import type { ReactNode } from "react";

type DashCardProps = {
  title: string
  description?: ReactNode
  action?: ReactNode
  children: ReactNode
  className?: string
}

export default function DashCard({ title, description, action, children, className }: DashCardProps) {
  return (
    <>
      <section className={cn("min-w-0 rounded-[18px] border border-[#1F2130] bg-[#12131A] px-6 py-[22px]", className)}>
        <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="min-w-0">
                <h2 className="text-base font-semibold">{title}</h2>
                {description && <p className="mt-1 text-[13px] leading-relaxed text-muted-foreground">{description}</p>}
            </div>
            {action}
        </div>
        {children}
      </section>
    </>
  )
}
