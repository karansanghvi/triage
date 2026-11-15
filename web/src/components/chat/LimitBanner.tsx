import type { DemoStatus } from "@/types/chat";
import { Clock, KeyRound } from "lucide-react";
import { Link } from "react-router-dom";

function formatResetTime(resetsAt: string): string | null {
    if (!resetsAt) return null
    const date = new Date(resetsAt)
    if (Number.isNaN(date.getTime())) return null
    return date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })
}

export default function LimitBanner({ status }: { status: DemoStatus }) {

    const resetTime = formatResetTime(status.resetsAt)

    const title = status.paused
        ? "The free demo is paused for today"
        : `You've used today's ${status.limit} free requests`
    
    const body = resetTime
        ? `New free requests arrive at ${resetTime}. Add your own API key to keep going now.`
        : "Add your own API key to keep going now."

  return (
    <>
      <div className="mb-3 flex items-center gap-4 rounded-2xl border border-warning/30 bg-warning/10 p-4">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-warning/15">
            <Clock className="h-4 w-4 text-warning" />
        </div>
        <div className="min-w-0 flex-1">
            <p className="text-sm font-medium">{title}</p>
            <p className="mt-0.5 text-xs text-muted-foreground">{body}</p>
        </div>
        <Link
            to="/settings"
            className="flex shrink-0 items-center gap-2 rounded-lg bg-primary px-3 py-2 text-xs font-medium text-primary-foreground hover:bg-primary/90"
        >
            <KeyRound className="h-3.5 w-3.5" />
            Add your keys
        </Link>
      </div>
    </>
  )
}
