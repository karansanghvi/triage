import { Link } from "react-router-dom"
import { AlertCircle, Clock, KeyRound, RotateCw, WifiOff } from "lucide-react"
import type { ChatError } from "@/types/chat"
import { cn } from "@/lib/utils"

type Action = "keys" | "retry" | null

type ErrorDetails = {
  title: string
  body: string
  action: Action
  tone: "warning" | "error"
  icon: typeof AlertCircle
}

function timeAfter(seconds?: number): string | null {
  if (!seconds) return null
  return new Date(Date.now() + seconds * 1000).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })
}

function describe(error: ChatError): ErrorDetails {
  switch (error.code) {
    case "demo_limit_reached": {
      const time = timeAfter(error.retryAfter)
      return {
        title: "You've used today's free requests",
        body: time
          ? `New free requests are available at ${time}. Add your own API key to keep going now.`
          : "Add your own API key to keep going now.",
        action: "keys",
        tone: "warning",
        icon: Clock,
      }
    }
    case "demo_paused":
      return {
        title: "The free demo is paused for today",
        body: "So many people tried Triage today that the free demo reached its daily budget. Add your own API key to keep going.",
        action: "keys",
        tone: "warning",
        icon: Clock,
      }
    case "demo_busy":
      return {
        title: "Lots of people are trying Triage",
        body: `The free demo is busy right now. Try again in ${error.retryAfter ?? "a few"} seconds.`,
        action: "retry",
        tone: "warning",
        icon: Clock,
      }
    case "demo_prompt_too_long":
      return {
        title: "This message is too long for the free demo",
        body: "Shorten it or start a new chat. With your own API key, longer messages work too.",
        action: "keys",
        tone: "warning",
        icon: AlertCircle,
      }
    case "invalid_provider_key":
      return {
        title: "Your API key was rejected",
        body: error.message,
        action: "keys",
        tone: "error",
        icon: KeyRound,
      }
    case "network_error":
      return {
        title: "Couldn't reach Triage",
        body: "Check your internet connection and try again.",
        action: "retry",
        tone: "error",
        icon: WifiOff,
      }
    case "interrupted":
      return {
        title: "This answer was interrupted",
        body: error.message,
        action: "retry",
        tone: "warning",
        icon: AlertCircle,
      }
    default:
      return {
        title: "Something went wrong",
        body: error.message,
        action: "retry",
        tone: "error",
        icon: AlertCircle,
      }
  }
}

type ErrorCardProps = {
  error: ChatError
  onRetry?: () => void
}

export default function ErrorCard({ error, onRetry }: ErrorCardProps) {
  const { title, body, action, tone, icon: Icon } = describe(error)

  return (
    <div
      className={cn(
        "rounded-xl border p-4",
        tone === "warning" ? "border-warning/30 bg-warning/10" : "border-destructive/30 bg-destructive/10"
      )}
    >
      <div className="flex items-start gap-3">
        <Icon className={cn("mt-0.5 h-4 w-4 shrink-0", tone === "warning" ? "text-warning" : "text-destructive")} />
        <div className="min-w-0">
          <p className="text-sm font-medium text-foreground">{title}</p>
          <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{body}</p>

          {action === "keys" && (
            <Link
              to="/settings"
              className="mt-3 inline-flex items-center gap-2 rounded-lg bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground hover:bg-primary/90"
            >
              <KeyRound className="h-3.5 w-3.5" />
              {error.code === "invalid_provider_key" ? "Check your keys" : "Add your keys"}
            </Link>
          )}

          {action === "retry" && onRetry && (
            <button
              onClick={onRetry}
              className="mt-3 inline-flex items-center gap-2 rounded-lg border border-border px-3 py-1.5 text-xs font-medium text-foreground hover:bg-accent"
            >
              <RotateCw className="h-3.5 w-3.5" />
              Try again
            </button>
          )}
        </div>
      </div>
    </div>
  )
}