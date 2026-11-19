import { Link } from "react-router-dom"
import { AlertCircle, Clock, KeyRound, LogOut, RotateCw, WifiOff } from "lucide-react"
import type { ChatError } from "@/types/chat"
import { cn } from "@/lib/utils"
import { useAuth } from "@/context/AuthContext";

type Action = "keys" | "fix" | "retry" | null

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

// "Try again in 30 seconds"; a long wait (the per-day limit) reads as a time instead
function waitText(seconds?: number): string {
  if (!seconds) return "Try again in a few seconds."
  if (seconds <= 90) return `Try again in ${seconds} ${seconds === 1 ? "second" : "seconds"}.`
  return `Try again at ${timeAfter(seconds)}.`
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
    // Account mode: one saved key was rejected, and the account still has another that works
    case "saved_key_rejected":
      return {
        title: error.provider ? `${error.provider} rejected your saved key` : "One of your saved keys was rejected",
        body: "Your other keys still work. Update this one in Provider keys.",
        action: "fix",
        tone: "error",
        icon: KeyRound,
      }
    // Account mode: that was the last usable key, so the next try goes to the free demo
    case "keys_rejected_demo":
      return {
        title: "Your key was rejected",
        body: "You're now on the free demo.",
        action: "retry",
        tone: "warning",
        icon: KeyRound,
      }
    case "keys_removed_demo":
      return {
        title: "Your saved keys were removed",
        body: "You're now on the free demo.",
        action: "retry",
        tone: "warning",
        icon: KeyRound,
      }
    case "keys_changed":
      return {
        title: "Your saved keys changed",
        body: "They changed while this message was sending. Try again to use them.",
        action: "retry",
        tone: "warning",
        icon: KeyRound,
      }
    case "account_rate_limited":
      return {
        title: "You're sending requests quickly",
        body: waitText(error.retryAfter),
        action: "retry",
        tone: "warning",
        icon: Clock,
      }
    case "invalid_session":
      return {
        title: "You've been logged out",
        body: "Log in again to use your keys. Try again sends this message on the free demo.",
        action: "retry",
        tone: "warning",
        icon: LogOut,
      }
    // Too large for the backend or a proxy (account mode sends the whole conversation): show what it said
    case "http_413":
    case "request_too_large":
    case "context_length_exceeded":
      return {
        title: "This conversation is too long",
        body: error.message,
        action: null,
        tone: "warning",
        icon: AlertCircle,
      }
    // Every model tried ran out of time. The backend names the provider: "Gemini took too long to answer. Try
    // again, or add another provider key so Triage can fall back." (without the key advice on the free demo)
    case "provider_timeout":
      return {
        title: "No answer in time",
        body: error.message || "The model took too long to answer. Try again.",
        action: "retry",
        tone: "warning",
        icon: Clock,
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
  const { user } = useAuth()

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
              to={user ? "/settings/provider-keys" : "/signup"}
              className="mt-3 inline-flex items-center gap-2 rounded-lg bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground hover:bg-primary/90"
            >
              <KeyRound className="h-3.5 w-3.5" />
              {error.code === "invalid_provider_key" ? "Check your keys" : "Add your keys"}
            </Link>
          )}

          {action === "fix" && (
            <Link
              to="/settings/provider-keys"
              className="mt-3 inline-flex items-center gap-2 rounded-lg bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground hover:bg-primary/90"
            >
              <KeyRound className="h-3.5 w-3.5" />
              Fix your keys
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