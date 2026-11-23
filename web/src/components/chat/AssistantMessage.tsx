import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { ChevronRight, Split, Scissors, Square, Clock } from "lucide-react"
import type { AssistantMessage as AssistantMessageType, Tier } from "@/types/chat"
import { formatPercent } from "@/lib/format"
import Usd from "@/components/common/Usd"
import { cn } from "@/lib/utils"
import { remarkHtmlBreaks } from "@/lib/remarkHtmlBreaks"
import ErrorCard from "./ErrorCard"
import { useServerWake } from "@/lib/server"
import { WAKE_ESTIMATE_SECONDS } from "@/lib/wake"

type AssistantMessageProps = {
  message: AssistantMessageType
  live?: string            // the text that has arrived so far, while this answer streams in
  selected: boolean
  onSelect: () => void
  onRetry?: () => void
}

const TIER_DOT: Record<Tier, string> = {
  low: "bg-success",
  medium: "bg-primary",
  high: "bg-warning",
}

const TIER_SHORT: Record<Tier, string> = {
  low: "Cheapest",
  medium: "Mid-range",
  high: "Most powerful",
}

// Markdown that is still arriving: an unclosed code fence is closed for display only, so the rest of the answer
// doesn't flash as code until the closing fence arrives
function displayable(text: string): string {
  const fences = text.match(/^\s*```/gm)?.length ?? 0
  return fences % 2 === 1 ? `${text}\n\`\`\`` : text
}

function Markdown({ text }: { text: string }) {
  return (
    <div className="prose prose-sm prose-invert max-w-none pt-1">
      <ReactMarkdown remarkPlugins={[remarkGfm, remarkHtmlBreaks]}>{text}</ReactMarkdown>
    </div>
  )
}

// Before the first token: dots, plus a note when the server is waking up or the message is long
function Waiting({ longPrompt }: { longPrompt: boolean }) {
  const { state, waitingSeconds } = useServerWake()
  return (
    <div className="py-3">
      <div className="flex items-center gap-1.5">
        <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground [animation-delay:-0.3s]" />
        <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground [animation-delay:-0.15s]" />
        <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground" />
      </div>
      {state === "waking" && (
        <p role="status" className="mt-2 flex items-start gap-1.5 text-xs text-muted-foreground">
          <Clock className="mt-px h-3.5 w-3.5 shrink-0" />
          <span>
            Waking up the server: it sleeps when nobody has used it for a while, and starting takes about{" "}
            {Math.round(WAKE_ESTIMATE_SECONDS / 60)} minutes ({waitingSeconds} s so far). Your message is sent as
            soon as it's up.
          </span>
        </p>
      )}
      {state !== "waking" && longPrompt && (
        <p role="status" className="mt-2 flex items-start gap-1.5 text-xs text-muted-foreground">
          <Clock className="mt-px h-3.5 w-3.5 shrink-0" />
          <span>Long message: on the free server, reading it to pick a model can take up to a minute.</span>
        </p>
      )}
    </div>
  )
}

export default function AssistantMessage({ message, live, selected, onSelect, onRetry }: AssistantMessageProps) {
  const routing = message.routing
  const savedShare =
    routing && routing.saved !== null && routing.strongCost ? routing.saved / routing.strongCost : null

  return (
    <div className="flex gap-3">
      {/* Triage avatar */}
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary">
        <Split className="h-4 w-4 text-primary-foreground" />
      </div>

      <div className="min-w-0 flex-1">
        {/* Arriving: the text so far */}
        {message.status === "pending" && live && <Markdown text={displayable(live)} />}

        {/* Thinking (nothing has arrived yet), saying why when it may take a while */}
        {message.status === "pending" && !live && <Waiting longPrompt={message.longPrompt === true} />}

        {/* Error: whatever arrived before it, then the card */}
        {message.status === "error" && message.content && <Markdown text={displayable(message.content)} />}
        {message.status === "error" && message.error && <ErrorCard error={message.error} onRetry={onRetry} />}

        {/* Answer */}
        {message.status === "done" && (
          <>
            {message.content && <Markdown text={message.stopped ? displayable(message.content) : message.content} />}

            {message.stopped && (
              <p className="mt-2 flex items-center gap-1.5 text-xs text-muted-foreground">
                <Square className="h-3 w-3" />
                {message.content ? "Stopped" : "Stopped before the answer started"}
              </p>
            )}

            {message.truncated && (
              <p className="mt-2 flex items-center gap-1.5 text-xs text-warning">
                <Scissors className="h-3.5 w-3.5" />
                This answer was cut off because it reached the length limit.
              </p>
            )}

            {routing && (
              <button
                onClick={onSelect}
                className={cn(
                  "group -ml-2 mt-2 flex max-w-full flex-wrap items-center gap-x-2 gap-y-0.5 rounded-md px-2 py-1 text-left text-xs transition-colors [&>span]:whitespace-nowrap max-md:min-h-11 pointer-coarse:min-h-11",
                  selected ? "text-foreground" : "text-muted-foreground hover:bg-accent hover:text-foreground"
                )}
              >
                <span className={cn("h-1.5 w-1.5 rounded-full", TIER_DOT[routing.tier])} />
                <span>{TIER_SHORT[routing.tier]}</span>
                <span className="text-muted-foreground/40">·</span>
                <span>{routing.model}</span>
                <span className="text-muted-foreground/40">·</span>
                {routing.cost === null ? (
                  <span>Stopped</span>
                ) : routing.costEstimated ? (
                  <span>
                    ≈ <Usd value={routing.cost} /> (stopped)
                  </span>
                ) : (
                  <Usd value={routing.cost} />
                )}
                {!message.stopped && savedShare !== null && savedShare > 0 && (
                  <>
                    <span className="text-muted-foreground/40">·</span>
                    <span className="font-medium text-success">saved {formatPercent(savedShare)}</span>
                  </>
                )}
                <ChevronRight
                  className={cn(
                    "h-3.5 w-3.5 transition-all",
                    selected ? "text-primary opacity-100" : "opacity-0 group-hover:opacity-100"
                  )}
                />
              </button>
            )}
          </>
        )}
      </div>
    </div>
  )
}