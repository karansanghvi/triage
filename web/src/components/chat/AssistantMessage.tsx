import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { ChevronRight, Split, Scissors } from "lucide-react"
import type { AssistantMessage as AssistantMessageType, Tier } from "@/types/chat"
import { formatPercent } from "@/lib/format"
import Usd from "@/components/common/Usd"
import { cn } from "@/lib/utils"
import ErrorCard from "./ErrorCard"

type AssistantMessageProps = {
  message: AssistantMessageType
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

export default function AssistantMessage({ message, selected, onSelect, onRetry }: AssistantMessageProps) {
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
        {/* Thinking */}
        {message.status === "pending" && (
          <div className="flex items-center gap-1.5 py-3">
            <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground [animation-delay:-0.3s]" />
            <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground [animation-delay:-0.15s]" />
            <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground" />
          </div>
        )}

        {/* Error */}
        {message.status === "error" && message.error && <ErrorCard error={message.error} onRetry={onRetry} />}

        {/* Answer */}
        {message.status === "done" && (
          <>
            <div className="prose prose-sm prose-invert max-w-none pt-1">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
            </div>

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
                <Usd value={routing.cost} />
                {savedShare !== null && savedShare > 0 && (
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