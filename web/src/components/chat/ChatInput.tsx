import { useState } from "react"
import { ArrowUp, SlidersHorizontal, Square } from "lucide-react"
import { cn } from "@/lib/utils"
import { QUALITY_MAX, QUALITY_MIN, QUALITY_PRESETS, qualityLabel } from "@/lib/quality"

type ChatInputProps = {
  value: string
  onChange: (value: string) => void
  onSend: () => void
  quality: number
  onQualityChange: (value: number) => void
  disabled?: boolean
  streaming?: boolean     // an answer is arriving: the send button becomes Stop
  onStop?: () => void
}

// Where a value sits along the slider, as a percentage of its width
function position(value: number): string {
  return `${((value - QUALITY_MIN) / (QUALITY_MAX - QUALITY_MIN)) * 100}%`
}

export default function ChatInput({ value, onChange, onSend, quality, onQualityChange, disabled = false, streaming = false, onStop }: ChatInputProps) {
  const [showDial, setShowDial] = useState(false)
  const canSend = value.trim().length > 0 && !disabled

  return (
    <div className="mx-auto w-full max-w-3xl">
      {/* Quality dial panel */}
      {showDial && (
        <div className="mb-3 rounded-2xl border border-border bg-card p-5">
          <div className="flex items-start justify-between gap-4">
            <div>
              <p className="text-sm font-medium">Quality dial</p>
              <p className="mt-0.5 text-xs text-muted-foreground">
                How sure Triage must be that a cheaper model can answer well.
              </p>
            </div>
            <span className="shrink-0 rounded-full bg-primary/15 px-2.5 py-1 text-xs font-medium text-primary">
              {qualityLabel(quality)} · {quality}%
            </span>
          </div>

          {/* Slider with preset tick marks */}
          <div className="relative mt-5">
            <input
              type="range"
              min={QUALITY_MIN}
              max={QUALITY_MAX}
              step={1}
              value={quality}
              onChange={(e) => onQualityChange(Number(e.target.value))}
              className="relative z-10 w-full accent-primary"
            />
            {QUALITY_PRESETS.map((preset) => (
              <span
                key={preset.value}
                className="pointer-events-none absolute top-1/2 h-3 w-0.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-muted-foreground/50"
                style={{ left: position(preset.value) }}
              />
            ))}
          </div>

          <div className="mt-1 flex justify-between text-xs text-muted-foreground">
            <span>Save money</span>
            <span>Best answers</span>
          </div>

          {/* Preset buttons */}
          <div className="mt-4 flex gap-2">
            {QUALITY_PRESETS.map((preset) => (
              <button
                key={preset.value}
                onClick={() => onQualityChange(preset.value)}
                className={cn(
                  "flex-1 rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors",
                  quality === preset.value
                    ? "border-primary/50 bg-primary/15 text-primary"
                    : "border-border text-muted-foreground hover:bg-accent hover:text-foreground"
                )}
              >
                {preset.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Input box */}
      <div className="rounded-2xl border border-border bg-card p-3 transition-colors focus-within:border-primary/60">
        <textarea
          value={value}
          disabled={disabled}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault()
              if (canSend) onSend()
            }
          }}
          rows={2}
          placeholder="Ask anything..."
          className="w-full resize-none bg-transparent px-2 py-1 text-sm outline-none placeholder:text-muted-foreground"
        />

        <div className="mt-2 flex items-center justify-between">
          <button
            onClick={() => setShowDial(!showDial)}
            className={cn(
              "flex items-center gap-2 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors max-md:min-h-11 pointer-coarse:min-h-11",
              showDial ? "bg-primary/15 text-primary" : "text-muted-foreground hover:bg-accent hover:text-foreground"
            )}
          >
            <SlidersHorizontal className="h-3.5 w-3.5" />
            Quality: {qualityLabel(quality)}
          </button>

          {streaming && onStop ? (
            <button
              type="button"
              onClick={onStop}
              aria-label="Stop answer"
              title="Stop"
              className="flex h-9 w-9 items-center justify-center rounded-xl bg-foreground text-background transition-all hover:bg-foreground/85 max-md:h-11 pointer-coarse:h-11 max-md:w-11 pointer-coarse:w-11"
            >
              <Square className="h-3.5 w-3.5 fill-current" />
            </button>
          ) : (
          <button
            onClick={onSend}
            disabled={!canSend}
            aria-label="Send"
            className={cn(
              "flex h-9 w-9 items-center justify-center rounded-xl transition-all max-md:h-11 pointer-coarse:h-11 max-md:w-11 pointer-coarse:w-11",
              canSend
                ? "bg-primary text-primary-foreground shadow-lg shadow-primary/30 hover:bg-primary/90"
                : "cursor-not-allowed bg-muted text-muted-foreground"
            )}
          >
            <ArrowUp className="h-4 w-4" />
          </button>
          )}
        </div>
      </div>
    </div>
  )
}