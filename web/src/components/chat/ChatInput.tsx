import { cn } from "@/lib/utils";
import { ArrowUp, SlidersHorizontal } from "lucide-react";
import { useState } from "react";

type ChatInputProps = {
    value: string
    onChange: (value: string) => void
    onSend: () => void
    quality: number
    onQualityChange: (value: number) => void 
}

function qualityLabel(quality: number): string {
  if (quality < 34) return "Cheapest"
  if (quality < 67) return "Balanced"
  return "Best quality"
}

export default function ChatInput({ value, onChange, onSend, quality, onQualityChange }: ChatInputProps) {

    const [showDial, setShowDial] = useState(false)
    const canSend = value.trim().length > 0 

  return (
    <>
      <div className="mx-auto w-full max-w-3xl">
        {/* Quality dial panel */}
        {showDial && (
            <div className="mb-3 rounded-2xl border border-border bg-card p-5">
                <div className="flex items-center justify-between">
                    <p className="text-sm font-medium">Quality Dial</p>
                    <span className="rounded-full bg-primary/15 px-2.5 py-1 text-xs font-medium text-primary">
                        {qualityLabel(quality)} · {quality}
                    </span>
                </div>
                <input
                    type="range"
                    min={0}
                    max={100}
                    value={quality}
                    onChange={(e) => onQualityChange(Number(e.target.value))}
                    className="mt-4 w-full accent-primary"
                />
                <div className="mt-2 flex justify-between text-xs text-muted-foreground">
                    <span>Save money</span>
                    <span>Best answers</span>
                </div>
            </div>
        )}

        {/* Input Box */}
        <div className="rounded-2xl border border-border bg-card p-3 transition-colors focus-within:border-primary/60">
            <textarea
                value={value}
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
                        "flex items-center gap-2 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors",
                        showDial ? "bg-primary/15 text-primary" : "text-muted-foreground hover:bg-accent hover:text-foreground"
                    )}
                >
                    <SlidersHorizontal className="h-3.5 w-3.5" />
                    Quality: {qualityLabel(quality)}
                </button>

                <button
                    onClick={onSend}
                    disabled={!canSend}
                    className={cn(
                        "flex h-9 w-9 items-center justify-center rounded-xl transition-all",
                        canSend
                            ? "bg-primary text-primary-foreground shadow-lg shadow-primary/30 hover:bg-primary/90"
                            : "cursor-not-allowed bg-muted text-muted-foreground"
                    )}
                >
                    <ArrowUp className="h-4 w-4" />
                </button>
            </div>
        </div>
      </div>
    </>
  )
}
