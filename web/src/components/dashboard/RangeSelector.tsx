import { useRange } from "@/hooks/useRange";
import { RANGES } from "@/lib/dashboard";
import { cn } from "@/lib/utils";

export default function RangeSelector() {

    const [range, setRange] = useRange()

  return (
    <>
      <div role="group" aria-label="Time range" className="flex gap-1 rounded-xl border border-[#1A1C26] bg-[#0E0F15] p-1">
        {RANGES.map(({ value, label }) => (
            <button
                key={value}
                type="button"
                aria-pressed={range === value}
                onClick={() => setRange(value)}
                className={cn(
                    "h-9 rounded-[9px] px-3.5 text-[13px] font-medium transition-colors max-md:min-h-11 pointer-coarse:min-h-11",
                    range === value ? "bg-[#1A1C26] text-foreground" : "text-muted-foreground hover:text-foreground"
                )}
            >
                {label}
            </button>
        ))}
      </div>
    </>
  )
}
