import type { Range } from "@/lib/dashboard";
import { useSearchParams } from "react-router-dom";

const VALID: Range[] = ["7d", "30d", "all"]

// The selected time range lives in the address (?range=7d), so refresh and links keep it 
export function useRange(): [Range, (range: Range) => void] {
  const [params, setParams] = useSearchParams()
  const raw = params.get("range")
  const range: Range = VALID.includes(raw as Range) ? (raw as Range) : "30d"

  function setRange(next: Range) {
    setParams(
        (prev) => {
            const updated = new URLSearchParams(prev)
            updated.set("range", next)
            return updated
        },
        { replace: true }
    )
  }

  return [range, setRange]
}
