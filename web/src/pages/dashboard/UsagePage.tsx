import { useRange } from "@/hooks/useRange"

export default function UsagePage() {
  const [range] = useRange()
  return <p className="text-sm text-muted-foreground">My usage for {range}, coming in step 6.</p>
}