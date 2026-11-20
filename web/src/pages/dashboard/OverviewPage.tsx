import { useRange } from "@/hooks/useRange"

export default function OverviewPage() {
  const [range] = useRange()
  return <p className="text-sm text-muted-foreground">Overview for {range}, coming in step 5.</p>
}