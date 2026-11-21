// Every error code the API can log (request log error_kind), in words. Anything else is "Other".
const ERROR_LABELS: Record<string, string> = {
  provider_timeout: "Too slow (timeout)",
  timeout: "Too slow (timeout)",
  provider_overloaded: "Provider busy",
  overloaded: "Provider busy",
  provider_unavailable: "Busy or too slow",
  all_models_failed: "No model could answer",
  rate_limit: "Rate limited",
  account_rate_limited: "Rate limited",
  invalid_provider_key: "Key rejected",
}

export function errorLabel(kind: string | null | undefined): string {
  return (kind && ERROR_LABELS[kind]) || "Other"
}

// errors_by_kind as rows with friendly labels, codes with the same label summed, largest first
export function errorRows(byKind: Record<string, number>): { label: string; count: number }[] {
  const totals = new Map<string, number>()
  for (const [kind, count] of Object.entries(byKind)) {
    const label = errorLabel(kind)
    totals.set(label, (totals.get(label) ?? 0) + count)
  }
  return [...totals.entries()]
    .map(([label, count]) => ({ label, count }))
    .filter((r) => r.count > 0)
    .sort((a, b) => b.count - a.count)
}
