// "just now", "5m ago", "3h ago", "2d ago", or a short date like "Nov 15"
export function relativeTime(iso: string): string {
  const seconds = (Date.now() - new Date(iso).getTime()) / 1000
  if (seconds < 60) return "just now"
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`
  if (seconds < 7 * 86400) return `${Math.floor(seconds / 86400)}d ago`
  return new Date(iso).toLocaleDateString([], { month: "short", day: "numeric" })
}

export function shortDate(iso: string): string {
  return new Date(iso).toLocaleDateString([], { month: "short", day: "numeric", year: "numeric" })
}
// A Retry-After wait in whole minutes, for "Try again in ...": "1 minute", "3 minutes"; "a few minutes" if unknown
export function minutesFrom(seconds?: number): string {
  if (!seconds || !Number.isFinite(seconds) || seconds <= 0) return "a few minutes"
  const minutes = Math.ceil(seconds / 60)
  return minutes === 1 ? "1 minute" : `${minutes} minutes`
}
