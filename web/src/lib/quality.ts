// The useful range of Triage's quality bar (tau), as a 0-100 slider value
export const QUALITY_MIN = 60
export const QUALITY_MAX = 95

// Triage's built-in presets (tau x 100). Hard-coded to match the backend config.
export const QUALITY_PRESETS = [
  { label: "Cheapest", value: 68 },
  { label: "Balanced", value: 86 },
  { label: "Best", value: 89 },
] as const

export const DEFAULT_QUALITY = 86

export function qualityLabel(quality: number): string {
  if (quality <= 77) return "Cheapest"
  if (quality <= 87) return "Balanced"
  return "Best quality"
}

// Slider value -> the X-Triage-Quality header value, e.g. 86 -> "0.86"
export function qualityToTau(quality: number): number {
  return quality / 100
}