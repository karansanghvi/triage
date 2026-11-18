import type { Provider } from "@/lib/account"

// Letter tiles only. A provider's logo may be added here only after that company grants written
// permission: their trademark policies don't allow showing (or recolouring) their logos in our UI.
const LETTERS: Record<Provider, string> = { openai: "O", anthropic: "A", groq: "G", gemini: "G" }

type ProviderLogoProps = {
  provider: Provider
  /** Tile width and height in pixels. */
  size?: number
}

/** The provider's tile. Decorative: always shown next to the provider's name. */
export default function ProviderLogo({ provider, size = 40 }: ProviderLogoProps) {
  return (
    <span
      aria-hidden="true"
      className="flex shrink-0 items-center justify-center rounded-xl border border-[#1F2130] bg-[#1A1C26] font-semibold"
      style={{ width: size, height: size, fontSize: Math.round(size * 0.375) }}
    >
      {LETTERS[provider]}
    </span>
  )
}
