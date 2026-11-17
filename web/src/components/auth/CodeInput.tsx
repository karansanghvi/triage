import { useRef, type ClipboardEvent, type KeyboardEvent } from "react"
import { cn } from "@/lib/utils"

type CodeInputProps = {
  value: string
  onChange: (value: string) => void
  invalid?: boolean
  disabled?: boolean
}

const LENGTH = 6

export default function CodeInput({ value, onChange, invalid = false, disabled = false }: CodeInputProps) {
  const refs = useRef<(HTMLInputElement | null)[]>([])
  const digits = Array.from({ length: LENGTH }, (_, i) => value[i] ?? "")

  function focus(index: number) {
    refs.current[Math.max(0, Math.min(LENGTH - 1, index))]?.focus()
  }

  function setDigit(index: number, digit: string) {
    const next = digits.slice()
    next[index] = digit
    onChange(next.join("").slice(0, LENGTH))
  }

  function handleChange(index: number, raw: string) {
    const onlyDigits = raw.replace(/\D/g, "")
    if (!onlyDigits) return
    if (onlyDigits.length > 1) {
      // Typed or autofilled several digits at once: spread them out from here
      const next = digits.slice()
      for (let i = 0; i < onlyDigits.length && index + i < LENGTH; i++) next[index + i] = onlyDigits[i]
      onChange(next.join(""))
      focus(index + onlyDigits.length)
      return
    }
    setDigit(index, onlyDigits)
    focus(index + 1)
  }

  function handleKeyDown(index: number, e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Backspace") {
      e.preventDefault()
      if (digits[index]) setDigit(index, "")
      else if (index > 0) {
        setDigit(index - 1, "")
        focus(index - 1)
      }
    } else if (e.key === "ArrowLeft") {
      focus(index - 1)
    } else if (e.key === "ArrowRight") {
      focus(index + 1)
    }
  }

  function handlePaste(e: ClipboardEvent<HTMLInputElement>) {
    const pasted = e.clipboardData.getData("text").replace(/\D/g, "").slice(0, LENGTH)
    if (!pasted) return
    e.preventDefault()
    onChange(pasted)
    focus(pasted.length)
  }

  return (
    <div className="grid grid-cols-6 gap-2">
      {digits.map((digit, i) => (
        <input
          key={i}
          ref={(el) => {
            refs.current[i] = el
          }}
          type="text"
          inputMode="numeric"
          autoComplete={i === 0 ? "one-time-code" : "off"}
          maxLength={LENGTH}
          aria-label={`Digit ${i + 1} of ${LENGTH}`}
          value={digit}
          disabled={disabled}
          onChange={(e) => handleChange(i, e.target.value)}
          onKeyDown={(e) => handleKeyDown(i, e)}
          onPaste={handlePaste}
          onFocus={(e) => e.target.select()}
          className={cn(
            "h-[58px] w-full rounded-xl border bg-[#0A0A0F] text-center text-2xl font-semibold outline-none transition-colors",
            "focus:border-[#3B82F6] focus:ring-4 focus:ring-[#3B82F6]/15 disabled:opacity-50",
            invalid ? "border-[#F87171]/70" : digit ? "border-[#2A2D3D]" : "border-[#1F2130]"
          )}
        />
      ))}
    </div>
  )
}