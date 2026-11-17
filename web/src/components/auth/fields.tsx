import { useState, type InputHTMLAttributes } from "react"
import { AlertCircle, Eye, EyeOff } from "lucide-react"
import { cn } from "@/lib/utils"

export const inputClass =
  "mt-2 h-12 w-full rounded-xl border bg-[#0A0A0F] px-3.5 text-[15px] outline-none transition-colors placeholder:text-[#7C8192] focus:border-[#3B82F6] focus:ring-4 focus:ring-[#3B82F6]/15"

export function fieldBorder(invalid: boolean) {
  return invalid ? "border-[#F87171]/70" : "border-[#1F2130]"
}

type PasswordInputProps = InputHTMLAttributes<HTMLInputElement> & { invalid?: boolean }

export function PasswordInput({ invalid = false, className, ...props }: PasswordInputProps) {
  const [visible, setVisible] = useState(false)
  return (
    <div className="relative">
      <input
        {...props}
        type={visible ? "text" : "password"}
        className={cn(inputClass, fieldBorder(invalid), "pr-[52px]", className)}
      />
      <button
        type="button"
        onClick={() => setVisible(!visible)}
        aria-label={visible ? "Hide password" : "Show password"}
        className="absolute right-0.5 top-[10px] flex h-11 w-11 items-center justify-center rounded-[10px] text-[#8B90A0] hover:text-foreground"
      >
        {visible ? <EyeOff className="h-[18px] w-[18px]" /> : <Eye className="h-[18px] w-[18px]" />}
      </button>
    </div>
  )
}

export function FormError({ message }: { message: string }) {
  return (
    <div role="alert" className="flex items-start gap-2.5 rounded-xl border border-[#F87171]/35 bg-[#EF4444]/8 px-3.5 py-3 text-[13px] text-[#FCA5A5]">
      <AlertCircle className="mt-px h-4 w-4 shrink-0" />
      {message}
    </div>
  )
}

export const primaryButton =
  "h-[50px] rounded-xl bg-[#2563EB] text-[15px] font-semibold text-white shadow-[0_10px_30px_rgba(37,99,235,0.35)] transition-colors hover:bg-[#1D4ED8] disabled:cursor-not-allowed disabled:opacity-60"

export const secondaryButton =
  "flex h-12 items-center justify-center gap-2 rounded-xl border border-[#2A2D3D] text-sm font-medium transition-colors hover:border-[#3B4058] hover:bg-white/[0.03]"