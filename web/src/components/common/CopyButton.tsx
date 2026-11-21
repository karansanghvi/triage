import { cn } from "@/lib/utils";
import { Check, Copy } from "lucide-react";
import { useState } from "react";

type CopyButtonProps = {
  text: string
  label?: string          // visible text; leave empty for an icon-only button
  ariaLabel?: string
  variant?: "primary" | "subtle" | "icon"
  className?: string
}

export default function CopyButton({ text, label, ariaLabel, variant = "subtle", className }: CopyButtonProps) {

    const [copied, setCopied] = useState(false);

    async function handleCopy() {
        try {
            await navigator.clipboard.writeText(text)
            setCopied(true)
            setTimeout(() => setCopied(false), 1500)
        } catch {
            // Clipboard blocked: nothing to show 
        }
    }

    const Icon = copied ? Check : Copy

  return (
    <>
        <button
        type="button"
        onClick={handleCopy}
        aria-label={ariaLabel ?? (label ? undefined : "Copy")}
        className={cn(
            "inline-flex shrink-0 items-center justify-center gap-1.5 rounded-[10px] text-[13px] font-medium transition-colors",
            variant === "primary" && "h-11 bg-[#2563EB] px-4 text-white hover:bg-[#1D4ED8]",
            variant === "subtle" && "h-8 px-2.5 text-[#8B90A0] hover:text-foreground max-md:min-h-11 pointer-coarse:min-h-11",
            variant === "icon" && "h-11 w-11 border border-[#1F2130] text-[#8B90A0] hover:bg-accent hover:text-foreground",
            className
        )}
        >
            <Icon className={cn("h-[15px] w-[15px]", copied && variant !== "primary" && "text-[#4ADE80]")} />
            {label && (copied ? "Copied" : label)}
        </button>
    </>
  )
}
