import type { ReactNode } from "react"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { cn } from "@/lib/utils"

type ConfirmDialogProps = {
  open: boolean
  title: string
  description: ReactNode
  confirmLabel: string
  tone?: "danger" | "default"
  busy?: boolean               // while the action runs: buttons disabled, can't close
  confirmDisabled?: boolean    // e.g. until a password is typed
  error?: string | null
  children?: ReactNode         // optional extra fields, like a password input
  onConfirm: () => void
  onCancel: () => void
}

export default function ConfirmDialog({
  open,
  title,
  description,
  confirmLabel,
  tone = "default",
  busy = false,
  confirmDisabled = false,
  error,
  children,
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  return (
    <Dialog
      open={open}
      onOpenChange={(isOpen) => {
        if (!isOpen && !busy) onCancel()
      }}
    >
      <DialogContent showCloseButton={false} className="gap-0 p-0 sm:max-w-md sm:rounded-2xl">
        <div className="p-6">
          <DialogHeader className="text-left">
            <DialogTitle className="text-lg font-semibold">{title}</DialogTitle>
            <DialogDescription className="mt-1.5 text-sm leading-relaxed text-[#A7ACBA]">{description}</DialogDescription>
          </DialogHeader>
          {children && <div className="mt-5">{children}</div>}
          {error && (
            <p
              role="alert"
              className="mt-4 rounded-xl border border-[#F87171]/35 bg-[#EF4444]/8 px-3.5 py-2.5 text-[13px] text-[#FCA5A5]"
            >
              {error}
            </p>
          )}
        </div>
        <div className="flex justify-end gap-2 border-t border-border px-6 py-4">
          <button
            type="button"
            onClick={onCancel}
            disabled={busy}
            className="h-10 rounded-[10px] px-4 text-sm font-medium text-[#A7ACBA] transition-colors hover:bg-accent hover:text-foreground disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={onConfirm}
            disabled={busy || confirmDisabled}
            className={cn(
              "h-10 rounded-[10px] px-4 text-sm font-semibold text-white transition-colors disabled:cursor-not-allowed disabled:opacity-50",
              tone === "danger" ? "bg-[#DC2626] hover:bg-[#B91C1C]" : "bg-[#2563EB] hover:bg-[#1D4ED8]"
            )}
          >
            {busy ? "Working…" : confirmLabel}
          </button>
        </div>
      </DialogContent>
    </Dialog>
  )
}