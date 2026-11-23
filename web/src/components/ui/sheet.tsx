import type { ReactNode, RefObject } from "react"
import { Dialog as DialogPrimitive } from "@base-ui/react/dialog"
import { XIcon } from "lucide-react"
import { cn } from "@/lib/utils"

// A panel that slides in from an edge, over a dimmed backdrop: the app's mobile menu (left), the public pages'
// mobile menu (right) and the chat's routing details (bottom). Base UI's Dialog does the hard parts: focus moves in when it opens and back to the trigger
// when it closes, Esc and a tap on the backdrop close it, and the page behind doesn't scroll.

type SheetProps = {
  open: boolean
  onOpenChange: (open: boolean) => void
  side: "left" | "right" | "bottom"
  title: string // for screen readers (visible titles go in children)
  closeLabel: string
  className?: string
  finalFocus?: RefObject<HTMLElement | null> // where focus goes when it closes (default: what had it before)
  onOpenChangeComplete?: (open: boolean) => void // after the open/close animation (and, closing, the scroll unlock)
  children: ReactNode
}

const SIDE = {
  left: "inset-y-0 left-0 h-dvh w-[min(85vw,288px)] border-r data-open:slide-in-from-left data-closed:slide-out-to-left",
  right: "inset-y-0 right-0 h-dvh w-[min(85vw,320px)] border-l data-open:slide-in-from-right data-closed:slide-out-to-right",
  bottom:
    "inset-x-0 bottom-0 max-h-[85dvh] rounded-t-2xl border-t data-open:slide-in-from-bottom data-closed:slide-out-to-bottom max-[380px]:h-dvh max-[380px]:max-h-none max-[380px]:rounded-none",
}

export function Sheet({
  open,
  onOpenChange,
  side,
  title,
  closeLabel,
  className,
  finalFocus,
  onOpenChangeComplete,
  children,
}: SheetProps) {
  return (
    <DialogPrimitive.Root
      open={open}
      onOpenChange={(next) => onOpenChange(next)}
      onOpenChangeComplete={onOpenChangeComplete}
    >
      <DialogPrimitive.Portal>
        <DialogPrimitive.Backdrop className="fixed inset-0 z-50 bg-black/60 duration-150 data-open:animate-in data-open:fade-in-0 data-closed:animate-out data-closed:fade-out-0" />
        <DialogPrimitive.Popup
          finalFocus={finalFocus}
          className={cn(
            "fixed z-50 flex flex-col overflow-hidden border-border bg-card text-foreground shadow-2xl outline-none duration-200 data-open:animate-in data-closed:animate-out motion-reduce:animate-none",
            SIDE[side],
            className
          )}
        >
          <DialogPrimitive.Title className="sr-only">{title}</DialogPrimitive.Title>
          <DialogPrimitive.Close
            aria-label={closeLabel}
            className="absolute right-2 z-10 flex h-11 w-11 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
            style={{ top: side === "bottom" ? "6px" : "calc(env(safe-area-inset-top) + 6px)" }}
          >
            <XIcon className="h-5 w-5" />
          </DialogPrimitive.Close>
          {children}
        </DialogPrimitive.Popup>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  )
}
