import { useState } from "react"
import { ArrowRight, Split, User } from "lucide-react"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { useVisitorName } from "@/hooks/useVisitorName"
import { cn } from "@/lib/utils"

export default function NamePrompt() {
  const { asked, setName } = useVisitorName()
  const [draft, setDraft] = useState("")
  const canSave = draft.trim().length > 0

  function handleSave() {
    if (canSave) setName(draft)
  }

  return (
    <Dialog open={!asked}>
      <DialogContent
        showCloseButton={false}
        aria-describedby={undefined}
        className="overflow-hidden border-border/60 bg-card p-0 shadow-2xl shadow-primary/10 sm:max-w-md sm:rounded-2xl"
      >
        {/* Soft blue glow at the top */}
        <div className="pointer-events-none absolute inset-x-0 -top-24 h-48 bg-primary/20 blur-3xl" />

        <div className="relative px-8 pb-8 pt-10">
          {/* Logo mark */}
          <div className="mx-auto mb-6 flex h-14 w-14 items-center justify-center rounded-2xl bg-primary shadow-lg shadow-primary/40">
            <Split className="h-7 w-7 text-primary-foreground" />
          </div>

          <DialogHeader>
            <DialogTitle className="text-center text-2xl font-semibold tracking-tight">
              Welcome to Triage
            </DialogTitle>
          </DialogHeader>

          {/* Name input with icon */}
          <div className="relative mt-8">
            <User className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <input
              autoFocus
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSave()}
              placeholder="Enter your name"
              className="h-12 w-full rounded-xl border border-input bg-background pl-11 pr-4 text-sm outline-none transition-all placeholder:text-muted-foreground focus:border-primary focus:ring-4 focus:ring-primary/15"
            />
          </div>

          {/* Continue button */}
          <button
            onClick={handleSave}
            disabled={!canSave}
            className={cn(
              "group mt-4 flex h-12 w-full items-center justify-center gap-2 rounded-xl text-sm font-medium transition-all",
              canSave
                ? "bg-primary text-primary-foreground shadow-lg shadow-primary/30 hover:bg-primary/90"
                : "cursor-not-allowed bg-muted text-muted-foreground"
            )}
          >
            Continue
            <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
          </button>
        </div>
      </DialogContent>
    </Dialog>
  )
}