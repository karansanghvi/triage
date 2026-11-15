import { Trash2 } from "lucide-react"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"

type DeleteChatDialogProps = {
  open: boolean
  title: string
  onConfirm: () => void
  onCancel: () => void
}

export default function DeleteChatDialog({ open, title, onConfirm, onCancel }: DeleteChatDialogProps) {
  return (
    <Dialog
      open={open}
      onOpenChange={(isOpen) => {
        // Esc or clicking outside counts as Cancel
        if (!isOpen) onCancel()
      }}
    >
      <DialogContent showCloseButton={false} className="gap-0 p-0 sm:max-w-sm sm:rounded-2xl">
        <div className="p-6">
          <div className="mb-4 flex h-11 w-11 items-center justify-center rounded-xl bg-destructive/15">
            <Trash2 className="h-5 w-5 text-destructive" />
          </div>

          <DialogHeader className="text-left">
            <DialogTitle className="text-lg font-semibold">Delete chat?</DialogTitle>
            <DialogDescription className="mt-1 text-sm leading-relaxed text-muted-foreground">
              <span className="font-medium text-foreground">“{title}”</span> will be permanently deleted from this
              browser.
            </DialogDescription>
          </DialogHeader>
        </div>

        <div className="flex justify-end gap-2 border-t border-border px-6 py-4">
          <button
            onClick={onCancel}
            className="rounded-lg px-4 py-2 text-sm font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className="rounded-lg bg-destructive px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-destructive/90"
          >
            Delete
          </button>
        </div>
      </DialogContent>
    </Dialog>
  )
}