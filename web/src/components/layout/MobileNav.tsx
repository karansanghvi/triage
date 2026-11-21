import { useRef, useState } from "react"
import { Link, useLocation, useNavigate } from "react-router-dom"
import { Menu, Split, SquarePen } from "lucide-react"
import { Sheet } from "@/components/ui/sheet"
import { useChats } from "@/context/ChatsContext"
import { SidebarContent } from "./Sidebar"

// Below md: a slim top bar (menu, logo, and New chat on chat pages) instead of the fixed sidebar, and the
// sidebar's content as a drawer from the left. The drawer belongs to the address it was opened on, so any
// navigation (a page, a chat, New chat) closes it without extra wiring.
export default function MobileNav() {
  const location = useLocation()
  const navigate = useNavigate()
  const { busy } = useChats()
  const [openedAt, setOpenedAt] = useState<string | null>(null)
  const menuButton = useRef<HTMLButtonElement>(null)
  const open = openedAt === location.key
  const onChatPage = location.pathname.startsWith("/chat")

  return (
    <>
      <header
        className="flex shrink-0 items-center gap-1 border-b border-border bg-card px-2 md:hidden"
        style={{ paddingTop: "env(safe-area-inset-top)", minHeight: "calc(56px + env(safe-area-inset-top))" }}
      >
        <button
          ref={menuButton}
          type="button"
          aria-label="Open menu"
          aria-expanded={open}
          onClick={() => setOpenedAt(location.key)}
          className="flex h-11 w-11 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
        >
          <Menu className="h-5 w-5" />
        </button>
        <Link to="/" className="flex min-h-11 items-center gap-2.5 rounded-lg px-1.5">
          <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary">
            <Split className="h-4 w-4 text-primary-foreground" />
          </span>
          <span className="text-lg font-semibold">Triage</span>
        </Link>
        {onChatPage && (
          <button
            type="button"
            aria-label="New chat"
            title="New chat"
            disabled={busy}
            onClick={() => navigate("/chat")}
            className="ml-auto flex h-11 w-11 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40"
          >
            <SquarePen className="h-5 w-5" />
          </button>
        )}
      </header>

      <Sheet
        open={open}
        onOpenChange={(next) => setOpenedAt(next ? location.key : null)}
        side="left"
        title="Menu"
        closeLabel="Close menu"
        finalFocus={menuButton}
      >
        <div
          className="flex min-h-0 flex-1 flex-col overflow-y-auto p-4"
          style={{ paddingTop: "calc(env(safe-area-inset-top) + 16px)", paddingBottom: "calc(env(safe-area-inset-bottom) + 16px)" }}
        >
          <SidebarContent />
        </div>
      </Sheet>
    </>
  )
}
