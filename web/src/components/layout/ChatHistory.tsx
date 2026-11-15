import { useNavigate, useLocation } from "react-router-dom"
import { MessageSquare, Plus } from "lucide-react"
import { useChats } from "@/context/ChatsContext"
import { cn } from "@/lib/utils"

// "just now", "5m ago", "3h ago", "2d ago", or a date for older chats
function relativeTime(iso: string): string {
  const seconds = (Date.now() - new Date(iso).getTime()) / 1000
  if (seconds < 60) return "just now"
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`
  if (seconds < 7 * 86400) return `${Math.floor(seconds / 86400)}d ago`
  return new Date(iso).toLocaleDateString([], { month: "short", day: "numeric" })
}

export default function ChatHistory() {
  const { chats, activeId, openChat, newChat, busy } = useChats()
  const navigate = useNavigate()
  const { pathname } = useLocation()
  const onChatPage = pathname === "/chat"

  function handleOpen(id: string) {
    if (busy) return
    openChat(id)
    navigate("/chat")
  }

  function handleNew() {
    if (busy) return
    newChat()
    navigate("/chat")
  }

  return (
    <div className="mt-8 flex min-h-0 flex-1 flex-col">
      <div className="mb-2 flex items-center justify-between px-3">
        <p className="text-xs font-medium text-muted-foreground">Recent chats</p>
        <button
          onClick={handleNew}
          disabled={busy}
          title="New chat"
          className="flex h-6 w-6 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40"
        >
          <Plus className="h-3.5 w-3.5" />
        </button>
      </div>

      {chats.length === 0 ? (
        <p className="px-3 text-xs text-muted-foreground/70">Your chats will appear here.</p>
      ) : (
        <div className="-mx-1 min-h-0 flex-1 space-y-0.5 overflow-y-auto px-1">
          {chats.map((chat) => {
            const active = onChatPage && chat.id === activeId
            return (
              <button
                key={chat.id}
                onClick={() => handleOpen(chat.id)}
                disabled={busy && !active}
                className={cn(
                  "group flex w-full items-start gap-2.5 rounded-lg px-3 py-2 text-left transition-colors disabled:cursor-not-allowed disabled:opacity-50",
                  active ? "bg-accent text-foreground" : "text-muted-foreground hover:bg-accent/60 hover:text-foreground"
                )}
              >
                <MessageSquare className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm">{chat.title}</p>
                  <p className="text-[11px] text-muted-foreground/70">{relativeTime(chat.updatedAt)}</p>
                </div>
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}