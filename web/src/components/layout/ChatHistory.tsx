import { useEffect, useRef, useState } from "react"
import { useMatch, useNavigate } from "react-router-dom"
import { MessageSquare, MoreHorizontal, Pencil, Plus, Trash2 } from "lucide-react"
import { useChats } from "@/context/ChatsContext"
import type { Chat } from "@/lib/chatStore"
import { cn } from "@/lib/utils"
import DeleteChatDialog from "./DeleteChatDialog"

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
  const { chats, busy } = useChats()
  const navigate = useNavigate()
  // The sidebar sits outside the chat route, so read the open chat from the address directly
  const activeId = useMatch("/chat/:id")?.params.id ?? null

  function handleNew() {
    if (busy) return
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
          className="flex h-6 w-6 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40 max-md:h-11 max-md:w-11 pointer-coarse:h-11 pointer-coarse:w-11"
        >
          <Plus className="h-3.5 w-3.5" />
        </button>
      </div>

      {chats.length === 0 ? (
        <p className="px-3 text-xs text-muted-foreground/70">Your chats will appear here.</p>
      ) : (
        <div className="-mx-1 min-h-0 flex-1 space-y-0.5 overflow-y-auto px-1">
          {chats.map((chat) => (
            <ChatItem
              key={chat.id}
              chat={chat}
              active={chat.id === activeId}
              busy={busy}
              onOpen={() => {
                if (busy || chat.id === activeId) return
                navigate(`/chat/${chat.id}`)
              }}
            />
          ))}
        </div>
      )}
    </div>
  )
}

type Mode = "idle" | "menu" | "renaming"

type ChatItemProps = {
  chat: Chat
  active: boolean
  busy: boolean
  onOpen: () => void
}

function ChatItem({ chat, active, busy, onOpen }: ChatItemProps) {
  const { renameChat, deleteChat } = useChats()
  const navigate = useNavigate()
  const [mode, setMode] = useState<Mode>("idle")
  const [confirmingDelete, setConfirmingDelete] = useState(false)
  const [draft, setDraft] = useState(chat.title)
  const itemRef = useRef<HTMLDivElement>(null)

  // Close the menu when clicking anywhere else
  useEffect(() => {
    if (mode !== "menu") return
    function handleClick(e: MouseEvent) {
      if (itemRef.current && !itemRef.current.contains(e.target as Node)) setMode("idle")
    }
    document.addEventListener("mousedown", handleClick)
    return () => document.removeEventListener("mousedown", handleClick)
  }, [mode])

  function startRename() {
    setDraft(chat.title)
    setMode("renaming")
  }

  function saveRename() {
    if (draft.trim()) renameChat(chat.id, draft)
    setMode("idle")
  }

  // ---------- Renaming: the title becomes a text field ----------
  if (mode === "renaming") {
    return (
      <div className="flex items-center gap-1 rounded-lg bg-accent px-2 py-1.5">
        <input
          autoFocus
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") saveRename()
            if (e.key === "Escape") setMode("idle")
          }}
          onBlur={saveRename}
          maxLength={80}
          className="min-w-0 flex-1 rounded-md border border-primary/50 bg-background px-2 py-1 text-sm outline-none"
        />
      </div>
    )
  }

  // ---------- Normal row, with the ⋯ menu ----------
  return (
    <>
      <div
        ref={itemRef}
        className={cn(
          "group relative flex items-start rounded-lg transition-colors",
          active ? "bg-accent" : "hover:bg-accent/60",
          busy && !active && "opacity-50"
        )}
      >
        <button
          onClick={onOpen}
          disabled={busy && !active}
          className={cn(
            "flex min-w-0 flex-1 items-start gap-2.5 px-3 py-2 text-left disabled:cursor-not-allowed max-md:min-h-11 pointer-coarse:min-h-11",
            active ? "text-foreground" : "text-muted-foreground hover:text-foreground"
          )}
        >
          <MessageSquare className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <div className="min-w-0 flex-1 pr-6 pointer-coarse:pr-10">
            <p className="truncate text-sm">{chat.title}</p>
            <p className="text-[11px] text-muted-foreground/70">{relativeTime(chat.updatedAt)}</p>
          </div>
        </button>

        {/* ⋯ button: visible on hover, while the menu is open, and on the open chat */}
        <button
          onClick={() => setMode(mode === "menu" ? "idle" : "menu")}
          disabled={busy}
          title="Chat options"
          className={cn(
            "absolute right-1.5 top-2 flex h-6 w-6 items-center justify-center rounded-md text-muted-foreground transition-opacity hover:bg-background hover:text-foreground disabled:hidden pointer-coarse:right-0 pointer-coarse:top-0.5 pointer-coarse:h-11 pointer-coarse:w-11",
            mode === "menu" || active ? "opacity-100" : "opacity-0 group-hover:opacity-100 pointer-coarse:opacity-100"
          )}
        >
          <MoreHorizontal className="h-4 w-4" />
        </button>

        {mode === "menu" && (
          <div className="absolute right-1.5 top-9 z-20 w-36 pointer-coarse:top-12 overflow-hidden rounded-lg border border-border bg-popover p-1 shadow-xl">
            <button
              onClick={startRename}
              className="flex w-full items-center gap-2 rounded-md px-2.5 py-1.5 text-xs text-foreground hover:bg-accent max-md:min-h-11 pointer-coarse:min-h-11"
            >
              <Pencil className="h-3.5 w-3.5" /> Rename
            </button>
            <button
              onClick={() => {
                setMode("idle")
                setConfirmingDelete(true)
              }}
              className="flex w-full items-center gap-2 rounded-md px-2.5 py-1.5 text-xs text-destructive hover:bg-destructive/10 max-md:min-h-11 pointer-coarse:min-h-11"
            >
              <Trash2 className="h-3.5 w-3.5" /> Delete
            </button>
          </div>
        )}
      </div>

      <DeleteChatDialog
        open={confirmingDelete}
        title={chat.title}
        onCancel={() => setConfirmingDelete(false)}
        onConfirm={() => {
          setConfirmingDelete(false)
          deleteChat(chat.id)
          if (active) navigate("/chat", { replace: true })     // the chat being viewed is gone
        }}
      />
    </>
  )
}