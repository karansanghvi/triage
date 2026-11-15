import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react"
import type { Message } from "@/types/chat"
import { type Chat, type ChatStore, clearStore, createChat, emptyStore, loadStore, saveStore } from "@/lib/chatStore"

type MessagesUpdater = (messages: Message[]) => Message[]

type ChatsContextValue = {
  chats: Chat[]                 // newest first
  activeId: string | null       // null = a fresh, unsaved chat
  activeMessages: Message[]
  busy: boolean                 // an answer is loading: switching chats is blocked
  setBusy: (busy: boolean) => void
  updateMessages: (chatId: string | null, updater: MessagesUpdater) => string
  openChat: (id: string) => void
  newChat: () => void
  renameChat: (id: string, title: string) => void
  deleteChat: (id: string) => void
  clearAll: () => void
}

const ChatsContext = createContext<ChatsContextValue | null>(null)

export function ChatsProvider({ children }: { children: ReactNode }) {
  const [store, setStore] = useState<ChatStore>(loadStore)
  const [busy, setBusy] = useState(false)

  // Write every change to the browser
  useEffect(() => {
    saveStore(store)
  }, [store])

  // Change one chat's messages. Passing null starts a new chat and returns its id,
  // so later updates (like the answer arriving) go to the same chat.
  const updateMessages = useCallback((chatId: string | null, updater: MessagesUpdater): string => {
    const id = chatId ?? crypto.randomUUID()

    setStore((prev) => {
      const existing = prev.chats.find((c) => c.id === id)

      if (!existing) {
        if (chatId !== null) return prev // the chat was deleted meanwhile: drop the update
        const messages = updater([])
        if (messages.length === 0) return prev
        return { ...prev, activeId: id, chats: [createChat(messages, id), ...prev.chats] }
      }

      const now = new Date().toISOString()
      return {
        ...prev,
        chats: prev.chats.map((c) => (c.id === id ? { ...c, messages: updater(c.messages), updatedAt: now } : c)),
      }
    })

    return id
  }, [])

  const openChat = useCallback((id: string) => {
    setStore((prev) => (prev.chats.some((c) => c.id === id) ? { ...prev, activeId: id } : prev))
  }, [])

  const newChat = useCallback(() => {
    setStore((prev) => ({ ...prev, activeId: null }))
  }, [])

  const renameChat = useCallback((id: string, title: string) => {
    const clean = title.replace(/\s+/g, " ").trim()
    if (!clean) return
    setStore((prev) => ({
      ...prev,
      chats: prev.chats.map((c) => (c.id === id ? { ...c, title: clean } : c)),
    }))
  }, [])

  const deleteChat = useCallback((id: string) => {
    setStore((prev) => ({
      ...prev,
      activeId: prev.activeId === id ? null : prev.activeId,
      chats: prev.chats.filter((c) => c.id !== id),
    }))
  }, [])

  const clearAll = useCallback(() => {
    clearStore()
    setStore(emptyStore())
  }, [])

  const chats = useMemo(
    () => [...store.chats].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt)),
    [store.chats]
  )
  const activeMessages = store.chats.find((c) => c.id === store.activeId)?.messages ?? []

  return (
    <ChatsContext.Provider
      value={{
        chats,
        activeId: store.activeId,
        activeMessages,
        busy,
        setBusy,
        updateMessages,
        openChat,
        newChat,
        renameChat,
        deleteChat,
        clearAll,
      }}
    >
      {children}
    </ChatsContext.Provider>
  )
}

export function useChats(): ChatsContextValue {
  const value = useContext(ChatsContext)
  if (!value) throw new Error("useChats must be used inside ChatsProvider")
  return value
}