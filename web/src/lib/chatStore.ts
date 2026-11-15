import type { Message } from "@/types/chat";

const STORAGE_KEY = "triage.chats"
const MAX_CHATS = 30
const TITLE_LENGTH = 40

// ------------ Shapes -------------

export type Chat = {
    id: string
    title: string
    createdAt: string // ISO time 
    updatedAt: string // ISO time, used for storing 
    messages: Message[]
}

export type ChatStore = {
    version: 1
    activeId: string | null
    chats: Chat[]
}

export function emptyStore(): ChatStore {
  return { version: 1, activeId: null, chats: [] }
}

// ------------ Titles -------------

// A chat's title is its first question, shortened 
export function makeTitle(messages: Message[]): string {
    const first = messages.find((m) => m.role === "user")
    if (!first) return "New chat"
    const text = first.content.replace(/\s+/g, " ").trim()
    return text.length > TITLE_LENGTH ? `${text.slice(0, TITLE_LENGTH).trimEnd()}…` : text
}

// ------------ Reading safely -------------

// Data from storage could be anything (old formats, manual edits), so check it
function isChat(value: unknown): value is Chat {
    if (typeof value !== "object" || value === null) return false
    const c = value as Record<string, unknown>
    return (
        typeof c.id === "string" &&
        typeof c.title === "string" &&
        typeof c.createdAt === "string" && 
        typeof c.updatedAt === "string" &&
        Array.isArray(c.messages)
    )
}

// An answer still "thinking" when the page closed will never arrive 
function repairMessages(messages: Message[]): Message[] {
    return messages.map((m) => 
        m.role === "assistant" && m.status === "pending"
            ? {
                ...m,
                status: "error",
                error: { code: "interrupted", message: "This answer was interrupted when the page was closed." },
            }
        : m
    )
}

export function loadStore(): ChatStore {
    try {
        const raw = localStorage.getItem(STORAGE_KEY)
        if (!raw) return emptyStore()

        const parsed: unknown = JSON.parse(raw)
        if (typeof parsed !== "object" || parsed === null) return emptyStore()
        const data = parsed as Record<string, unknown>
        if (data.version !== 1 || !Array.isArray(data.chats)) return emptyStore()

        const chats = data.chats.filter(isChat).map((c) => ({ ...c, messages: repairMessages(c.messages) }))
        const activeId = typeof data.activeId === "string" && chats.some((c) => c.id === data.activeId) ? data.activeId : null

        return { version: 1, activeId, chats }
    } catch {
        return emptyStore() // broken data: start fresh instead of crashing 
    }
}

// ------------ Writing -------------

function newestFirst(chats: Chat[]): Chat[] {
    return [...chats].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))
}

// Saves the store. Empty chats are never saved, only the newest MAX_CHATS are kept,
// and if storage is full the oldest chats are dropped until it fits.
// Returns false if nothing could be saved (e.g. storage blocked).
export function saveStore(store: ChatStore): boolean {
  let chats = newestFirst(store.chats.filter((c) => c.messages.length > 0)).slice(0, MAX_CHATS)

  while (true) {
    const activeId = chats.some((c) => c.id === store.activeId) ? store.activeId : null
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ version: 1, activeId, chats }))
      return true
    } catch {
      if (chats.length <= 1) return false
      chats = chats.slice(0, -1) // storage full: drop the oldest chat and try again
    }
  }
}

export function clearStore(): void {
    localStorage.removeItem(STORAGE_KEY)
}

// ------------ Creating -------------

export function createChat(messages: Message[]): Chat {
    const now = new Date().toISOString()
    return {
        id: crypto.randomUUID(),
        title: makeTitle(messages),
        createdAt: now,
        updatedAt: now,
        messages,
    }
}