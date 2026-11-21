import { useEffect, useRef } from "react"
import type { Message } from "@/types/chat"
import UserMessage from "./UserMessage"
import AssistantMessage from "./AssistantMessage"

type MessageListProps = {
  messages: Message[]
  selectedId: string | null
  onSelect: (id: string) => void
  onRetry: (id: string) => void
  live?: { id: string; content: string } | null   // the answer arriving now (shown, not yet saved)
}

// The nearest scrolling ancestor (the chat's message area)
function scroller(el: HTMLElement | null): HTMLElement | null {
  for (let p = el?.parentElement ?? null; p; p = p.parentElement) {
    const overflow = getComputedStyle(p).overflowY
    if (overflow === "auto" || overflow === "scroll") return p
  }
  return null
}

export default function MessageList({ messages, selectedId, onSelect, onRetry, live = null }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  // While an answer arrives, follow it, but only if the reader is already at the bottom (not reading above)
  useEffect(() => {
    const area = scroller(bottomRef.current)
    if (area && area.scrollHeight - area.scrollTop - area.clientHeight < 120) area.scrollTop = area.scrollHeight
  }, [live?.content])

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-6 py-4">
      {messages.map((message, index) =>
        message.role === "user" ? (
          <UserMessage key={message.id} content={message.content} />
        ) : (
          <AssistantMessage
            key={message.id}
            message={message}
            live={live && live.id === message.id ? live.content : undefined}
            selected={message.id === selectedId}
            onSelect={() => onSelect(message.id)}
            onRetry={index === messages.length - 1 ? () => onRetry(message.id) : undefined}
          />
        )
      )}
      <div ref={bottomRef} />
    </div>
  )
}