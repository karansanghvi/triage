import { useEffect, useRef } from "react"
import type { Message } from "@/types/chat"
import UserMessage from "./UserMessage"
import AssistantMessage from "./AssistantMessage"

type MessageListProps = {
  messages: Message[]
  selectedId: string | null
  onSelect: (id: string) => void
  onRetry: (id: string) => void
}

export default function MessageList({ messages, selectedId, onSelect, onRetry }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-6 py-4">
      {messages.map((message, index) =>
        message.role === "user" ? (
          <UserMessage key={message.id} content={message.content} />
        ) : (
          <AssistantMessage
            key={message.id}
            message={message}
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