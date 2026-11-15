import { useState } from "react"
import Welcome from "@/components/chat/Welcome"
import SuggestionChips from "@/components/chat/SuggestionChips"
import ChatInput from "@/components/chat/ChatInput"
import MessageList from "@/components/chat/MessageList"
import { useVisitorName } from "@/hooks/useVisitorName"
import { sendChat } from "@/lib/api"
import { TriageError } from "@/lib/errors"
import type { AssistantMessage, ChatError, Message, UserMessage } from "@/types/chat"
import RoutingPanel from "@/components/chat/RoutingPanel";
import { DEFAULT_QUALITY } from "@/lib/quality";

export default function ChatPage() {
  const { name } = useVisitorName()
  const [input, setInput] = useState("")
  const [quality, setQuality] = useState(DEFAULT_QUALITY)
  const [messages, setMessages] = useState<Message[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [sending, setSending] = useState(false)

  const isEmpty = messages.length === 0
  const selectedMessage = messages.find((m) => m.id === selectedId)
  const selectedRouting =
    selectedMessage && selectedMessage.role === "assistant" ? selectedMessage.routing ?? null : null

  async function handleSend() {
    const content = input.trim()
    if (!content || sending) return

    const userMessage: UserMessage = { id: crypto.randomUUID(), role: "user", content }
    const pendingId = crypto.randomUUID()
    const pending: AssistantMessage = { id: pendingId, role: "assistant", content: "", status: "pending" }
    const history = [...messages, userMessage]

    setMessages([...history, pending])
    setInput("")
    setSending(true)

    try {
      const result = await sendChat(history, quality)
      setMessages((prev) =>
        prev.map((m) =>
          m.id === pendingId && m.role === "assistant"
            ? { ...m, content: result.content, status: "done", routing: result.routing }
            : m
        )
      )
      setSelectedId(pendingId)
    } catch (err) {
      const error: ChatError =
        err instanceof TriageError
          ? { code: err.code, message: err.message, retryAfter: err.retryAfter }
          : { code: "network_error", message: "Couldn't reach Triage. Please try again." }

      setMessages((prev) =>
        prev.map((m) => (m.id === pendingId && m.role === "assistant" ? { ...m, status: "error", error } : m))
      )
    } finally {
      setSending(false)
    }
  }

  const chatInput = (
    <ChatInput
      value={input}
      onChange={setInput}
      onSend={handleSend}
      quality={quality}
      onQualityChange={setQuality}
    />
  )

  return (
    <div className="flex h-full gap-6">
      <div className="flex min-h-0 flex-1 flex-col">
        {isEmpty ? (
          /* Scene 1: greeting, input and suggestions centered */
          <div className="flex flex-1 flex-col items-center justify-center">
            <Welcome name={name} />
            <div className="mt-10 w-full max-w-3xl">
              {chatInput}
              <SuggestionChips onPick={setInput} />
            </div>
          </div>
        ) : (
          /* Scene 2: messages scroll, input at the bottom */
          <>
            <div className="min-h-0 flex-1 overflow-y-auto">
              <MessageList messages={messages} selectedId={selectedId} onSelect={setSelectedId} />
            </div>
            <div className="pt-4">{chatInput}</div>
          </>
        )}
      </div>

      {/* Routing panel */}
      <RoutingPanel selected={selectedRouting} messages={messages} />
    </div>
  )
}