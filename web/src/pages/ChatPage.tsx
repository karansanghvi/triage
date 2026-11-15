import { useState } from "react"
import Welcome from "@/components/chat/Welcome"
import SuggestionChips from "@/components/chat/SuggestionChips"
import ChatInput from "@/components/chat/ChatInput"
import MessageList from "@/components/chat/MessageList"
import RoutingPanel from "@/components/chat/RoutingPanel"
import { useVisitorName } from "@/hooks/useVisitorName"
import { useDemoStatus } from "@/context/DemoStatusContext"
import { sendChat } from "@/lib/api"
import { TriageError } from "@/lib/errors"
import { DEFAULT_QUALITY } from "@/lib/quality"
import type { AssistantMessage, ChatError, Message, UserMessage } from "@/types/chat"

export default function ChatPage() {
  const { name } = useVisitorName()
  const demoStatus = useDemoStatus()
  const [input, setInput] = useState("")
  const [quality, setQuality] = useState(DEFAULT_QUALITY)
  const [messages, setMessages] = useState<Message[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [sending, setSending] = useState(false)

  const isEmpty = messages.length === 0
  const selectedMessage = messages.find((m) => m.id === selectedId)
  const selectedRouting =
    selectedMessage && selectedMessage.role === "assistant" ? selectedMessage.routing ?? null : null

  // Sends a conversation to Triage and fills in the answer
  async function requestAnswer(history: Message[]) {
    const pendingId = crypto.randomUUID()
    const pending: AssistantMessage = { id: pendingId, role: "assistant", content: "", status: "pending" }

    setMessages([...history, pending])
    setSending(true)

    try {
      const result = await sendChat(history, quality)
      if (result.demo) demoStatus.update(result.demo)
      setMessages((prev) =>
        prev.map((m) =>
          m.id === pendingId && m.role === "assistant"
            ? { ...m, content: result.content, status: "done", routing: result.routing, truncated: result.truncated }
            : m
        )
      )
      setSelectedId(pendingId)
    } catch (err) {
      demoStatus.refresh()

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

  function handleSend() {
    const content = input.trim()
    if (!content || sending) return

    const userMessage: UserMessage = { id: crypto.randomUUID(), role: "user", content }
    setInput("")
    requestAnswer([...messages, userMessage])
  }

  // Resend the conversation up to the failed answer
  function handleRetry(failedId: string) {
    if (sending) return
    const index = messages.findIndex((m) => m.id === failedId)
    if (index === -1) return
    requestAnswer(messages.slice(0, index))
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
              <MessageList
                messages={messages}
                selectedId={selectedId}
                onSelect={setSelectedId}
                onRetry={handleRetry}
              />
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