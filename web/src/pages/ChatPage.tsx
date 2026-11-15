import { useState } from "react"
import { Plus } from "lucide-react"
import Welcome from "@/components/chat/Welcome"
import SuggestionChips from "@/components/chat/SuggestionChips"
import ChatInput from "@/components/chat/ChatInput"
import MessageList from "@/components/chat/MessageList"
import RoutingPanel from "@/components/chat/RoutingPanel"
import { useVisitorName } from "@/hooks/useVisitorName"
import { useDemoStatus } from "@/context/DemoStatusContext"
import { useChats } from "@/context/ChatsContext"
import { sendChat } from "@/lib/api"
import { TriageError } from "@/lib/errors"
import { DEFAULT_QUALITY } from "@/lib/quality"
import type { AssistantMessage, ChatError, Message, UserMessage } from "@/types/chat"
import LimitBanner from "@/components/chat/LimitBanner";

export default function ChatPage() {
  const { name } = useVisitorName()
  const demoStatus = useDemoStatus()
  const { activeId, activeMessages: messages, updateMessages, newChat, busy, setBusy } = useChats()

  const [input, setInput] = useState("")
  const [quality, setQuality] = useState(DEFAULT_QUALITY)
  const [selectedId, setSelectedId] = useState<string | null>(null)

  const isEmpty = messages.length === 0
  // Free demo used up (when BYOK keys exist, this will also check that the visitor has none)
  const limitStatus = demoStatus.status
  const outOfRequests = !!limitStatus && (limitStatus.remaining === 0 || limitStatus.paused)

  // The selected answer: the one the visitor clicked, or else the latest answer in this chat.
  // Switching chats falls back to the latest answer automatically.
  const latestAnswer = [...messages].reverse().find((m) => m.role === "assistant" && m.routing)
  const clicked = messages.find((m) => m.id === selectedId)
  const selectedMessage = clicked ?? latestAnswer
  const selectedRouting =
    selectedMessage && selectedMessage.role === "assistant" ? selectedMessage.routing ?? null : null

  // Sends a conversation to Triage and fills in the answer
  async function requestAnswer(history: Message[], chatId: string | null) {
    const pendingId = crypto.randomUUID()
    const pending: AssistantMessage = { id: pendingId, role: "assistant", content: "", status: "pending" }

    // Passing null starts a new chat; keep its id so the answer lands in the same chat
    const id = updateMessages(chatId, () => [...history, pending])
    setBusy(true)

    try {
      const result = await sendChat(history, quality)
      if (result.demo) demoStatus.update(result.demo)
      updateMessages(id, (prev) =>
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

      updateMessages(id, (prev) =>
        prev.map((m) => (m.id === pendingId && m.role === "assistant" ? { ...m, status: "error", error } : m))
      )
    } finally {
      setBusy(false)
    }
  }

  function handleSend() {
    const content = input.trim()
    if (!content || busy) return

    const userMessage: UserMessage = { id: crypto.randomUUID(), role: "user", content }
    setInput("")
    requestAnswer([...messages, userMessage], activeId)
  }

  // Resend the conversation up to the failed answer
  function handleRetry(failedId: string) {
    if (busy) return
    const index = messages.findIndex((m) => m.id === failedId)
    if (index === -1) return
    requestAnswer(messages.slice(0, index), activeId)
  }

  function handleNewChat() {
    if (busy) return
    newChat()
    setSelectedId(null)
    setInput("")
  }

  const chatInput = (
    <>
        {outOfRequests && limitStatus && <LimitBanner status={limitStatus} />}
        <ChatInput
            value={input}
            onChange={setInput}
            onSend={handleSend}
            quality={quality}
            onQualityChange={setQuality}
        />
    </>
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
            <div className="flex items-center justify-end pb-2">
              <button
                onClick={handleNewChat}
                disabled={busy}
                className="flex items-center gap-2 rounded-lg px-3 py-1.5 text-xs font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40"
              >
                <Plus className="h-3.5 w-3.5" />
                New chat
              </button>
            </div>
            <div className="min-h-0 flex-1 overflow-y-auto">
              <MessageList
                messages={messages}
                selectedId={selectedMessage?.id ?? null}
                onSelect={setSelectedId}
                onRetry={handleRetry}
              />
            </div>
            <div className="pt-4">{chatInput}</div>
          </>
        )}
      </div>

      <RoutingPanel selected={selectedRouting} messages={messages} />
    </div>
  )
}