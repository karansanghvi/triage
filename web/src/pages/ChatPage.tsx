import { useState } from "react"
import { Navigate, useNavigate, useParams } from "react-router-dom"
import { Plus } from "lucide-react"
import Welcome from "@/components/chat/Welcome"
import SuggestionChips from "@/components/chat/SuggestionChips"
import ChatInput from "@/components/chat/ChatInput"
import MessageList from "@/components/chat/MessageList"
import RoutingPanel, { RoutingDetails } from "@/components/chat/RoutingPanel"
import { Sheet } from "@/components/ui/sheet"
import { useVisitorName } from "@/hooks/useVisitorName"
import { useDemoStatus } from "@/context/DemoStatusContext"
import { useChats } from "@/context/ChatsContext"
import { useAuth } from "@/context/AuthContext"
import { isUsableKey, useProviderKeys } from "@/context/ProviderKeysContext"
import { sendChat } from "@/lib/api"
import { TriageError } from "@/lib/errors"
import { DEFAULT_QUALITY } from "@/lib/quality"
import { PROVIDERS, type ProviderKey } from "@/lib/account"
import type { AssistantMessage, ChatError, Message, UserMessage } from "@/types/chat"
import LimitBanner from "@/components/chat/LimitBanner";
import KeysRejectedBanner from "@/components/chat/KeysRejectedBanner"

// How one request is sent: the free demo (no Authorization, as for guests), or account mode with the
// session token, where the backend uses only the account's saved keys and preferred provider
type SendMode = { kind: "demo" } | { kind: "account"; token: string; keys: ProviderKey[] }

// After an account-mode key error, the card to show, from the keys reloaded just now (null: reload failed)
function keyErrorAfterRefresh(code: string, before: ProviderKey[], after: ProviderKey[] | null): ChatError {
  if (after === null) {
    return code === "no_provider_keys"
      ? { code: "keys_changed", message: "" }
      : { code: "saved_key_rejected", message: "" }
  }
  if (after.some(isUsableKey)) {
    if (code === "no_provider_keys") return { code: "keys_changed", message: "" }
    // The provider whose key was usable when this was sent and isn't any more
    const nowUsable = new Set(after.filter(isUsableKey).map((k) => k.provider))
    const rejected = before.filter((k) => isUsableKey(k) && !nowUsable.has(k.provider)).map((k) => k.provider)
    const names = PROVIDERS.filter((p) => rejected.includes(p.id)).map((p) => p.name)
    return { code: "saved_key_rejected", message: "", provider: names.length === 1 ? names[0] : undefined }
  }
  return { code: code === "no_provider_keys" ? "keys_removed_demo" : "keys_rejected_demo", message: "" }
}

export default function ChatPage() {
  const { name } = useVisitorName()
  const demoStatus = useDemoStatus()
  const { user, token, expireSession } = useAuth()
  const providerKeys = useProviderKeys()
  const { chats, updateMessages, busy, setBusy } = useChats()
  const navigate = useNavigate()
  // /chat is always a new, empty chat; /chat/:id is that saved chat
  const { id: routeId } = useParams()
  const activeChat = routeId ? chats.find((c) => c.id === routeId) : undefined
  const activeId = activeChat?.id ?? null
  const messages = activeChat?.messages ?? []

  const [input, setInput] = useState("")
  const [quality, setQuality] = useState(DEFAULT_QUALITY)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [detailsOpen, setDetailsOpen] = useState(false) // below lg: the routing details as a bottom sheet

  const isEmpty = messages.length === 0

  // Logged in with at least one usable saved key: account mode. Anyone else (guests, no keys, every key
  // rejected, or the keys still loading or failed to load): the free demo.
  const accountMode = !!user && !!token && providerKeys.hasUsableKey
  const onDemoBecauseRejected = !!user && !accountMode && providerKeys.allRejected

  // Free demo used up: only matters for requests that go to the demo
  const limitStatus = demoStatus.status
  const outOfRequests = !accountMode && !!limitStatus && (limitStatus.remaining === 0 || limitStatus.paused)

  // Decided when each request is sent (and again on Try again), from the latest login and keys
  function currentMode(): SendMode {
    return accountMode && token ? { kind: "account", token, keys: providerKeys.keys ?? [] } : { kind: "demo" }
  }

  // The selected answer: the one the visitor clicked, or else the latest answer in this chat.
  // Switching chats falls back to the latest answer automatically.
  const latestAnswer = [...messages].reverse().find((m) => m.role === "assistant" && m.routing)
  const clicked = messages.find((m) => m.id === selectedId)
  const selectedMessage = clicked ?? latestAnswer
  const selectedRouting =
    selectedMessage && selectedMessage.role === "assistant" ? selectedMessage.routing ?? null : null

  // Sends a conversation to Triage and fills in the answer
  async function requestAnswer(history: Message[], chatId: string | null, mode: SendMode) {
    const pendingId = crypto.randomUUID()
    const pending: AssistantMessage = { id: pendingId, role: "assistant", content: "", status: "pending" }

    // Passing null starts a new chat; keep its id so the answer lands in the same chat
    const id = updateMessages(chatId, () => [...history, pending])
    setBusy(true)
    // A new chat is saved with its first message: give it its own address, replacing /chat in history
    if (chatId === null) navigate(`/chat/${id}`, { replace: true })

    try {
      const result = await sendChat(history, quality, { sessionToken: mode.kind === "account" ? mode.token : null })
      if (mode.kind === "demo" && result.demo) demoStatus.update(result.demo)
      updateMessages(id, (prev) =>
        prev.map((m) =>
          m.id === pendingId && m.role === "assistant"
            ? { ...m, content: result.content, status: "done", routing: result.routing, truncated: result.truncated }
            : m
        )
      )
      setSelectedId(pendingId)
      // A backup model answered: the backend may have just marked a saved key invalid, so reload them quietly
      if (mode.kind === "account" && result.routing.fellBack) void providerKeys.refresh()
    } catch (err) {
      if (mode.kind === "demo") demoStatus.refresh()

      let error: ChatError =
        err instanceof TriageError
          ? { code: err.code, message: err.message, retryAfter: err.retryAfter }
          : { code: "network_error", message: "Couldn't reach Triage. Please try again." }

      if (mode.kind === "account") {
        if (error.code === "invalid_provider_key" || error.code === "no_provider_keys") {
          // The backend has marked the rejected key invalid (or the keys were removed elsewhere): reload them
          const after = await providerKeys.refresh()
          error = keyErrorAfterRefresh(error.code, mode.keys, after)
        } else if (error.code === "invalid_session") {
          expireSession()           // log out locally; Try again then goes to the free demo
        }
      }

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
    requestAnswer([...messages, userMessage], activeId, currentMode())
  }

  // Resend the conversation up to the failed answer
  function handleRetry(failedId: string) {
    if (busy) return
    const index = messages.findIndex((m) => m.id === failedId)
    if (index === -1) return
    requestAnswer(messages.slice(0, index), activeId, currentMode())
  }

  // Tapping an answer's footer: select it, and below lg (no side panel) open the details as a sheet
  function selectAnswer(id: string) {
    setSelectedId(id)
    if (!window.matchMedia("(min-width: 1024px)").matches) setDetailsOpen(true)
  }

  function handleNewChat() {
    if (busy) return
    setSelectedId(null)
    setInput("")
    navigate("/chat")
  }

  // A chat that doesn't exist (deleted, or a stale link): start fresh instead
  if (routeId && !activeChat) return <Navigate to="/chat" replace />

  const chatInput = (
    <>
        {outOfRequests && limitStatus ? (
          <LimitBanner status={limitStatus} />
        ) : (
          onDemoBecauseRejected && <KeysRejectedBanner />
        )}
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
                className="flex items-center gap-2 rounded-lg px-3 py-1.5 text-xs font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40 max-md:min-h-11 pointer-coarse:min-h-11"
              >
                <Plus className="h-3.5 w-3.5" />
                New chat
              </button>
            </div>
            <div className="min-h-0 flex-1 overflow-y-auto">
              <MessageList
                messages={messages}
                selectedId={selectedMessage?.id ?? null}
                onSelect={selectAnswer}
                onRetry={handleRetry}
              />
            </div>
            <div className="pt-4">{chatInput}</div>
          </>
        )}
      </div>

      <RoutingPanel selected={selectedRouting} messages={messages} />
      <Sheet
        open={detailsOpen}
        onOpenChange={setDetailsOpen}
        side="bottom"
        title="How Triage answered"
        closeLabel="Close details"
        className="lg:hidden"
      >
        <div className="flex min-h-0 flex-1 flex-col" style={{ paddingBottom: "env(safe-area-inset-bottom)" }}>
          <RoutingDetails selected={selectedRouting} messages={messages} />
        </div>
      </Sheet>
    </div>
  )
}