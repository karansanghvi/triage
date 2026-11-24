import { useRef, useState } from "react"
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
import { fetchStoppedCost, streamChat } from "@/lib/api"
import { server } from "@/lib/server"
import { isLongPrompt } from "@/lib/wake"
import { createSendGuard, type SendGuard } from "@/lib/sendGuard"
import { TriageError } from "@/lib/errors"
import { DEFAULT_QUALITY } from "@/lib/quality"
import { PROVIDERS, type ProviderKey } from "@/lib/account"
import type { AssistantMessage, ChatError, Message, RoutingInfo, UserMessage } from "@/types/chat"
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
  // The answer arriving now: shown as it streams, saved to the chat only once it finishes, stops or fails
  const [live, setLive] = useState<{ id: string; content: string } | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  // One request at a time, claimed synchronously: `busy` is React state and lags a render behind, so it alone
  // can't stop a second send in the same moment (src/lib/sendGuard.ts)
  const guardRef = useRef<SendGuard | null>(null)
  if (guardRef.current === null) guardRef.current = createSendGuard()
  const sendGuard = guardRef.current

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
    const pending: AssistantMessage = {
      id: pendingId, role: "assistant", content: "", status: "pending",
      ...(isLongPrompt(history) ? { longPrompt: true } : {}),
    }

    // Passing null starts a new chat; keep its id so the answer lands in the same chat
    const id = updateMessages(chatId, () => [...history, pending])
    setBusy(true)
    // A new chat is saved with its first message: give it its own address, replacing /chat in history
    if (chatId === null) navigate(`/chat/${id}`, { replace: true })

    const sessionToken = mode.kind === "account" ? mode.token : null
    const controller = new AbortController()
    abortRef.current = controller
    let text = ""                                   // what has arrived, kept here so Stop and errors can save it
    let started: RoutingInfo | null = null         // the routing, known once the first token exists
    let frame = 0                                    // one screen update per animation frame, not per token
    setLive({ id: pendingId, content: "" })
    const show = () => {
      if (frame) return
      frame = requestAnimationFrame(() => {
        frame = 0
        setLive({ id: pendingId, content: text })
      })
    }
    const finishAnswer = (patch: Partial<AssistantMessage>) =>
      updateMessages(id, (prev) =>
        prev.map((m) => (m.id === pendingId && m.role === "assistant" ? { ...m, ...patch } : m))
      )

    try {
      // The API may be asleep (it sleeps when idle): wait until it answers, while the answer shows it's waking up
      if (!(await server.ensureAwake(controller.signal))) {
        if (controller.signal.aborted) throw new DOMException("Stopped while the server was waking up", "AbortError")
        throw new TriageError("server_unreachable", "The server didn't wake up. Please try again in a minute.")
      }
      const result = await streamChat(history, quality, { sessionToken }, {
        signal: controller.signal,
        onStart: (routing, demo) => {
          started = routing
          if (mode.kind === "demo" && demo) demoStatus.update(demo)
        },
        onDelta: (piece) => {
          text += piece
          show()
        },
      })
      if (mode.kind === "demo" && result.demo) demoStatus.update(result.demo)
      finishAnswer({ content: result.content, status: "done", routing: result.routing, truncated: result.truncated })
      setSelectedId(pendingId)
      // A backup model answered: the backend may have just marked a saved key invalid, so reload them quietly
      if (mode.kind === "account" && result.routing.fellBack) void providerKeys.refresh()
    } catch (err) {
      if (mode.kind === "demo") demoStatus.refresh()

      if (controller.signal.aborted) {
        // Stopped: keep what arrived. Its cost isn't known yet: Triage estimates it once the model has stopped,
        // so it's read back by request id and filled in; if that fails, the answer stays "Stopped" without one
        const routing: RoutingInfo | undefined = started
          ? { ...(started as RoutingInfo), cost: null, strongCost: null, saved: null }
          : undefined
        finishAnswer({ content: text, status: "done", stopped: true, routing })
        if (routing) {
          setSelectedId(pendingId)
          void fetchStoppedCost(routing.requestId, sessionToken).then((cost) => {
            if (!cost) return
            finishAnswer({ routing: { ...routing, ...cost, costEstimated: true } })
          })
        }
        return
      }

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

      finishAnswer({ status: "error", error, content: text })
    } finally {
      if (frame) cancelAnimationFrame(frame)
      abortRef.current = null
      setLive(null)
      setBusy(false)
    }
  }

  function handleStop() {
    abortRef.current?.abort()
  }

  // Every request goes through the guard: one POST per message sent, never a second for the same one
  function send(history: Message[]): boolean {
    return sendGuard.run(() => requestAnswer(history, activeId, currentMode())) !== null
  }

  function handleSend() {
    const content = input.trim()
    if (!content || busy || sendGuard.busy()) return

    const userMessage: UserMessage = { id: crypto.randomUUID(), role: "user", content }
    if (send([...messages, userMessage])) setInput("")
  }

  // Resend the conversation up to the failed answer
  function handleRetry(failedId: string) {
    if (busy || sendGuard.busy()) return
    const index = messages.findIndex((m) => m.id === failedId)
    if (index === -1) return
    send(messages.slice(0, index))
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
            disabled={busy}
            streaming={busy && live !== null}
            onStop={handleStop}
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
                live={live}
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