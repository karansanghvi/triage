import { useState } from "react"
import Welcome from "@/components/chat/Welcome"
import ChatInput from "@/components/chat/ChatInput"
import { useVisitorName } from "@/hooks/useVisitorName"
import SuggestionChips from "@/components/chat/SuggestionChipsProps";

export default function ChatPage() {
  const { name } = useVisitorName()
  const [input, setInput] = useState("")
  const [quality, setQuality] = useState(70)

  function handleSend() {
    // Connected to the Triage API in a later step
    console.log("send:", input, "quality:", quality)
    setInput("")
  }

  return (
    <div className="flex h-full gap-6">
      {/* Main chat area: greeting, input and suggestions centered together */}
      <div className="flex flex-1 flex-col items-center justify-center">
        <Welcome name={name} />

        <div className="mt-10 w-full max-w-3xl">
          <ChatInput
            value={input}
            onChange={setInput}
            onSend={handleSend}
            quality={quality}
            onQualityChange={setQuality}
          />
          <SuggestionChips onPick={setInput} />
        </div>
      </div>

      {/* Routing panel: built in the next step */}
      <aside className="hidden w-80 shrink-0 rounded-2xl border border-border bg-card p-5 xl:block">
        <p className="text-sm font-semibold">Routing</p>
        <p className="mt-2 text-xs text-muted-foreground">Send a message to see how Triage routes it.</p>
      </aside>
    </div>
  )
}