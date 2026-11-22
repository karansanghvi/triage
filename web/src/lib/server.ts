// The one Waker for this page (src/lib/wake.ts), pointed at the API's /healthz.
import { useEffect, useState } from "react"
import { createWaker, type WakeState } from "@/lib/wake"

const BASE_URL = import.meta.env.VITE_TRIAGE_URL ?? "http://localhost:8000"

export const server = createWaker(`${BASE_URL}/healthz`)

// Starts a background ping (call once, when the site loads); the chat waits on server.ensureAwake() before sending
export function wakeServerInBackground(): void {
  void server.ensureAwake()
}

// The wake state, and how many seconds it has been waking (counting up once a second while it is)
export function useServerWake(): { state: WakeState; waitingSeconds: number } {
  const [state, setState] = useState(server.state())
  const [waitingSeconds, setWaitingSeconds] = useState(0)
  useEffect(() => server.subscribe(() => setState(server.state())), [])
  useEffect(() => {
    if (state !== "waking") return
    const update = () => {
      const since = server.wakingSince()
      setWaitingSeconds(since === null ? 0 : Math.floor((Date.now() - since) / 1000))
    }
    update()
    const timer = setInterval(update, 1000)
    return () => clearInterval(timer)
  }, [state])
  return { state, waitingSeconds: state === "waking" ? waitingSeconds : 0 }
}
