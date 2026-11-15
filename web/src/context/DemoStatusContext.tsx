/* eslint-disable react-hooks/set-state-in-effect */
import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react"
import { getDemoStatus, type DemoHeaders } from "@/lib/api"
import type { DemoStatus } from "@/types/chat"

type DemoStatusContextValue = {
  status: DemoStatus | null   // null: demo is off, or Triage couldn't be reached
  loading: boolean
  refresh: () => Promise<void>
  update: (headers: DemoHeaders) => void
}

const DemoStatusContext = createContext<DemoStatusContextValue | null>(null)

export function DemoStatusProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<DemoStatus | null>(null)
  const [loading, setLoading] = useState(true)

  const refresh = useCallback(async () => {
    const latest = await getDemoStatus()
    setStatus(latest)
    setLoading(false)
  }, [])

  // Load once when the app opens
  useEffect(() => {
    refresh()
  }, [refresh])

  // Cheap update from a chat response's headers, no extra request
  const update = useCallback((headers: DemoHeaders) => {
    setStatus((prev) => ({
      enabled: true,
      paused: prev?.paused ?? false,
      limit: headers.limit,
      remaining: headers.remaining,
      used: headers.limit - headers.remaining,
      resetsAt: headers.resetsAt ?? prev?.resetsAt ?? "",
    }))
  }, [])

  return (
    <DemoStatusContext.Provider value={{ status, loading, refresh, update }}>
      {children}
    </DemoStatusContext.Provider>
  )
}

export function useDemoStatus(): DemoStatusContextValue {
  const value = useContext(DemoStatusContext)
  if (!value) throw new Error("useDemoStatus must be used inside DemoStatusProvider")
  return value
}