import type { ReactNode } from "react"
import { Navigate, useLocation } from "react-router-dom"
import { Loader2 } from "lucide-react"
import { useAuth } from "@/context/AuthContext"

export default function RequireAuth({ children }: { children: ReactNode }) {
  const { user, loading } = useAuth()
  const location = useLocation()

  // Still checking a saved session: don't bounce someone who is actually logged in
  if (loading) {
    return (
      <div className="flex h-full items-center justify-center text-[#8B90A0]">
        <Loader2 className="h-5 w-5 animate-spin" aria-label="Loading" />
      </div>
    )
  }

  if (!user) {
    return <Navigate to="/login" replace state={{ from: location.pathname }} />
  }

  return <>{children}</>
}