import { useAuth } from "@/context/AuthContext"
import { useVisitorName } from "@/hooks/useVisitorName"

export function useDisplayName(): string {
  const { user } = useAuth()
  const { name: visitorName } = useVisitorName()
  if (user) return user.display_name || user.email.split("@")[0]
  return visitorName
}