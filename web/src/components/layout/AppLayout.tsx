import { Outlet } from "react-router-dom"
import Sidebar from "./Sidebar"
import MobileNav from "./MobileNav"
import NamePrompt from "@/components/NamePrompt"
import { DemoStatusProvider } from "@/context/DemoStatusContext"
import { ChatsProvider } from "@/context/ChatsContext"
import { ProviderKeysProvider } from "@/context/ProviderKeysContext"

export default function AppLayout() {
  return (
    <DemoStatusProvider>
      <ProviderKeysProvider>
        <ChatsProvider>
          {/* Below md: top bar + drawer (MobileNav). md and up: the fixed sidebar, as before */}
          <div className="flex h-dvh flex-col bg-background text-foreground md:flex-row">
            <MobileNav />
            <Sidebar />
            <main className="min-h-0 min-w-0 flex-1 overflow-y-auto p-4 pb-[max(1rem,env(safe-area-inset-bottom))] md:p-6">
              <Outlet />
            </main>
            <NamePrompt />
          </div>
        </ChatsProvider>
      </ProviderKeysProvider>
    </DemoStatusProvider>
  )
}