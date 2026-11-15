import { Outlet } from "react-router-dom"
import Sidebar from "./Sidebar"
import NamePrompt from "@/components/NamePrompt"
import { DemoStatusProvider } from "@/context/DemoStatusContext"

export default function AppLayout() {
  return (
    <DemoStatusProvider>
      <div className="flex h-screen bg-background text-foreground">
        <Sidebar />
        <main className="flex-1 overflow-y-auto p-6">
          <Outlet />
        </main>
        <NamePrompt />
      </div>
    </DemoStatusProvider>
  )
}