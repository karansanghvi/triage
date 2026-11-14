import { Outlet } from "react-router-dom"
import Sidebar from "./Sidebar"
import NamePrompt from "../NamePrompt";

export default function AppLayout() {
  return (
    <div className="flex h-screen bg-background text-foreground">
      <Sidebar />
      <main className="flex-1 overflow-y-auto p-6">
        <Outlet />
      </main>
      <NamePrompt />
    </div>
  )
}