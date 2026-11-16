import { Route, Routes } from "react-router-dom"
import AppLayout from "@/components/layout/AppLayout"
import LandingPage from "@/pages/LandingPage"
import ChatPage from "@/pages/ChatPage"
import DashboardPage from "@/pages/DashboardPage"
import SettingsPage from "@/pages/SettingsPage"

export default function App() {
  return (
    <Routes>
      {/* Public landing page: no sidebar */}
      <Route path="/" element={<LandingPage />} />

      {/* The app: sidebar + pages */}
      <Route element={<AppLayout />}>
        <Route path="/chat" element={<ChatPage />} />
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/settings" element={<SettingsPage />} />
      </Route>
    </Routes>
  )
}