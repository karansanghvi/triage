import { Route, Routes } from "react-router-dom"
import AppLayout from "@/components/layout/AppLayout"
import LandingPage from "@/pages/LandingPage"
import ChatPage from "@/pages/ChatPage"
import DashboardPage from "@/pages/DashboardPage"
import SettingsPage from "@/pages/SettingsPage"
import SignupPage from "./pages/SignupPage";
import LoginPage from "./pages/LoginPage";
import RequireAuth from "./components/auth/RequireAuth";
import ForgotPasswordPage from "./pages/ForgotPasswordPage";

export default function App() {
  return (
    <Routes>
      {/* Public landing page: no sidebar */}
      <Route path="/" element={<LandingPage />} />
      <Route path="/signup" element={<SignupPage />} />
      <Route path="/login" element={<LoginPage />} />
      <Route path="/forgot-password" element={<ForgotPasswordPage />} />

      {/* The app: sidebar + pages */}
      <Route element={<AppLayout />}>
        <Route path="/chat" element={<ChatPage />} />
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route 
          path="/settings" 
          element={
            <RequireAuth>
              <SettingsPage />
            </RequireAuth>
          } 
        />
      </Route>
    </Routes>
  )
}