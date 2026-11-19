import { Navigate, Route, Routes } from "react-router-dom"
import AppLayout from "@/components/layout/AppLayout"
import LandingPage from "@/pages/LandingPage"
import ChatPage from "@/pages/ChatPage"
import DashboardPage from "@/pages/DashboardPage"
import SignupPage from "./pages/SignupPage";
import LoginPage from "./pages/LoginPage";
import RequireAuth from "./components/auth/RequireAuth";
import ForgotPasswordPage from "./pages/ForgotPasswordPage";
import SettingsLayout from "./components/settings/SettingsLayout";
import GetStartedPage from "./components/settings/GetStartedPage";
import ApiKeysPage from "./components/settings/ApiKeysPage";
import ProviderKeysPage from "./components/settings/ProviderKeysPage";
import AccountPage from "./components/settings/AccountPage";

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
              <SettingsLayout />
            </RequireAuth>
          } 
        >
          <Route index element={<Navigate to="get-started" replace />} />
          <Route path="get-started" element={<GetStartedPage />} />
          <Route path="api-keys" element={<ApiKeysPage />} />
          <Route path="provider-keys" element={<ProviderKeysPage />} />
          <Route path="account" element={<AccountPage />} />
        </Route>
      </Route>
    </Routes>
  )
}