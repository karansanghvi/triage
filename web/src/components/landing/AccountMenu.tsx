import { Link } from "react-router-dom"
import { ChevronDown, LayoutDashboard, LogOut, MessageSquare, Settings } from "lucide-react"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { useAuth } from "@/context/AuthContext"

const itemClass =
  "cursor-pointer gap-2.5 rounded-lg px-2.5 py-2 text-[13px] text-[#C9CDD8] focus:bg-[#1A1C26] focus:text-[#F5F7FA] [&_svg]:text-[#8B90A0] focus:[&_svg]:text-[#F5F7FA]"

// The landing nav's top right for a logged-in user: avatar and name, opening a menu
export default function AccountMenu() {
  const { user, logout } = useAuth()
  if (!user) return null
  const name = user.display_name || user.email.split("@")[0]

  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label={`Account menu for ${name}`}
        className="inline-flex h-10 max-w-[220px] items-center gap-2 rounded-[10px] border border-[#2A2D3D] bg-[#12131A]/60 pl-1.5 pr-2.5 text-sm font-medium transition-colors outline-none hover:border-[#3B4058] hover:bg-[#12131A] focus-visible:ring-2 focus-visible:ring-[#60A5FA] focus-visible:ring-offset-2 focus-visible:ring-offset-[#07070B] data-popup-open:border-[#3B4058]"
      >
        <span
          aria-hidden="true"
          className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary/15 text-[13px] font-semibold text-primary"
        >
          {name[0]?.toUpperCase() ?? "?"}
        </span>
        <span className="min-w-0 truncate">{name}</span>
        <ChevronDown aria-hidden="true" className="h-3.5 w-3.5 shrink-0 text-[#8B90A0]" />
      </DropdownMenuTrigger>

      <DropdownMenuContent
        align="end"
        sideOffset={8}
        className="w-60 rounded-xl border border-[#2A2D3D] bg-[#12131A] p-1.5 text-[#F5F7FA] shadow-[0_16px_48px_rgba(0,0,0,0.5)] ring-0"
      >
        <div className="px-2.5 pb-2 pt-1.5">
          <p className="truncate text-sm font-medium text-[#F5F7FA]">{name}</p>
          <p className="truncate text-xs text-[#8B90A0]">{user.email}</p>
        </div>
        <DropdownMenuSeparator className="mx-0 bg-[#1F2130]" />
        <DropdownMenuItem className={itemClass} render={<Link to="/chat" />}>
          <MessageSquare /> Open chat
        </DropdownMenuItem>
        <DropdownMenuItem className={itemClass} render={<Link to="/dashboard" />}>
          <LayoutDashboard /> Dashboard
        </DropdownMenuItem>
        <DropdownMenuItem className={itemClass} render={<Link to="/settings" />}>
          <Settings /> Settings
        </DropdownMenuItem>
        <DropdownMenuSeparator className="mx-0 bg-[#1F2130]" />
        {/* Logs out and stays here: the nav then shows Log in / Get started */}
        <DropdownMenuItem className={itemClass} onClick={() => void logout()}>
          <LogOut /> Log out
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
