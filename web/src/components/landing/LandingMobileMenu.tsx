import { useEffect, useRef, type MouseEvent } from "react"
import { Link, useLocation } from "react-router-dom"
import { LayoutDashboard, LogOut, Menu, MessageSquare, Settings } from "lucide-react"
import { Sheet } from "@/components/ui/sheet"
import { useAuth } from "@/context/AuthContext"
import { landingLinks } from "./landingLinks"

// The public pages' menu below md (LandingNav hides its section links there): a sheet from the right with the
// section links and the account's links. Base UI's Dialog (in Sheet) moves focus in on open and back to the menu
// button on close, closes on Esc and on a tap outside, and stops the page behind from scrolling.

const MD = "(min-width: 48rem)" // Tailwind's md breakpoint

const rowClass =
  "flex min-h-11 items-center gap-3 rounded-[10px] px-3 text-[15px] text-[#C9CDD8] transition-colors hover:bg-white/5 hover:text-[#F5F7FA] [&_svg]:h-[18px] [&_svg]:w-[18px] [&_svg]:shrink-0 [&_svg]:text-[#8B90A0]"
const separator = <div aria-hidden="true" className="my-3 h-px bg-[#1F2130]" />

type Props = {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export default function LandingMobileMenu({ open, onOpenChange }: Props) {
  const { user, logout } = useAuth()
  const location = useLocation()
  const menuButton = useRef<HTMLButtonElement>(null)
  // A section to scroll to once the menu has closed: while it's open the page's scroll is locked
  const pendingSection = useRef<string | null>(null)

  // Close it if the screen widens to md, where the header shows everything again
  useEffect(() => {
    if (!open) return
    const query = window.matchMedia(MD)
    const close = () => {
      if (query.matches) onOpenChange(false)
    }
    close()
    query.addEventListener("change", close)
    return () => query.removeEventListener("change", close)
  }, [open, onOpenChange])

  // On the landing page a section link scrolls once the menu has closed (to just below the sticky header, by the
  // section's scroll-margin). From /privacy or /case-study it's a link to /#how etc., where LandingPage scrolls.
  const onSectionClick = (event: MouseEvent<HTMLAnchorElement>, id: string) => {
    if (location.pathname === "/") {
      event.preventDefault()
      pendingSection.current = id
    }
    onOpenChange(false)
  }

  const onClosed = (isOpen: boolean) => {
    const id = pendingSection.current
    pendingSection.current = null
    if (isOpen || !id) return
    document.getElementById(id)?.scrollIntoView()
    // Record it in the address like the desktop's anchors do; keep the router's state so Back still works
    if (window.location.hash !== `#${id}`) window.history.pushState(window.history.state, "", `#${id}`)
  }

  const close = () => onOpenChange(false)
  const name = user ? user.display_name || user.email.split("@")[0] : ""

  return (
    <>
      <button
        ref={menuButton}
        type="button"
        aria-label="Open menu"
        aria-expanded={open}
        aria-haspopup="dialog"
        onClick={() => onOpenChange(true)}
        className="flex h-11 w-11 shrink-0 items-center justify-center rounded-[10px] text-[#C9CDD8] transition-colors hover:bg-white/5 hover:text-[#F5F7FA] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#60A5FA] md:hidden"
      >
        <Menu className="h-5 w-5" />
      </button>

      <Sheet
        open={open}
        onOpenChange={onOpenChange}
        onOpenChangeComplete={onClosed}
        side="right"
        title="Menu"
        closeLabel="Close menu"
        finalFocus={menuButton}
        className="border-[#1A1C26] bg-[#07070B]/80 text-[#F5F7FA] backdrop-blur-md"
      >
        <div
          className="flex min-h-0 flex-1 flex-col overflow-y-auto px-3"
          style={{
            paddingTop: "calc(env(safe-area-inset-top) + 64px)",
            paddingBottom: "calc(env(safe-area-inset-bottom) + 16px)",
          }}
        >
          {user && (
            <>
              <div className="flex items-center gap-3 px-3 pb-1">
                <span
                  aria-hidden="true"
                  className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-primary/15 text-sm font-semibold text-primary"
                >
                  {name[0]?.toUpperCase() ?? "?"}
                </span>
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium text-[#F5F7FA]">{name}</p>
                  <p className="truncate text-xs text-[#8B90A0]">{user.email}</p>
                </div>
              </div>
              {separator}
            </>
          )}

          <nav aria-label="Sections" className="flex flex-col">
            {landingLinks.map((link) =>
              link.href.startsWith("/#") ? (
                <Link
                  key={link.href}
                  to={link.href}
                  onClick={(event) => onSectionClick(event, link.href.slice(2))}
                  className={rowClass}
                >
                  {link.label}
                </Link>
              ) : (
                <Link key={link.href} to={link.href} onClick={close} className={rowClass}>
                  {link.label}
                </Link>
              )
            )}
          </nav>

          {separator}

          {user ? (
            <nav aria-label="Account" className="flex flex-col">
              <Link to="/chat" onClick={close} className={rowClass}>
                <MessageSquare aria-hidden="true" /> Open chat
              </Link>
              <Link to="/dashboard" onClick={close} className={rowClass}>
                <LayoutDashboard aria-hidden="true" /> Dashboard
              </Link>
              <Link to="/settings" onClick={close} className={rowClass}>
                <Settings aria-hidden="true" /> Settings
              </Link>
              {separator}
              {/* Logs out and stays here, as the desktop menu does */}
              <button
                type="button"
                onClick={() => {
                  close()
                  void logout()
                }}
                className={`${rowClass} w-full text-left`}
              >
                <LogOut aria-hidden="true" /> Log out
              </button>
            </nav>
          ) : (
            <div className="flex flex-col gap-2 pt-1">
              <Link
                to="/login"
                onClick={close}
                className="inline-flex min-h-11 w-full items-center justify-center rounded-[10px] border border-[#2A2D3D] text-sm font-medium transition-colors hover:bg-white/5"
              >
                Log in
              </Link>
              <Link
                to="/signup"
                onClick={close}
                className="inline-flex min-h-11 w-full items-center justify-center rounded-[10px] bg-[#2563EB] text-sm font-medium text-white transition-colors hover:bg-[#1D4ED8]"
              >
                Get started
              </Link>
            </div>
          )}
        </div>
      </Sheet>
    </>
  )
}
