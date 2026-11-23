import { useEffect, useState } from "react"
import { Link } from "react-router-dom"
import { Split } from "lucide-react"
import { useAuth } from "@/context/AuthContext"
import { cn } from "@/lib/utils"
import AccountMenu from "./AccountMenu"
import LandingMobileMenu from "./LandingMobileMenu"
import { landingLinks } from "./landingLinks"

// The sticky header is 64px tall (h-16), plus the top safe area on notched phones. Sections and headings it can
// scroll over use scroll-mt-[calc(72px+env(safe-area-inset-top))] (64 + 8), and the /privacy and /case-study menus
// stick at lg:top-[88px] (64 + 24): change those together with it.

const navLinkClass =
  "text-[#A7ACBA] transition-colors hover:text-foreground pointer-coarse:inline-flex pointer-coarse:min-h-11 pointer-coarse:items-center"

// True once the page has scrolled past `offset` px. A passive scroll listener; React skips the re-render when the
// value doesn't change, so scrolling only re-renders when the state flips.
function useScrolledPast(offset: number): boolean {
  const [past, setPast] = useState(() => typeof window !== "undefined" && window.scrollY > offset)
  useEffect(() => {
    const update = () => setPast(window.scrollY > offset)
    update()
    window.addEventListener("scroll", update, { passive: true })
    return () => window.removeEventListener("scroll", update)
  }, [offset])
  return past
}

export default function LandingNav() {
  const { user, loading } = useAuth()
  const scrolled = useScrolledPast(8)
  const [menuOpen, setMenuOpen] = useState(false)

  return (
    <>
      {/* Above the header: scrolls away. With the header's 12px bottom margin, the page's top looks exactly as
          before (an 88px band), and the header is a compact 64px once stuck */}
      <div aria-hidden="true" className="h-3" />
      <header
        className={cn(
          "sticky top-0 z-50 mb-3 h-[calc(4rem+env(safe-area-inset-top))] border-b pt-[env(safe-area-inset-top)] transition-[background-color,border-color,backdrop-filter] duration-200 motion-reduce:transition-none",
          scrolled
            ? "border-[#1A1C26] bg-[#07070B]/80 backdrop-blur-md"
            : "border-transparent bg-transparent backdrop-blur-none"
        )}
      >
        <div className="mx-auto flex h-full max-w-6xl items-center justify-between gap-4 px-4 sm:px-6">
          <Link to="/" className="flex shrink-0 items-center gap-2.5 max-md:min-h-11 pointer-coarse:min-h-11">
            <span className="flex h-9 w-9 items-center justify-center rounded-[10px] bg-[#2563EB]">
              <Split className="h-5 w-5 text-white" />
            </span>
            <span className="text-xl font-semibold">Triage</span>
          </Link>

          <nav aria-label="Main" className="hidden items-center gap-4 whitespace-nowrap text-sm md:flex lg:gap-7">
            {landingLinks.map((link) =>
              // Section links need a real anchor (they scroll on the landing page); pages use the router
              link.href.startsWith("/#") ? (
                <a key={link.href} href={link.href} className={navLinkClass}>
                  {link.label}
                </a>
              ) : (
                <Link key={link.href} to={link.href} className={navLinkClass}>
                  {link.label}
                </Link>
              )
            )}
          </nav>

          {/* Below md: the primary action (Get started, or the account avatar) and the menu button; Log in and the
              section links are in the menu */}
          <div className="flex min-w-0 items-center gap-2 md:contents">
          {loading ? (
            // Checking a saved session: hold the account button's space, so the logged-out buttons never flash
            <div aria-hidden="true" className="h-10 w-[160px] shrink animate-pulse rounded-[10px] bg-white/5 max-md:h-11 max-md:w-11" />
          ) : user ? (
            <AccountMenu />
          ) : (
            <div className="flex shrink-0 items-center gap-2 sm:gap-2.5">
              <Link
                to="/login"
                className="inline-flex min-h-10 items-center rounded-[10px] px-3 text-sm font-medium transition-colors hover:bg-white/5 sm:px-4 max-md:hidden pointer-coarse:min-h-11"
              >
                Log in
              </Link>
              <Link
                to="/signup"
                className="inline-flex min-h-10 items-center whitespace-nowrap rounded-[10px] bg-[#2563EB] px-3 text-sm font-medium text-white transition-colors hover:bg-[#1D4ED8] sm:px-[18px] max-md:min-h-11 pointer-coarse:min-h-11"
              >
                Get started
              </Link>
            </div>
          )}
          <LandingMobileMenu open={menuOpen} onOpenChange={setMenuOpen} />
          </div>
        </div>
      </header>
    </>
  )
}
