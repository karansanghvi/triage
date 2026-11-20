import { Link } from "react-router-dom"
import { Split } from "lucide-react"
import { useAuth } from "@/context/AuthContext"
import AccountMenu from "./AccountMenu"

const links = [
  // "/#..." so they also work from other public pages (e.g. /privacy); on the landing page they just scroll
  { href: "/#how", label: "How it works" },
  { href: "/#integrate", label: "Integrate" },
  { href: "/#results", label: "Results" },
]

export default function LandingNav() {
  const { user, loading } = useAuth()

  return (
    <header className="relative z-10 mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-4 px-6 py-6">
      <Link to="/" className="flex items-center gap-2.5">
        <span className="flex h-9 w-9 items-center justify-center rounded-[10px] bg-[#2563EB]">
          <Split className="h-5 w-5 text-white" />
        </span>
        <span className="text-xl font-semibold">Triage</span>
      </Link>

      <nav aria-label="Main" className="hidden items-center gap-7 text-sm md:flex">
        {links.map((link) => (
          <a key={link.href} href={link.href} className="text-[#A7ACBA] transition-colors hover:text-foreground">
            {link.label}
          </a>
        ))}
      </nav>

      {loading ? (
        // Checking a saved session: hold the account button's space, so the logged-out buttons never flash
        <div aria-hidden="true" className="h-10 w-[160px] animate-pulse rounded-[10px] bg-white/5" />
      ) : user ? (
        <AccountMenu />
      ) : (
      <div className="flex items-center gap-2.5">
        <Link
          to="/login"
          className="inline-flex min-h-10 items-center rounded-[10px] px-4 text-sm font-medium transition-colors hover:bg-white/5"
        >
          Log in
        </Link>
        <Link
          to="/signup"
          className="inline-flex min-h-10 items-center rounded-[10px] bg-[#2563EB] px-[18px] text-sm font-medium text-white transition-colors hover:bg-[#1D4ED8]"
        >
          Get started
        </Link>
      </div>
      )}
    </header>
  )
}