import { Link } from "react-router-dom"
import { Split } from "lucide-react"

// Replace with your repository's address
const GITHUB_URL = "https://github.com/karansanghvi/triage"

export default function Footer() {
  return (
    <footer className="border-t border-[#1A1C26] px-6 py-8">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-4 text-[13px] text-[#8B90A0]">
        <div className="flex items-center gap-2.5">
          <span className="flex h-[26px] w-[26px] items-center justify-center rounded-lg bg-[#2563EB]">
            <Split className="h-3.5 w-3.5 text-white" strokeWidth={2.2} />
          </span>
          <span>Triage · Built by Karan Sanghvi</span>
        </div>
        <nav aria-label="Footer" className="flex flex-wrap gap-6">
          <a href={GITHUB_URL} target="_blank" rel="noreferrer" className="text-[#A7ACBA] hover:text-foreground">
            GitHub
          </a>
          <Link to="/privacy" className="text-[#A7ACBA] hover:text-foreground">
            Privacy
          </Link>
          <Link to="/dashboard" className="text-[#A7ACBA] hover:text-foreground">
            Dashboard
          </Link>
        </nav>
      </div>
    </footer>
  )
}