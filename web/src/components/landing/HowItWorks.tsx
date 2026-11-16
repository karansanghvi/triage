import { Gauge, RefreshCw, ShieldCheck, Split, type LucideIcon } from "lucide-react"
import { cn } from "@/lib/utils"
import SectionHeading from "./SectionHeading"

type Step = {
  number: string
  title: string
  body: string
  icon: LucideIcon
  highlight?: boolean
}

const STEPS: Step[] = [
  { number: "01", title: "Predict", body: "Estimates how likely each model is to answer well.", icon: Gauge },
  { number: "02", title: "Route", body: "Sends it to the cheapest model that clears your bar.", icon: Split, highlight: true },
  { number: "03", title: "Protect", body: "Fallbacks and caching keep answers coming.", icon: ShieldCheck },
  { number: "04", title: "Learn", body: "Retrains on real traffic to route better over time.", icon: RefreshCw },
]

export default function HowItWorks() {
  return (
    <section id="how" className="scroll-mt-8 px-6 pb-[120px]">
      <div className="mx-auto max-w-6xl">
        <SectionHeading eyebrow="How it works" title="A decision for every single prompt." />

        <div className="relative mt-14">
          {/* The line connecting the four steps (wide screens only) */}
          <svg
            aria-hidden="true"
            viewBox="0 0 1000 4"
            preserveAspectRatio="none"
            className="pointer-events-none absolute left-[12.5%] top-[34px] hidden h-1 w-3/4 overflow-visible lg:block"
          >
            <line x1="0" y1="2" x2="1000" y2="2" stroke="#1F2130" strokeWidth="2" />
            <line x1="0" y1="2" x2="1000" y2="2" stroke="#3B82F6" strokeOpacity="0.7" strokeWidth="2" className="triage-flow" />
          </svg>

          <ol className="relative grid grid-cols-1 gap-x-6 gap-y-10 sm:grid-cols-2 lg:grid-cols-4">
            {STEPS.map(({ number, title, body, icon: Icon, highlight }) => (
              <li key={title} className="flex flex-col items-center text-center">
                <span
                  className={cn(
                    "flex h-[68px] w-[68px] items-center justify-center rounded-full border",
                    highlight
                      ? "border-[#60A5FA]/70 bg-[#2563EB] shadow-[0_0_40px_rgba(37,99,235,0.55)]"
                      : "border-[#2A2D3D] bg-[#0E0F15]"
                  )}
                >
                  <Icon className={cn("h-[26px] w-[26px]", highlight ? "text-white" : "text-[#60A5FA]")} />
                </span>
                <span className="mt-5 font-mono text-xs text-[#60A5FA]">{number}</span>
                <p className="mt-1.5 text-xl font-semibold">{title}</p>
                <p className="mt-2 max-w-[230px] text-sm leading-relaxed text-[#8B90A0]">{body}</p>
              </li>
            ))}
          </ol>
        </div>
      </div>
    </section>
  )
}