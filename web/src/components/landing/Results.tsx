import { Link } from "react-router-dom"
import { ArrowRight, TrendingUp } from "lucide-react"
import SectionHeading from "./SectionHeading"

export default function Results() {
  return (
    <section id="results" className="scroll-mt-[72px] px-6 pb-[120px]">
      <div className="mx-auto max-w-6xl">
        <div className="flex flex-wrap items-end justify-between gap-5">
          <div className="min-w-0 flex-[1_1_480px]">
            <SectionHeading
              eyebrow="Results"
              title="Measured, not promised."
              description="Benchmarked against always using one model, with an answer checker grading every response."
            />
          </div>
          <Link
            to="/dashboard"
            className="group inline-flex min-h-11 items-center gap-2 text-sm font-medium text-[#60A5FA] hover:text-[#93C5FD]"
          >
            See all results
            <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
          </Link>
        </div>

        {/* Placeholder until the real benchmark chart is ready */}
        <div className="mt-10 flex h-[380px] flex-col items-center justify-center gap-2.5 rounded-3xl border border-dashed border-[#2A2D3D] bg-[#0B0C12] p-6 text-center">
          <TrendingUp className="h-8 w-8 text-[#60A5FA]" />
          <p className="text-base font-semibold">[Cost vs quality chart from your benchmark]</p>
          <p className="text-[13px] text-[#8B90A0]">Triage's curve against always-cheap and always-strong baselines</p>
        </div>
      </div>
    </section>
  )
}