import { Link } from "react-router-dom"
import { ArrowRight } from "lucide-react"
import CostQualityChart from "@/components/dashboard/CostQualityChart"
import { benchmark } from "@/lib/benchmark"
import { formatPercent } from "@/lib/format"
import SectionHeading from "./SectionHeading"

export default function Results() {
  const h = benchmark.headline

  return (
    <section id="results" className="scroll-mt-[calc(72px+env(safe-area-inset-top))] px-6 pb-[120px]">
      <div className="mx-auto max-w-6xl">
        <div className="flex flex-wrap items-end justify-between gap-5">
          <div className="min-w-0 flex-[1_1_480px]">
            <SectionHeading
              eyebrow="Results"
              title="Measured, not promised."
              description={`Keeps ${formatPercent(h.quality_kept)} of the strongest model's quality at ${formatPercent(h.cost_saved)} lower cost.`}
            />
          </div>
          <Link
            to="/dashboard/benchmark"
            className="group inline-flex min-h-11 items-center gap-2 text-sm font-medium text-[#60A5FA] hover:text-[#93C5FD]"
          >
            See all results
            <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
          </Link>
        </div>

        <div className="mt-10 min-w-0 rounded-3xl border border-[#1F2130] bg-[#0E0F15] p-5 sm:p-8">
          <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
            <h3 className="text-base font-semibold">Cost vs quality</h3>
            <p className="text-[13px] text-[#8B90A0]">
              {benchmark.split.n} held-out test questions · each point on the line is one quality setting
            </p>
          </div>
          <CostQualityChart benchmark={benchmark} />
        </div>
      </div>
    </section>
  )
}
