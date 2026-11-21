import benchmarkData from "@/content/benchmark.json"
import type { Benchmark } from "@/lib/dashboard"
import { formatPercent } from "@/lib/format"
import DashCard from "@/components/dashboard/DashCard"
import TierMixBar from "@/components/dashboard/TierMixBar"
import CostQualityChart from "@/components/dashboard/CostQualityChart"

const benchmark = benchmarkData as unknown as Benchmark

// The latest date in answers_judged, e.g. "Nov 2026"
const judgedDates = Object.values(benchmark.dataset.answers_judged)
  .filter((v) => /^\d{4}-\d{2}-\d{2}/.test(v))
  .sort()
const MONTH = judgedDates.length
  ? new Date(`${judgedDates[judgedDates.length - 1].slice(0, 10)}T00:00:00Z`).toLocaleDateString([], {
      month: "short",
      year: "numeric",
      timeZone: "UTC",
    })
  : "Nov 2026"

export default function BenchmarkPage() {
  const h = benchmark.headline
  const strongest = benchmark.references.always_strongest
  const n = benchmark.split.n

  return (
    <>
      {/* The result */}
      <section className="relative overflow-hidden rounded-[18px] border border-[#1F2130] bg-[#12131A] p-8">
        <div
          aria-hidden="true"
          className="pointer-events-none absolute -left-32 -top-40 h-[420px] w-[600px] max-w-full rounded-full"
          style={{ background: "radial-gradient(closest-side, rgba(37,99,235,0.2), rgba(18,19,26,0))" }}
        />
        <div className="relative flex flex-wrap items-center justify-between gap-7">
          <div className="min-w-0 flex-[1_1_420px]">
            <p className="text-[13px] font-semibold uppercase tracking-[0.08em] text-[#60A5FA]">
              The result · measured {MONTH} · {n} test questions
            </p>
            <h2 className="mt-3 text-[32px] font-bold leading-tight tracking-[-0.02em]">
              Keeps <span className="text-[#60A5FA]">{formatPercent(h.quality_kept)}</span> of the strongest model's
              quality at <span className="text-[#4ADE80]">{formatPercent(h.cost_saved)} lower cost</span>.
            </h2>
            <p className="mt-3 text-sm leading-relaxed text-[#A7ACBA]">
              On the default Balanced setting, Triage answered {h.correct} of {n} test questions correctly, against{" "}
              {strongest.correct ?? "—"} for always using the strongest model.
            </p>
          </div>
          <div className="flex gap-3">
            <div className="min-w-[150px] rounded-[14px] border border-[#1F2130] bg-[#0A0A0F] px-5 py-[18px]">
              <p className="text-xs text-muted-foreground">Quality kept</p>
              <p className="mt-2 text-[34px] font-bold">{formatPercent(h.quality_kept, 1)}</p>
            </div>
            <div className="min-w-[150px] rounded-[14px] border border-[#1F2130] bg-[#0A0A0F] px-5 py-[18px]">
              <p className="text-xs text-muted-foreground">Cost saved</p>
              <p className="mt-2 text-[34px] font-bold text-[#4ADE80]">{formatPercent(h.cost_saved, 1)}</p>
            </div>
          </div>
        </div>
      </section>

      {/* The chart */}
      <DashCard
        title="Cost vs quality"
        description="Each point on the line is one quality-dial setting. Further left is cheaper; higher keeps more of the strongest model's quality."
      >
        <CostQualityChart benchmark={benchmark} />
      </DashCard>

      {/* Tier mix */}
      <DashCard title="Where requests went" description="Tier mix on the Balanced setting.">
        <TierMixBar mix={benchmark.tier_mix_default} />
        {benchmark.tier_mix_default.low === 0 && (
          <p className="mt-4 text-[13px] leading-relaxed text-muted-foreground">
            On Balanced, no test question was confident enough for the cheapest model. The Cheapest setting uses it
            far more.
          </p>
        )}
      </DashCard>

      {/* How we measured */}
      {/* <section className="rounded-[18px] border border-[#1F2130] bg-[#0E0F15] px-6 py-[22px]">
        <h2 className="text-base font-semibold">How we measured</h2>
        <ul className="mt-3 grid gap-x-7 gap-y-2.5 text-[13px] leading-relaxed text-[#A7ACBA] md:grid-cols-2">
          <li>
            {benchmark.dataset.prompts} prompts, {benchmark.dataset.labeled} labeled by checking each model's answer.{" "}
            {benchmark.dataset.test_labeled} were held out as a test set.
          </li>
          {benchmark.caveats.map((caveat) => (
            <li key={caveat}>{caveat}</li>
          ))}
        </ul>
      </section> */}
    </>
  )
}