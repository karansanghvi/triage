import SectionHeading from "./SectionHeading"

const PROMPTS = [
  { text: "What's 15% of 80?", tier: "Cheapest", color: "#22C55E" },
  { text: "Summarize this support email", tier: "Mid-range", color: "#3B82F6" },
  { text: "Find the bug in this function", tier: "Most powerful", color: "#F59E0B" },
]

function Row({ text, tier, color }: { text: string; tier: string; color: string }) {
  return (
    <li className="flex items-center justify-between gap-3 rounded-xl border border-[#1A1C26] bg-[#0A0A0F] px-3.5 py-2.5">
      <span className="min-w-0 truncate text-[13px] text-[#C9CDD8]">{text}</span>
      <span className="flex shrink-0 items-center gap-1.5 text-xs font-medium" style={{ color }}>
        <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full" style={{ background: color }} />
        {tier}
      </span>
    </li>
  )
}

export default function Problem() {
  return (
    <section className="px-6 pb-[120px]">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-12">
        <div className="min-w-0 flex-[1_1_440px]">
          <SectionHeading eyebrow="What is Triage" title="Most prompts don't need your most expensive model." />
          <p className="mt-6 max-w-2xl text-[17px] leading-relaxed text-[#A7ACBA]">
            Triage sits between your app and your AI providers. For every prompt, it predicts which model can answer it
            well and sends it to the cheapest one that clears your quality bar, falling back automatically if a provider
            fails. Every answer shows which model replied and what you saved, and since it speaks the OpenAI API,
            switching takes two lines.
          </p>
        </div>

        {/* Before vs after: an illustration, not measured data */}
        <div className="min-w-0 flex-[1_1_360px] rounded-[20px] border border-[#1F2130] bg-[#0E0F15] p-6">
          <p className="text-xs font-semibold uppercase tracking-[0.08em] text-[#8B90A0]">Without Triage</p>
          <ul className="mt-3 space-y-2">
            {PROMPTS.map((p) => (
              <Row key={p.text} text={p.text} tier="Most powerful" color="#F59E0B" />
            ))}
          </ul>

          <p className="mt-6 text-xs font-semibold uppercase tracking-[0.08em] text-[#60A5FA]">With Triage</p>
          <ul className="mt-3 space-y-2">
            {PROMPTS.map((p) => (
              <Row key={p.text} {...p} />
            ))}
          </ul>
        </div>
      </div>
    </section>
  )
}