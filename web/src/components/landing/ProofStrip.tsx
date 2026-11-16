// Placeholder numbers until the real benchmark results are in
const STATS = [
  { value: "[X]%", label: ["cheaper than always using", "the most powerful model"] },
  { value: "[Y]%", label: ["of answers judged", "just as good"] },
  { value: "2 lines", label: ["of code to switch", "your app over"] },
]

const PROVIDERS = ["OpenAI", "Anthropic", "Gemini", "Groq"]

export default function ProofStrip() {
  return (
    <>
      <section className="px-6 pb-24">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-8 border-y border-[#1A1C26] py-9">
            <div className="flex flex-wrap gap-14">
                {STATS.map((stat) => (
                    <div key={stat.value}>
                        <p className="text-4xl font-bold tracking-[-0.02em]">{stat.value}</p>
                        <p className="mt-1 text-[13px] leading-snug text-[#8B90A0]">
                            {stat.label[0]}
                            <br />
                            {stat.label[1]}
                        </p>
                    </div>
                ))}
            </div>

            <div>
                <p className="mb-3 text-xs uppercase tracking-[0.08em] text-[#8B90A0]">Works with your keys for</p>
                <div className="flex flex-wrap gap-2">
                    {PROVIDERS.map((name) => (
                        <span key={name} className="rounded-full border border-[#1F2130] px-3.5 py-2 text-[13px] text-[#C9CDD8]">
                            {name}
                        </span>
                    ))}
                </div>
            </div>
        </div>
      </section>
    </>
  )
}
