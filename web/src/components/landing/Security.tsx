import { Check, Lock } from "lucide-react"

const PROMISES = ["Encrypted at rest", "Never logged", "Only the last 4 characters are ever shown"]

const KEY_LIFE = [
  { label: "You paste", value: "sk-ant-api03-7Hq2Lm…Xk9p", highlight: false },
  { label: "Stored", value: "gAAAAABn4fK9x2…Q9wE=", highlight: true },
  { label: "You see", value: "••••••••Xk9p", highlight: false },
]

export default function Security() {
  return (
    <section className="relative px-6 pb-[120px]">
      <div
        aria-hidden="true"
        className="pointer-events-none absolute -left-[140px] -top-[60px] h-[480px] w-[760px] rounded-full"
        style={{ background: "radial-gradient(closest-side, rgba(37,99,235,0.16), rgba(7,7,11,0))" }}
      />

      <div className="relative mx-auto flex max-w-6xl flex-wrap items-center gap-10 rounded-[28px] border border-[#1F2130] bg-[#0B0C12] p-6 sm:p-10">
        {/* Message */}
        <div className="min-w-0 flex-[1_1_340px]">
          <p className="text-[13px] font-semibold uppercase tracking-[0.1em] text-[#60A5FA]">Security</p>
          <h2 className="mt-3 text-4xl font-bold leading-[1.15] tracking-[-0.02em]">Your keys are yours.</h2>
          <ul className="mt-[22px] space-y-3">
            {PROMISES.map((promise) => (
              <li key={promise} className="flex items-center gap-2.5 text-[15px] text-[#C9CDD8]">
                <Check className="h-[18px] w-[18px] shrink-0 text-[#60A5FA]" strokeWidth={2.5} />
                {promise}
              </li>
            ))}
          </ul>
        </div>

        {/* The life of a key */}
        <div className="flex min-w-0 flex-[1_1_440px] flex-col gap-2.5">
          {KEY_LIFE.map((row) => (
            <div
              key={row.label}
              className={
                row.highlight
                  ? "flex items-center justify-between gap-4 rounded-[14px] border border-[#60A5FA]/50 bg-[#2563EB]/10 px-5 py-4 shadow-[0_0_40px_rgba(37,99,235,0.2)]"
                  : "flex items-center justify-between gap-4 rounded-[14px] border border-[#1F2130] bg-[#0E0F15] px-5 py-4"
              }
            >
              <span
                className={
                  row.highlight
                    ? "flex shrink-0 items-center gap-2 text-xs font-semibold uppercase tracking-[0.08em] text-[#93C5FD]"
                    : "shrink-0 text-xs font-semibold uppercase tracking-[0.08em] text-[#8B90A0]"
                }
              >
                {row.highlight && <Lock className="h-3.5 w-3.5" strokeWidth={2.2} />}
                {row.label}
              </span>
              <span
                className={
                  row.highlight
                    ? "min-w-0 truncate font-mono text-sm text-[#A7ACBA]"
                    : "min-w-0 truncate font-mono text-sm tracking-[0.04em]"
                }
              >
                {row.value}
              </span>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}