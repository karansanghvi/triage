import CodeSnippet from "@/components/common/CodeSnippet"
import SectionHeading from "./SectionHeading"

const STEPS = [
  { title: "Create an account", body: "Free, with email and password." },
  { title: "Add your provider keys", body: "Encrypted the moment you save them." },
  { title: "Use your Triage key", body: "Point your client at Triage and set the model to auto." },
]

export default function Integrate() {
  return (
    <section id="integrate" className="relative scroll-mt-[calc(72px+env(safe-area-inset-top))] px-6 pb-[120px]">
      <div
        aria-hidden="true"
        className="pointer-events-none absolute right-[2%] top-10 h-[520px] w-[820px] rounded-full"
        style={{ background: "radial-gradient(closest-side, rgba(37,99,235,0.16), rgba(7,7,11,0))" }}
      />

      <div className="relative mx-auto flex max-w-6xl flex-wrap items-start gap-14">
        {/* Steps */}
        <div className="min-w-0 flex-[1_1_380px]">
          <SectionHeading
            eyebrow="Integrate"
            title="Two lines to switch."
            description="Triage speaks the OpenAI API. Keep your code, change where it points."
          />
          <ol className="mt-8 space-y-5">
            {STEPS.map((step, i) => (
              <li key={step.title} className="flex gap-4">
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-[10px] border border-[#2A2D3D] text-[13px] font-semibold text-[#93C5FD]">
                  {i + 1}
                </span>
                <div>
                  <p className="text-[15px] font-semibold">{step.title}</p>
                  <p className="mt-1 text-sm text-[#8B90A0]">{step.body}</p>
                </div>
              </li>
            ))}
          </ol>
        </div>

        {/* Code: shared with the Settings page */}
        <CodeSnippet className="min-w-0 flex-[1_1_520px] shadow-[0_30px_100px_rgba(37,99,235,0.12)]" />
      </div>
    </section>
  )
}