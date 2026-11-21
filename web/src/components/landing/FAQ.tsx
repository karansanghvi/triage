import { ChevronDown } from "lucide-react"
import SectionHeading from "./SectionHeading"

const FAQS = [
  {
    q: "Is Triage free?",
    a: "Yes. Triage itself doesn't charge anything. The free demo gives you 10 requests a day on our keys. With an account, requests use your own provider keys, so you pay your provider directly, and Triage helps you pay less.",
  },
  {
    q: "Do you store my prompts?",
    a: "No. Triage never stores prompt text, only a fingerprint of each conversation and routing details like the model, cost and timing. Answers are cached for up to 24 hours so a repeated question isn't paid for twice.",
  },
  {
    q: "How are my provider keys kept safe?",
    a: "They're encrypted with AES-256-GCM before they're saved, never written to logs, and only the last 4 characters are ever shown, even to you.",
  },
  {
    q: "What if Triage picks a model that's too weak?",
    a: "It only picks a cheaper model when it predicts that model will answer well enough for your quality setting. When it's unsure, it plays safe and uses a stronger one. You can also raise your quality bar, or name a specific model in any request to skip routing.",
  },
  {
    q: "Which providers does it work with?",
    a: "OpenAI, Anthropic, Groq and Google Gemini. Save keys for any of them and choose a preferred provider. If one is slow or down, Triage falls back to another automatically.",
  },
  {
    q: "Do I have to change my code?",
    a: "Only two lines. Triage speaks the OpenAI API, so you point your existing OpenAI client at Triage, use your Triage API key, and set the model to auto. Streaming works too.",
  },
]

export default function FAQ() {
  return (
    <section id="faq" className="scroll-mt-20 px-6 pb-[120px]">
      <div className="mx-auto max-w-3xl">
        <SectionHeading eyebrow="FAQ" title="Questions, answered." />
        <div className="mt-10 divide-y divide-[#1A1C26] rounded-[20px] border border-[#1F2130] bg-[#0E0F15]">
          {FAQS.map(({ q, a }) => (
            <details key={q} className="group px-6">
              <summary className="flex min-h-[64px] cursor-pointer list-none items-center justify-between gap-4 py-4 text-[15px] font-medium [&::-webkit-details-marker]:hidden">
                {q}
                <ChevronDown
                  aria-hidden="true"
                  className="h-4 w-4 shrink-0 text-[#8B90A0] transition-transform group-open:rotate-180"
                />
              </summary>
              <p className="pb-5 text-sm leading-relaxed text-[#A7ACBA]">{a}</p>
            </details>
          ))}
        </div>
      </div>
    </section>
  )
}