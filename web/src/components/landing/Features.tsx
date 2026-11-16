import { Database, Eye, KeyRound, Layers, LifeBuoy, TrendingUp, type LucideIcon } from "lucide-react"
import SectionHeading from "./SectionHeading"

const FEATURES: { title: string; body: string; icon: LucideIcon }[] = [
  {
    title: "Your keys, encrypted",
    body: "Bring keys for four providers. Stored encrypted, never logged, never shown again in full.",
    icon: KeyRound,
  },
  {
    title: "Preferred provider",
    body: "Put OpenAI or Anthropic first. Triage still picks the right size of model.",
    icon: Layers,
  },
  {
    title: "Automatic fallbacks",
    body: "If a model fails, the next one in line answers. Your users never see the outage.",
    icon: LifeBuoy,
  },
  { title: "Caching", body: "Identical requests are answered from cache, for free.", icon: Database },
  {
    title: "Every decision explained",
    body: "Each response says which model answered, why, what it cost and what you saved.",
    icon: Eye,
  },
  { title: "Usage dashboard", body: "Costs, savings, model mix and reliability for your own traffic.", icon: TrendingUp },
]

export default function Features() {
  return (
    <section className="px-6 pb-[120px]">
      <div className="mx-auto max-w-6xl">
        <SectionHeading eyebrow="Built in" title="Everything a router should do." />

        <div className="mt-12 grid grid-cols-1 gap-px overflow-hidden rounded-[20px] border border-[#1A1C26] bg-[#1A1C26] sm:grid-cols-2 lg:grid-cols-3">
          {FEATURES.map(({ title, body, icon: Icon }) => (
            <div key={title} className="bg-[#0B0C12] p-8">
              <Icon className="h-[22px] w-[22px] text-[#60A5FA]" />
              <p className="mt-4 text-base font-semibold">{title}</p>
              <p className="mt-2 text-sm leading-[1.7] text-[#8B90A0]">{body}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}