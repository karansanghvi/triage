import SectionHeading from "./SectionHeading"

const PROBLEMS = [
  { title: "Flagship prices for simple questions", body: "A quick lookup costs as much as a hard proof." },
  { title: "Cheap models fail quietly", body: "Hard questions get weak answers, and nobody notices." },
  { title: "Picking by hand doesn't scale", body: "Rules break as models and prices change." },
]

export default function Problem() {
  return (
    <section className="px-6 pb-[120px]">
      <div className="mx-auto max-w-6xl">
        <SectionHeading eyebrow="The problem" title="Most prompts don't need your most expensive model." />

        <div className="mt-12 grid grid-cols-[repeat(auto-fit,minmax(260px,1fr))] gap-5">
          {PROBLEMS.map((problem) => (
            <div key={problem.title} className="rounded-[20px] border border-[#1A1C26] bg-[#0E0F15] p-7">
              <p className="text-lg font-semibold">{problem.title}</p>
              <p className="mt-2.5 text-sm leading-[1.7] text-[#8B90A0]">{problem.body}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}