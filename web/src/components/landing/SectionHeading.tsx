type SectionHeadingProps = {
  eyebrow: string
  title: string
  description?: string
}

export default function SectionHeading({ eyebrow, title, description }: SectionHeadingProps) {
  return (
    <div>
      <p className="text-[13px] font-semibold uppercase tracking-[0.1em] text-[#60A5FA]">{eyebrow}</p>
      <h2 className="mt-3.5 max-w-[760px] text-4xl font-bold leading-[1.15] tracking-[-0.02em] sm:text-[44px]">
        {title}
      </h2>
      {description && <p className="mt-[18px] max-w-[520px] text-base leading-[1.7] text-[#A7ACBA]">{description}</p>}
    </div>
  )
}