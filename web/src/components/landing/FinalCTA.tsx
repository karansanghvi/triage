import { Link } from "react-router-dom"

export default function FinalCTA() {
  return (
    <section className="relative overflow-hidden px-6 pb-[120px] pt-10">
      <div
        aria-hidden="true"
        className="pointer-events-none absolute left-1/2 top-0 h-[500px] w-[900px] -translate-x-1/2 rounded-full"
        style={{ background: "radial-gradient(closest-side, rgba(37,99,235,0.28), rgba(7,7,11,0))" }}
      />

      <div className="relative mx-auto max-w-[760px] text-center">
        <h2 className="text-4xl font-bold leading-[1.1] tracking-[-0.03em] sm:text-[52px]">Start routing smarter.</h2>
        <p className="mx-auto mt-[18px] max-w-[520px] text-[17px] leading-relaxed text-[#A7ACBA]">
          Try it in seconds with no signup, or create an account and connect your app today.
        </p>
        <div className="mt-9 flex flex-wrap justify-center gap-3">
          <Link
            to="/chat"
            className="inline-flex min-h-[52px] items-center rounded-[14px] bg-[#2563EB] px-[26px] text-base font-semibold text-white shadow-[0_10px_40px_rgba(37,99,235,0.4)] transition-colors hover:bg-[#1D4ED8]"
          >
            Try the demo
          </Link>
          <Link
            to="/signup"
            className="inline-flex min-h-[52px] items-center rounded-[14px] border border-[#2A2D3D] px-[26px] text-base font-medium transition-colors hover:border-[#3B4058] hover:bg-[#12131A]"
          >
            Create an account
          </Link>
        </div>
      </div>
    </section>
  )
}