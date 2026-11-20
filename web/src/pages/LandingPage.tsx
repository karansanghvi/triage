import { useEffect } from "react"
import Features from "@/components/landing/Features";
import FinalCTA from "@/components/landing/FinalCTA";
import Footer from "@/components/landing/Footer";
import Hero from "@/components/landing/Hero";
import HowItWorks from "@/components/landing/HowItWorks";
import Integrate from "@/components/landing/Integrate";
import LandingNav from "@/components/landing/LandingNav";
import Problem from "@/components/landing/Problem";
import ProofStrip from "@/components/landing/ProofStrip";
import QualityDialDemo from "@/components/landing/QualityDialDemo";
import Results from "@/components/landing/Results";
import Security from "@/components/landing/Security";

export default function LandingPage() {
  // Arriving from another page at /#how etc.: the browser looks for the section before React has drawn it, so
  // jump there once it exists (scroll-margin keeps it clear of the sticky nav)
  useEffect(() => {
    const id = decodeURIComponent(window.location.hash.slice(1))
    if (id) document.getElementById(id)?.scrollIntoView()
  }, [])

  return (
    <>
      {/* overflow-x-clip, not hidden: hidden would make this a scroll container and stop the nav from sticking */}
      <div className="min-h-screen overflow-x-clip bg-[#07070B] text-foreground">
        <LandingNav />
        <main>
            <Hero />
            <ProofStrip />
            <Problem />
            <HowItWorks />
            <QualityDialDemo />
            <Integrate />
            <Features />
            <Results />
            <Security />
            <FinalCTA />
        </main>
        <Footer />
      </div>
    </>
  )
}
