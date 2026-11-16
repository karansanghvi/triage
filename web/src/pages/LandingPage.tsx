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
  return (
    <>
      <div className="min-h-screen overflow-x-hidden bg-[#07070B] text-foreground">
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
