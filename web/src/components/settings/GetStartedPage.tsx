import { useAccountApi } from "@/hooks/useAccountApi";
import SettingsCard from "./SettingsCard";
import { useEffect, useState } from "react";
import { ArrowRight, Check, Loader2 } from "lucide-react";
import { TRIAGE_API_BASE } from "@/lib/triageUrl";
import CopyButton from "../common/CopyButton";
import CodeSnippet from "../common/CodeSnippet";
import { cn } from "@/lib/utils";
import { Link } from "react-router-dom";

type Progress = { hasProviderKey: boolean, hasApiKey: boolean }

export default function GetStartedPage() {

    const api = useAccountApi()
    const [progress, setProgress] = useState<Progress | null>(null)
    const [loadError, setLoadError] = useState(false)

    // Ask the backend what's already set up
    useEffect(() => {
        let cancelled = false
        Promise.all([api.listProviderKeys(), api.listApiKeys()])
            .then(([providerKeys, apiKeys]) => {
                if (cancelled) return
                setProgress({
                    hasProviderKey: providerKeys.some((k) => k.status !== "invalid"),
                    hasApiKey: apiKeys.some((k) => !k.revoked_at),
                })
            })
            .catch(() => {
                if (!cancelled) setLoadError(true)
            })
        return () => {
            cancelled = true
        }
    }, [api])

    const allDone = progress?.hasProviderKey && progress?.hasApiKey

  return (
    <>
      {/* Checklist */}
      <SettingsCard 
        title={allDone ? "You're all set" : "Get started"}
        description={
            allDone
                ? "Your app can call Triage now. Use the details below."
                : "Two quick steps, then your app can call Triage."
        }
      >
        <div className="px-6 py-5">
            {loadError ? (
                <p className="text-sm text-[#FCA5A5]">Couldn't load your step. Refresh the page to try again.</p>
            ) : !progress ? (
                <p className="flex items-center gap-2 text-sm text-muted-foreground">
                    <Loader2 className="h-4 w-4 animate-spin" /> Checking your setup...
                </p>
            ) : (
                <ol className="space-y-3">
                    <ChecklistItem
                        step={1}
                        done={progress.hasProviderKey}
                        title="Add a provider key"
                        body="Your OpenAI, Anthropic, Groq or Gemini key. Triage uses it to answer your requests."
                        to="/settings/provider-keys"
                        cta="Add a key"
                    />
                    <ChecklistItem
                        step={2}
                        done={progress.hasApiKey}
                        title="Create a Triage API key"
                        body="Your app sends this key to Triage instead of your provider keys."
                        to="/settings/api-keys"
                        cta="Create a key"
                    />
                </ol>
            )}
        </div>
      </SettingsCard>

      <SettingsCard
        title="Connect your app"
        description={
          <>
            Triage speaks the OpenAI API. Point your client here and set the model to{" "}
            <code className="font-mono text-[#C9CDD8]">auto</code>.
          </>
        }
      >
        <div className="flex flex-wrap gap-4 px-6 pt-5">
            <div className="min-w-0 flex-[1_1_320px]">
                <p className="text-xs font-medium uppercase tracking-[0.08em] text-muted-foreground">Base URL</p>
                <div className="mt-2 flex gap-2">
                    <code className="flex h-11 min-w-0 flex-1 items-center overflow-hidden text-ellipsis whitespace-nowrap rounded-xl border border-[#1F2130] bg-[#0A0A0F] px-3.5 font-mono text-sm text-[#C9CDD8]">
                        {TRIAGE_API_BASE}
                    </code>
                </div>
            </div>
            <div className="min-w-0 flex-[0_1_200px]">
            <p className="text-xs font-medium uppercase tracking-[0.08em] text-muted-foreground">Model</p>
            <div className="mt-2 flex gap-2">
              <code className="flex h-11 min-w-0 flex-1 items-center rounded-xl border border-[#1F2130] bg-[#0A0A0F] px-3.5 font-mono text-sm text-[#C9CDD8]">
                auto
              </code>
              <CopyButton text="auto" variant="icon" ariaLabel="Copy model name" />
            </div>
          </div>
        </div>
        <div className="p-6">
            <CodeSnippet />
        </div>
      </SettingsCard>
    </>
  )
}

type ChecklistItemProps = {
  step: number
  done: boolean
  title: string
  body: string
  to: string
  cta: string
}

function ChecklistItem({ step, done, title, body, to, cta }: ChecklistItemProps) {
    return (
        <>
            <li className="flex flex-wrap items-center gap-4 rounded-xl border border-[#1F2130] bg-[#0E0F15] px-4 py-3.5">
                <span
                    className={cn(
                        "flex h-7 w-7 shrink-0 items-center justify-center rounded-full border text-[13px] font-semibold",
                        done ? "border-[#22C55E]/40 bg-[#22C55E]/15 text-[#4ADE80]" : "border-[#2A2D3D] bg-[#1A1C26] text-[#93C5FD]"
                    )}
                    aria-hidden="true"
                >
                    {done ? <Check className="h-4 w-4" strokeWidth={2.5} /> : step}
                </span>
                <div className="min-w-0 flex-[1_1_240px]">
                    <p className={cn("text-sm font-medium", done && "text-muted-foreground line-through decoration-[#4ADE80]/60")}>
                        {title}
                    <span className="sr-only">{done ? " (done)" : " (not done yet)"}</span>
                    </p>
                    <p className="mt-0.5 text-[13px] text-muted-foreground">{body}</p>
                </div>
                {!done && (
                    <Link
                        to={to}
                        className="group inline-flex h-9 w-[136px] shrink-0 items-center justify-center gap-1.5 rounded-[10px] bg-[#2563EB] px-3.5 text-[13px] font-semibold text-white transition-colors hover:bg-[#1D4ED8]"
                    >
                        {cta}
                        <ArrowRight className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5" />
                    </Link>
                )}
            </li>
        </>
    )
}