import { useVisitorName } from "@/hooks/useVisitorName";
import { useState } from "react";


export default function NameSection() {

    const { name, setName } = useVisitorName()
    const [draft, setDraft] = useState(name)
    const [saved, setSaved] = useState(false)

    function handleSave() {
        setName(draft)
        setSaved(true)
        setTimeout(() => setSaved(false), 2000)
    }

  return (
    <>
      <section className="rounded-xl border border-border bg-card p-6">
        <h2 className="text-base font-semibold">Your Name</h2>

        <div className="mt-4 flex gap-3">
            <input
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSave()}
                placeholder="eg: John Doe"
                className="flex-1 rounded-lg border border-input bg-background px-3 py-2 text-sm outline-none focus:border-primary"
            />
            <button
                onClick={handleSave}
                disabled={!draft.trim()}
                className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90"
            >
                {saved ? "Saved" : "Save"}
            </button>
        </div>
      </section>
    </>
  )
}
