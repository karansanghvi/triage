import NameSection from "@/components/settings/NameSection"

export default function SettingsPage() {
  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Settings</h1>
        <p className="mt-1 text-sm text-muted-foreground">Your name and API keys.</p>
      </div>

      <NameSection />

      {/* API keys section comes next */}
    </div>
  )
}