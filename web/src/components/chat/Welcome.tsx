import { Split } from "lucide-react"

type WelcomeProps = {
  name: string
}

export default function Welcome({ name }: WelcomeProps) {
  const firstName = name.trim().split(" ")[0]

  return (
    <div className="flex flex-col items-center text-center">
      <div className="relative mb-6">
        <div className="absolute inset-0 rounded-3xl bg-primary/40 blur-2xl" />
        <div className="relative flex h-16 w-16 items-center justify-center rounded-3xl bg-primary">
          <Split className="h-8 w-8 text-primary-foreground" />
        </div>
      </div>

      <h1 className="text-4xl font-semibold tracking-tight">
        {firstName ? `Hi ${firstName}, ask anything` : "Ask anything"}
      </h1>
      <p className="mt-3 text-sm text-muted-foreground">
        Triage picks the cheapest model that can handle your question.
      </p>
    </div>
  )
}