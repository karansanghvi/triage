import { Brain, Calculator, FileText, type LucideIcon } from "lucide-react"

type Suggestion = {
  label: string
  prompt: string
  icon: LucideIcon
  iconColor: string
}

// Three difficulty levels, so visitors can watch Triage pick different models
const suggestions: Suggestion[] = [
  {
    label: "Quick math",
    prompt: "What is 15% of 80?",
    icon: Calculator,
    iconColor: "text-success",
  },
  {
    label: "Explain a concept",
    prompt: "Explain how a hash map works in three sentences.",
    icon: FileText,
    iconColor: "text-primary",
  },
  {
    label: "Hard reasoning",
    prompt: "Prove that the square root of 2 is irrational.",
    icon: Brain,
    iconColor: "text-warning",
  },
]

type SuggestionChipsProps = {
  onPick: (prompt: string) => void
}

export default function SuggestionChips({ onPick }: SuggestionChipsProps) {
  return (
    <div className="mt-4 flex flex-wrap justify-center gap-2">
      {suggestions.map(({ label, prompt, icon: Icon, iconColor }) => (
        <button
          key={label}
          onClick={() => onPick(prompt)}
          className="flex items-center gap-2 rounded-full border border-border bg-card px-4 py-2 text-sm text-muted-foreground transition-colors hover:border-primary/40 hover:bg-accent hover:text-foreground"
        >
          <Icon className={`h-4 w-4 ${iconColor}`} />
          {label}
        </button>
      ))}
    </div>
  )
}