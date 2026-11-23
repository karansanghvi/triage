import { useEffect, useState, type FormEvent } from "react"
import { Link, useNavigate } from "react-router-dom"
import { ArrowRight } from "lucide-react"
import AuthLayout from "@/components/auth/AuthLayout"
import { FormError, PasswordInput, fieldBorder, inputClass, primaryButton, secondaryButton } from "@/components/auth/fields"
import { useAuth } from "@/context/AuthContext"
import { TriageError } from "@/lib/errors"
import { PASSWORD_MAX, PASSWORD_MIN, passwordLengthProblem } from "@/lib/passwordRules"
import { cn } from "@/lib/utils"

type FieldErrors = { name?: string; email?: string; password?: string }

function validate(name: string, email: string, password: string): FieldErrors {
  const errors: FieldErrors = {}
  if (!name.trim()) errors.name = "Enter your name."
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim())) errors.email = "Enter a valid email address."
  const lengthProblem = passwordLengthProblem(password)
  if (lengthProblem === "short") errors.password = `Use at least ${PASSWORD_MIN} characters.`
  else if (lengthProblem === "long") errors.password = `Use ${PASSWORD_MAX} characters or fewer.`
  return errors
}

export default function SignupPage() {
  const { user, signup } = useAuth()
  const navigate = useNavigate()

  const [name, setName] = useState("")
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [fieldErrors, setFieldErrors] = useState<FieldErrors>({})
  const [formError, setFormError] = useState<string | null>(null)
  const [emailTaken, setEmailTaken] = useState(false)
  const [submitting, setSubmitting] = useState(false)

  // Already logged in: nothing to do here
  useEffect(() => {
    if (user) navigate("/settings", { replace: true })
  }, [user, navigate])

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setFormError(null)
    setEmailTaken(false)

    const errors = validate(name, email, password)
    setFieldErrors(errors)
    if (errors.name || errors.email || errors.password) return

    setSubmitting(true)
    try {
      await signup(email.trim(), password, name.trim())
      navigate("/settings", { replace: true })
    } catch (err) {
      if (err instanceof TriageError && err.code === "email_taken") {
        setEmailTaken(true)
      } else if (err instanceof TriageError) {
        setFormError(err.message)
      } else {
        setFormError("Something went wrong. Please try again.")
      }
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <AuthLayout
      title="Create your account"
      subtitle="Start routing in minutes. No credit card needed."
      footer={
        <>
          <p className="text-sm text-[#A7ACBA]">
            Already have an account?{" "}
            <Link to="/login" className="font-medium text-[#60A5FA] hover:text-[#93C5FD]">
              Log in
            </Link>
          </p>
          <p className="mt-2.5 text-xs text-[#8B90A0]">
            By creating an account you agree to our{" "}
            <Link to="/privacy" className="text-[#60A5FA] hover:text-[#93C5FD]">
              Privacy policy
            </Link>
            .
          </p>
        </>
      }
    >
      <form onSubmit={handleSubmit} noValidate className="mt-7 flex flex-col gap-[18px]">
        <div>
          <label htmlFor="su-name" className="block text-sm font-medium">
            Name
          </label>
          <input
            id="su-name"
            type="text"
            autoComplete="name"
            placeholder="John Doe"
            maxLength={80}
            value={name}
            onChange={(e) => setName(e.target.value)}
            aria-invalid={!!fieldErrors.name}
            aria-describedby="su-name-error"
            className={cn(inputClass, fieldBorder(!!fieldErrors.name))}
          />
          {fieldErrors.name && (
            <p id="su-name-error" className="mt-2 text-xs text-[#FCA5A5]">
              {fieldErrors.name}
            </p>
          )}
        </div>

        <div>
          <label htmlFor="su-email" className="block text-sm font-medium">
            Email
          </label>
          <input
            id="su-email"
            type="email"
            autoComplete="email"
            placeholder="you@company.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            aria-invalid={!!fieldErrors.email || emailTaken}
            aria-describedby="su-email-error"
            className={cn(inputClass, fieldBorder(!!fieldErrors.email || emailTaken))}
          />
          <div id="su-email-error">
            {fieldErrors.email && <p className="mt-2 text-xs text-[#FCA5A5]">{fieldErrors.email}</p>}
            {emailTaken && (
              <p className="mt-2 text-xs text-[#FCA5A5]">
                An account with this email already exists.{" "}
                <Link to="/login" className="font-medium text-[#60A5FA] hover:text-[#93C5FD]">
                  Log in instead
                </Link>
              </p>
            )}
          </div>
        </div>

        <div>
          <label htmlFor="su-password" className="block text-sm font-medium">
            Password
          </label>
          <PasswordInput
            id="su-password"
            autoComplete="new-password"
            placeholder="At least 10 characters"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            invalid={!!fieldErrors.password}
            aria-invalid={!!fieldErrors.password}
            aria-describedby="su-password-hint"
          />
          <p id="su-password-hint" className={cn("mt-2 text-xs", fieldErrors.password ? "text-[#FCA5A5]" : "text-[#8B90A0]")}>
            {fieldErrors.password ?? "Use 10 or more characters. A short phrase is easiest to remember."}
          </p>
        </div>

        {formError && <FormError message={formError} />}

        <button type="submit" disabled={submitting} className={cn(primaryButton, "mt-1.5")}>
          {submitting ? "Creating account…" : "Create account"}
        </button>
      </form>

      <div className="my-[22px] flex items-center gap-3 text-xs text-[#8B90A0]">
        <span className="h-px flex-1 bg-[#1F2130]" />
        or
        <span className="h-px flex-1 bg-[#1F2130]" />
      </div>

      <Link to="/chat" className={secondaryButton}>
        Try the demo without an account
        <ArrowRight className="h-4 w-4" />
      </Link>
    </AuthLayout>
  )
}