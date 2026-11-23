import { useEffect, useState, type FormEvent } from "react"
import { Link, useLocation, useNavigate } from "react-router-dom"
import { ArrowRight } from "lucide-react"
import AuthLayout from "@/components/auth/AuthLayout"
import { FormError, PasswordInput, fieldBorder, inputClass, primaryButton, secondaryButton } from "@/components/auth/fields"
import { useAuth } from "@/context/AuthContext"
import { TriageError } from "@/lib/errors"
import { minutesFrom } from "@/lib/time"
import { cn } from "@/lib/utils"

type FieldErrors = { email?: string; password?: string }

function validate(email: string, password: string): FieldErrors {
  const errors: FieldErrors = {}
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim())) errors.email = "Enter a valid email address."
  if (!password) errors.password = "Enter your password."
  return errors
}

export default function LoginPage() {
  const { user, login } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()

  // Where to go after logging in: the page that sent us here, or the chat
  const from = (location.state as { from?: string } | null)?.from ?? "/chat"

  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [fieldErrors, setFieldErrors] = useState<FieldErrors>({})
  const [formError, setFormError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  useEffect(() => {
    if (user) navigate(from, { replace: true })
  }, [user, navigate, from])

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setFormError(null)

    const errors = validate(email, password)
    setFieldErrors(errors)
    if (errors.email || errors.password) return

    setSubmitting(true)
    try {
      await login(email.trim(), password)
      navigate(from, { replace: true })
    } catch (err) {
      if (err instanceof TriageError && err.code === "invalid_credentials") {
        setFormError("Incorrect email or password.")
      } else if (err instanceof TriageError && err.retryAfter) {
        setFormError(`Too many attempts. Try again in ${minutesFrom(err.retryAfter)}.`)
      } else if (err instanceof TriageError) {
        setFormError(err.message)
      } else {
        setFormError("Something went wrong. Please try again.")
      }
      setPassword("")
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <AuthLayout
      title="Welcome back"
      subtitle="Log in to manage your keys and see your usage."
      footer={
        <p className="text-sm text-[#A7ACBA]">
          New to Triage?{" "}
          <Link to="/signup" className="font-medium text-[#60A5FA] hover:text-[#93C5FD]">
            Create an account
          </Link>
        </p>
      }
    >
      <form onSubmit={handleSubmit} noValidate className="mt-7 flex flex-col gap-[18px]">
        <div>
          <label htmlFor="li-email" className="block text-sm font-medium">
            Email
          </label>
          <input
            id="li-email"
            type="email"
            autoComplete="email"
            placeholder="you@company.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            aria-invalid={!!fieldErrors.email}
            aria-describedby="li-email-error"
            className={cn(inputClass, fieldBorder(!!fieldErrors.email))}
          />
          {fieldErrors.email && (
            <p id="li-email-error" className="mt-2 text-xs text-[#FCA5A5]">
              {fieldErrors.email}
            </p>
          )}
        </div>

        <div>
          <div className="flex items-center justify-between gap-3">
            <label htmlFor="li-password" className="text-sm font-medium">
              Password
            </label>
            <Link to="/forgot-password" className="inline-flex items-center text-[13px] font-medium text-[#60A5FA] hover:text-[#93C5FD] max-md:min-h-11 pointer-coarse:min-h-11">
              Forgot password?
            </Link>
          </div>
          <PasswordInput
            id="li-password"
            autoComplete="current-password"
            placeholder="Your password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            invalid={!!fieldErrors.password}
            aria-invalid={!!fieldErrors.password}
            aria-describedby="li-password-error"
          />
          {fieldErrors.password && (
            <p id="li-password-error" className="mt-2 text-xs text-[#FCA5A5]">
              {fieldErrors.password}
            </p>
          )}
        </div>

        {formError && <FormError message={formError} />}

        <button type="submit" disabled={submitting} className={primaryButton}>
          {submitting ? "Logging in…" : "Log in"}
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