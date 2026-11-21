import { useEffect, useState, type FormEvent } from "react"
import { Link, useNavigate } from "react-router-dom"
import AuthLayout from "@/components/auth/AuthLayout"
import CodeInput from "@/components/auth/CodeInput"
import { FormError, PasswordInput, fieldBorder, inputClass, primaryButton } from "@/components/auth/fields"
import { useAuth } from "@/context/AuthContext"
import { requestPasswordReset, verifyResetCode } from "@/lib/auth"
import { TriageError } from "@/lib/errors"
import { cn } from "@/lib/utils"

type Step = "email" | "code" | "password"

const RESEND_SECONDS = 60

function messageFor(err: unknown, fallback: string): string {
  if (!(err instanceof TriageError)) return fallback
  if (err.code === "email_unavailable") return "Password reset isn't available right now. Please try again later."
  if (err.retryAfter) {
    const minutes = Math.ceil(err.retryAfter / 60)
    return `Too many attempts. Try again in ${minutes === 1 ? "1 minute" : `${minutes} minutes`}.`
  }
  return err.message || fallback
}

const backToLogin = (
  <Link to="/login" className="inline-flex items-center text-sm font-medium text-[#60A5FA] hover:text-[#93C5FD] max-md:min-h-11 pointer-coarse:min-h-11">
    Back to log in
  </Link>
)

export default function ForgotPasswordPage() {
  const { resetPassword } = useAuth()
  const navigate = useNavigate()

  const [step, setStep] = useState<Step>("email")
  const [email, setEmail] = useState("")
  const [code, setCode] = useState("")
  const [resetToken, setResetToken] = useState("")
  const [password, setPassword] = useState("")
  const [confirm, setConfirm] = useState("")

  const [error, setError] = useState<string | null>(null)
  const [fieldError, setFieldError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [secondsLeft, setSecondsLeft] = useState(0)

  // Count down until "Resend code" is allowed again
  useEffect(() => {
    if (secondsLeft <= 0) return
    const timer = setTimeout(() => setSecondsLeft((s) => s - 1), 1000)
    return () => clearTimeout(timer)
  }, [secondsLeft])

  /* ---------- Step 1: email ---------- */

  async function handleSendCode(e: FormEvent) {
    e.preventDefault()
    setError(null)
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim())) {
      setFieldError("Enter a valid email address.")
      return
    }
    setFieldError(null)
    setSubmitting(true)
    try {
      await requestPasswordReset(email.trim())
      setStep("code")
      setCode("")
      setSecondsLeft(RESEND_SECONDS)
    } catch (err) {
      setError(messageFor(err, "Couldn't send the code. Please try again."))
    } finally {
      setSubmitting(false)
    }
  }

  async function handleResend() {
    if (secondsLeft > 0) return
    setError(null)
    try {
      await requestPasswordReset(email.trim())
      setSecondsLeft(RESEND_SECONDS)
      setCode("")
    } catch (err) {
      setError(messageFor(err, "Couldn't resend the code. Please try again."))
    }
  }

  /* ---------- Step 2: code ---------- */

  async function handleVerify(e: FormEvent) {
    e.preventDefault()
    setError(null)
    if (code.length !== 6) {
      setFieldError("Enter all 6 digits.")
      return
    }
    setFieldError(null)
    setSubmitting(true)
    try {
      const res = await verifyResetCode(email.trim(), code)
      setResetToken(res.reset_token)
      setStep("password")
    } catch (err) {
      if (err instanceof TriageError && err.code === "invalid_code") {
        setError("That code isn't right or has expired. Check the email, or request a new code.")
      } else {
        setError(messageFor(err, "Couldn't check the code. Please try again."))
      }
      setCode("")
    } finally {
      setSubmitting(false)
    }
  }

  /* ---------- Step 3: new password ---------- */

  async function handleSetPassword(e: FormEvent) {
    e.preventDefault()
    setError(null)
    if (password.length < 10 || password.length > 128) {
      setFieldError("Use 10 to 128 characters.")
      return
    }
    if (password !== confirm) {
      setFieldError("The two passwords don't match.")
      return
    }
    setFieldError(null)
    setSubmitting(true)
    try {
      await resetPassword(resetToken, password)
      navigate("/chat", { replace: true })
    } catch (err) {
      if (err instanceof TriageError && err.code === "invalid_reset_token") {
        setStep("email")
        setCode("")
        setPassword("")
        setConfirm("")
        setError("Your reset expired. Enter your email to get a new code.")
      } else {
        setError(
          messageFor(err, "Couldn't update your password. Your reset may have expired, so please start again.")
        )
      }
    } finally {
      setSubmitting(false)
    }
  }

  /* ---------- Screens ---------- */

  if (step === "email") {
    return (
      <AuthLayout
        title="Reset your password"
        subtitle="Enter your account's email and we'll send you a 6-digit code."
        footer={backToLogin}
      >
        <form onSubmit={handleSendCode} noValidate className="mt-7 flex flex-col gap-[18px]">
          <div>
            <label htmlFor="fp-email" className="block text-sm font-medium">
              Email
            </label>
            <input
              id="fp-email"
              type="email"
              autoComplete="email"
              placeholder="you@company.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              aria-invalid={!!fieldError}
              aria-describedby="fp-email-error"
              className={cn(inputClass, fieldBorder(!!fieldError))}
            />
            {fieldError && (
              <p id="fp-email-error" className="mt-2 text-xs text-[#FCA5A5]">
                {fieldError}
              </p>
            )}
          </div>
          {error && <FormError message={error} />}
          <button type="submit" disabled={submitting} className={primaryButton}>
            {submitting ? "Sending…" : "Send code"}
          </button>
        </form>
      </AuthLayout>
    )
  }

  if (step === "code") {
    return (
      <AuthLayout
        title="Check your email"
        subtitle={`If an account exists for ${email.trim()}, we sent it a 6-digit code.`}
        footer={backToLogin}
      >
        <form onSubmit={handleVerify} noValidate className="mt-7 flex flex-col gap-[18px]">
          <fieldset>
            <legend className="mb-2.5 text-sm font-medium">Verification code</legend>
            <CodeInput value={code} onChange={setCode} invalid={!!fieldError || !!error} disabled={submitting} />
            {fieldError && <p className="mt-2 text-xs text-[#FCA5A5]">{fieldError}</p>}
          </fieldset>
          {error && <FormError message={error} />}
          <button type="submit" disabled={submitting} className={primaryButton}>
            {submitting ? "Checking…" : "Verify code"}
          </button>
        </form>

        <p className="mt-5 text-center text-[13px] text-[#8B90A0]">
          Didn't get it? Check spam, or{" "}
          {secondsLeft > 0 ? (
            <span className="text-[#A7ACBA]">
              resend in 0:{String(secondsLeft).padStart(2, "0")}
            </span>
          ) : (
            <button type="button" onClick={handleResend} className="font-medium text-[#60A5FA] hover:text-[#93C5FD]">
              resend the code
            </button>
          )}
        </p>
        <p className="mt-1.5 text-center text-[13px]">
          <button
            type="button"
            onClick={() => {
              setStep("email")
              setCode("")
              setError(null)
              setFieldError(null)
            }}
            className="text-[#60A5FA] hover:text-[#93C5FD]"
          >
            Use a different email
          </button>
        </p>
      </AuthLayout>
    )
  }

  return (
    <AuthLayout
      title="Set a new password"
      subtitle="Choose a password you haven't used here before."
      footer={backToLogin}
    >
      <form onSubmit={handleSetPassword} noValidate className="mt-7 flex flex-col gap-[18px]">
        <div>
          <label htmlFor="np-new" className="block text-sm font-medium">
            New password
          </label>
          <PasswordInput
            id="np-new"
            autoComplete="new-password"
            placeholder="At least 10 characters"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            invalid={!!fieldError}
          />
          <p className="mt-2 text-xs text-[#8B90A0]">Use 10 or more characters. A short phrase is easiest to remember.</p>
        </div>
        <div>
          <label htmlFor="np-confirm" className="block text-sm font-medium">
            Confirm new password
          </label>
          <PasswordInput
            id="np-confirm"
            autoComplete="new-password"
            placeholder="Type it again"
            value={confirm}
            onChange={(e) => setConfirm(e.target.value)}
            invalid={!!fieldError}
          />
          {fieldError && <p className="mt-2 text-xs text-[#FCA5A5]">{fieldError}</p>}
        </div>
        {error && <FormError message={error} />}
        <button type="submit" disabled={submitting} className={primaryButton}>
          {submitting ? "Updating…" : "Update password"}
        </button>
      </form>
      <p className="mt-[18px] text-xs leading-relaxed text-[#8B90A0]">
        For your security, you'll be logged out on every other device.
      </p>
    </AuthLayout>
  )
}