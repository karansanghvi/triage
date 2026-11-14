// A failed Triage request, with Triage's error code so the UI can react to it
export class TriageError extends Error {
  code: string
  retryAfter?: number

  constructor(code: string, message: string, retryAfter?: number) {
    super(message)
    this.name = "TriageError"
    this.code = code
    this.retryAfter = retryAfter
  }
}