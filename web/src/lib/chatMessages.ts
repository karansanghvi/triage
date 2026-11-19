// The messages the web chat sends to Triage. No value imports, so `node --test` can run its test directly.

export type ApiMessage = { role: "system" | "user" | "assistant"; content: string }

// Every chat request carries this (unless the conversation already has a system message): some models, such as
// groq-allam (ALLaM, Arabic-English), sometimes answer short English messages like "hello" in Arabic.
// The backend also puts the same sentence first for groq-allam (routing.yaml model_params system_prefix).
export const LANGUAGE_INSTRUCTION = "Always reply in the same language as the user's latest message."

export function withLanguageInstruction(messages: ApiMessage[]): ApiMessage[] {
  if (messages.some((m) => m.role === "system")) return messages
  return [{ role: "system", content: LANGUAGE_INSTRUCTION }, ...messages]
}
