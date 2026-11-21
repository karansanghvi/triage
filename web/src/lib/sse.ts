// Reads a streamed chat answer: OpenAI-style server-sent events, "data: {...}" lines with events separated by a
// blank line, ending with "data: [DONE]". No imports from "@/", so the Node tests can load it directly.

export type StreamTriage = { [key: string]: unknown }

export type StreamChunk = {
  choices?: { delta?: { content?: string | null } | null; finish_reason?: string | null }[]
  triage?: StreamTriage
  error?: { code?: string; message?: string }
}

export type StreamHandlers = {
  onStartTriage?: (triage: StreamTriage) => void   // the first chunk's routing (cost not known yet)
  onDelta?: (text: string) => void                 // the next piece of the answer
}

export type StreamOutcome =
  // The server finished the answer: a finish_reason chunk or [DONE] arrived. finalTriage is the finish chunk's
  // "triage" (with cost), or null if the server didn't send one there.
  | { kind: "complete"; content: string; finishReason: string | null; startTriage: StreamTriage | null;
      finalTriage: StreamTriage | null }
  // The server reported a failure part-way ({"error": ...} event)
  | { kind: "error"; content: string; code: string | null; message: string | null; startTriage: StreamTriage | null }
  // The connection closed before the answer was finished: no finish_reason chunk and no [DONE]
  | { kind: "interrupted"; content: string; startTriage: StreamTriage | null }

export class StreamFormatError extends Error {}

// Splits text that arrives in arbitrary pieces into complete SSE events and returns each event's data. Lines may
// end with \n, \r\n or \r; a "data:" field may or may not have a space after the colon; several data lines in one
// event are joined with "\n" (the SSE spec). Nothing is returned until its event's blank line has arrived.
export class SseParser {
  private buffer = ""
  private data: string[] = []

  push(text: string): string[] {
    this.buffer += text
    const events: string[] = []
    for (;;) {
      const match = /\r\n|\r|\n/.exec(this.buffer)
      if (!match) break
      // A lone \r at the very end may be the first half of \r\n: wait for the next piece
      if (match[0] === "\r" && match.index === this.buffer.length - 1) break
      const line = this.buffer.slice(0, match.index)
      this.buffer = this.buffer.slice(match.index + match[0].length)
      this.line(line, events)
    }
    return events
  }

  // The stream has ended: a last event without its closing blank line still counts
  end(): string[] {
    const events: string[] = []
    if (this.buffer) this.line(this.buffer, events)
    this.buffer = ""
    this.line("", events)
    return events
  }

  private line(line: string, events: string[]) {
    if (line === "") {
      if (this.data.length) events.push(this.data.join("\n"))
      this.data = []
      return
    }
    if (line.startsWith(":")) return                      // a comment (keep-alive)
    const colon = line.indexOf(":")
    const field = colon === -1 ? line : line.slice(0, colon)
    let value = colon === -1 ? "" : line.slice(colon + 1)
    if (value.startsWith(" ")) value = value.slice(1)
    if (field === "data") this.data.push(value)
  }
}

// Reads the whole stream. Resolves with how it ended; rejects only if the body itself fails (a dropped
// connection, an abort) or an event isn't valid JSON (StreamFormatError).
export async function readChatStream(
  body: ReadableStream<Uint8Array>,
  { onStartTriage, onDelta }: StreamHandlers = {}
): Promise<StreamOutcome> {
  const reader = body.getReader()
  const decoder = new TextDecoder()                    // stream: true keeps a UTF-8 character split across reads
  const parser = new SseParser()
  let content = ""
  let startTriage: StreamTriage | null = null
  let finalTriage: StreamTriage | null = null
  let finishReason: string | null = null
  let finished = false                                 // a finish_reason chunk arrived
  let done = false                                     // [DONE] arrived
  let failure: { code: string | null; message: string | null } | null = null

  const handle = (data: string) => {
    if (data === "[DONE]") {
      done = true
      return
    }
    let chunk: StreamChunk
    try {
      chunk = JSON.parse(data) as StreamChunk
    } catch {
      throw new StreamFormatError("The answer arrived in a form Triage's chat couldn't read.")
    }
    if (chunk.error) {
      failure = { code: chunk.error.code ?? null, message: chunk.error.message ?? null }
      return
    }
    const choice = chunk.choices?.[0]
    if (chunk.triage) {
      if (choice?.finish_reason) finalTriage = chunk.triage
      else if (startTriage === null) {
        startTriage = chunk.triage
        onStartTriage?.(chunk.triage)
      }
    }
    const piece = choice?.delta?.content
    if (piece) {
      content += piece
      onDelta?.(piece)
    }
    if (choice?.finish_reason) {
      finishReason = choice.finish_reason
      finished = true
    }
  }

  try {
    while (!done && !failure) {
      const { value, done: ended } = await reader.read()
      if (ended) {
        for (const data of parser.push(decoder.decode())) handle(data)    // flush the decoder, then the parser
        for (const data of parser.end()) handle(data)
        break
      }
      for (const data of parser.push(decoder.decode(value, { stream: true }))) {
        handle(data)
        if (done || failure) break
      }
    }
  } finally {
    reader.releaseLock()
  }

  if (failure !== null) {
    const { code, message } = failure as { code: string | null; message: string | null }
    return { kind: "error", content, code, message, startTriage }
  }
  if (finished || done) return { kind: "complete", content, finishReason, startTriage, finalTriage }
  return { kind: "interrupted", content, startTriage }
}
