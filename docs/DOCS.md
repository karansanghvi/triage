# Triage documentation

Triage is an OpenAI-compatible API. It sends each chat request to the cheapest model that is likely to answer it
well, using your own provider keys. If your app already uses the OpenAI SDK, you only change two lines.

## 1. Quickstart

**1. Create an account.** Sign up at [gettriage.dev](https://gettriage.dev/signup).

**2. Add a provider key.** In **Settings → Provider keys**, save a key for at least one of OpenAI, Anthropic, Groq
or Gemini. Triage checks it with the provider and stores it encrypted. You pay your provider directly.

**3. Create a Triage API key.** In **Settings → API keys**, create a key for your app. It starts with `trk_`.

> Warning: Copy your Triage API key when it appears. It is shown only once. If you lose it, revoke it and create a
> new one.

**4. Switch your client.** Point the OpenAI SDK at Triage and use your Triage API key. These are the only two lines
that change:

```python title="Python"
base_url="https://api.gettriage.dev/v1"
api_key="trk_..."
```

```js title="Node"
baseURL: "https://api.gettriage.dev/v1",
apiKey: "trk_...",
```

**5. Send your first request** with `model="auto"`, so Triage picks the model.

```python title="Python"
# pip install openai
from openai import OpenAI

client = OpenAI(base_url="https://api.gettriage.dev/v1", api_key="trk_...")

reply = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "What is 2 + 3?"}],
)
print(reply.choices[0].message.content)
```

```js title="Node"
// npm install openai
import OpenAI from "openai";

const client = new OpenAI({ baseURL: "https://api.gettriage.dev/v1", apiKey: "trk_..." });

const reply = await client.chat.completions.create({
  model: "auto",
  messages: [{ role: "user", content: "What is 2 + 3?" }],
});
console.log(reply.choices[0].message.content);
```

```bash title="curl"
curl https://api.gettriage.dev/v1/chat/completions \
  -H "Authorization: Bearer trk_..." \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "What is 2 + 3?"}]}'
```

**What a request can contain.** Triage supports the chat completions endpoint, `POST /v1/chat/completions`, with
these fields: `model`, `messages`, `stream`, `stream_options`, `temperature`, `max_tokens` (or
`max_completion_tokens`), `n` (only `1`) and `user`. Messages have the role `system`, `user` or `assistant` and text
content. Any other field is refused with a clear error, so nothing is silently ignored.

> Note: Triage works best with its default settings (`temperature` 0 and no `max_tokens`), because its routing was
> learned with them. Other values work, but the routing is less exact.

## 2. Choosing quality

With `model="auto"`, Triage predicts how likely a cheap and a medium model are to answer your prompt well. It
then uses the cheapest model that clears your **quality bar**. If neither clears it, it uses the strong model.

Set the bar with the `X-Triage-Quality` header. Leave it out to get `balanced`.

| Value | Bar | In plain words |
|---|---|---|
| `cheap` | 0.68 | Saves the most. Use it when an occasional weaker answer is fine. |
| `balanced` | 0.86 | The default. Close to the strong model's quality, at a lower cost. |
| `high` | 0.89 | Uses the strong model more often. Saves less, gives up the least quality. |
| a number from 0 to 1 | that number | Your own bar. Higher means stronger and pricier models. |

```python title="Python"
reply = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Prove that the square root of 2 is irrational."}],
    extra_headers={"X-Triage-Quality": "high"},
)
```

```js title="Node"
const reply = await client.chat.completions.create(
  { model: "auto", messages: [{ role: "user", content: "Prove that the square root of 2 is irrational." }] },
  { headers: { "X-Triage-Quality": "high" } },
);
```

```bash title="curl"
curl https://api.gettriage.dev/v1/chat/completions \
  -H "Authorization: Bearer trk_..." \
  -H "Content-Type: application/json" \
  -H "X-Triage-Quality: high" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "Prove that the square root of 2 is irrational."}]}'
```

**Telling Triage the task.** Triage guesses the kind of task from the prompt. If you already know it, send
`X-Triage-Task` with one of `qa`, `coding`, `reasoning`, `extraction`, `summarization`, `classification`,
`creative` or `open_ended`.

**Greetings.** A message that is only a greeting or a thank-you, such as "hi" or "Thanks!", always goes to the
cheapest model.

### Naming a specific model

Instead of `auto`, you can ask for one model by name. Triage calls exactly that model, with retries if the
provider has a short outage, but it does not route, does not switch to another model, and claims no savings for it.
You need a key for that model's provider.

| Model name | Provider | Tier | The provider's model |
|---|---|---|---|
| `groq-allam` | Groq | low | ALLaM 2 7B |
| `groq-small` | Groq | medium | gpt-oss-20b |
| `groq-large` | Groq | high | gpt-oss-120b |
| `gemini-flash-lite` | Gemini | low | Gemini 3.5 Flash-Lite |
| `gemini-flash` | Gemini | high | Gemini 3.8 Flash |
| `openai-luna` | OpenAI | low | GPT-6 Luna |
| `openai-terra` | OpenAI | medium | GPT-6.1 Terra |
| `openai-sol` | OpenAI | high | GPT-6.1 Sol |
| `claude-haiku` | Anthropic | low | Claude Haiku 5.5 |
| `claude-sonnet` | Anthropic | medium | Claude Sonnet 5.5 |
| `claude-opus` | Anthropic | high | Claude Opus 5.5 |

`GET /v1/models` lists the models with their prices per million tokens.

```python title="Python"
reply = client.chat.completions.create(
    model="claude-sonnet",
    messages=[{"role": "user", "content": "Write a haiku about routers."}],
)
```

```bash title="curl"
curl https://api.gettriage.dev/v1/models -H "Authorization: Bearer trk_..."
```

> Note: The OpenAI and Anthropic models are reasoning models. Their thinking counts toward `max_tokens`. If you set
> a low `max_tokens` on a named reasoning model, it can run out before it writes an answer; send a higher value.

## 3. Reading the response

A Triage response is a normal OpenAI chat completion. Triage adds what it did in two places: `X-Triage-*` response
headers, and a `triage` object in the response body. OpenAI clients ignore the extra field.

### Response headers

| Header | Meaning |
|---|---|
| `X-Triage-Request-Id` | The request's id. Look it up later with `GET /v1/requests/{id}`. |
| `X-Triage-Mode` | How the request was made: `account` (your Triage API key), `byok` (provider keys only) or `demo`. |
| `X-Triage-Model` | The model that answered. |
| `X-Triage-Tier` | Its tier: `low`, `medium` or `high`. |
| `X-Triage-Reason` | Why that tier, in words, e.g. `cheap 0.410 < tau 0.860; medium 0.950 >= tau`. |
| `X-Triage-P-Low` | The predicted chance that the cheap model answers well (`auto` only). |
| `X-Triage-P-Medium` | The predicted chance that the medium model answers well (`auto` only). |
| `X-Triage-Tau` | The quality bar used (`auto` only). |
| `X-Triage-Routed-By` | What chose the tier: `classifier` (the prediction), `small_talk` (a greeting) or `requested_model` (you named one). |
| `X-Triage-Preferred-Provider` | The preferred provider that was applied, or `none`. |
| `X-Triage-Fallback` | `none`, or e.g. `groq-small->gemini-flash` when the first model failed and a backup answered. |
| `X-Triage-Cache` | `hit` (a saved answer, free), `miss`, or `bypass` (not cacheable: `temperature` above 0). |
| `X-Triage-Verified` | Answer checking: `off` or `sync`. |
| `X-Triage-Check` | Which check decided, when checking is on (see [section 5](#5-optional-answer-checking)). Otherwise `none`. |
| `X-Triage-Escalated` | `true` if a check rejected the answer and a stronger model answered instead. |
| `X-Triage-Cost` | What this answer cost, in US dollars at the providers' list prices. |
| `X-Triage-Strong-Cost` | What the strong model would have cost for the same number of tokens. |
| `X-Triage-Saved` | The difference: your estimated saving (`auto` only). |

The probability headers have 6 decimals. Cost headers are left out when they aren't known, never guessed. On a
stream, the cost headers are never sent (see [Streaming](#4-streaming)).

> Note: `X-Triage-Saved` is an estimate: it assumes the strong model would have used as many tokens. When you name
> a model, Triage claims no saving.

### The `triage` object

The body carries the same routing details as the headers, with numbers as JSON numbers. A value that isn't known
is `null`.

```json title="Response body"
{
  "id": "chatcmpl-triage-5f0c...",
  "object": "chat.completion",
  "model": "groq-small",
  "choices": [
    { "index": 0, "message": { "role": "assistant", "content": "2 + 3 = 5." }, "finish_reason": "stop" }
  ],
  "usage": { "prompt_tokens": 14, "completion_tokens": 9, "total_tokens": 23 },
  "triage": {
    "request_id": "chatcmpl-triage-5f0c...",
    "model": "groq-small",
    "tier": "medium",
    "p_low": 0.41,
    "p_medium": 0.95,
    "tau": 0.86,
    "reason": "cheap 0.410 < tau 0.860; medium 0.950 >= tau",
    "cost": 0.00000375,
    "strong_cost": 0.0000075,
    "saved": 0.00000375,
    "cache": "miss",
    "fallback": "none",
    "mode": "account",
    "routed_by": "classifier",
    "preferred_provider": null
  }
}
```

Reading it in your code:

```python title="Python"
raw = client.chat.completions.with_raw_response.create(
    model="auto",
    messages=[{"role": "user", "content": "What is 2 + 3?"}],
)
print(raw.headers["X-Triage-Model"], raw.headers.get("X-Triage-Saved"))

reply = raw.parse()
print(reply.triage["tier"], reply.triage["reason"])
```

```js title="Node"
const { data: reply, response } = await client.chat.completions
  .create({ model: "auto", messages: [{ role: "user", content: "What is 2 + 3?" }] })
  .withResponse();

console.log(response.headers.get("X-Triage-Model"), response.headers.get("X-Triage-Saved"));
console.log(reply.triage.tier, reply.triage.reason);
```

```bash title="curl"
curl -i https://api.gettriage.dev/v1/chat/completions \
  -H "Authorization: Bearer trk_..." \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "What is 2 + 3?"}]}'
```

### Looking a request up later

| Endpoint | What it returns |
|---|---|
| `GET /v1/requests/{id}` | One of your requests: its routing details, tokens, cost and outcome. Never the prompt or the answer. |
| `GET /v1/stats` | Your account's totals: requests, spend and savings. |
| `GET /v1/models` | Every model name, its tier and its prices. |

The website's **Dashboard** shows the same numbers, with spend and savings per API key.

## 4. Streaming

Send `stream: true` and read the stream with the OpenAI SDK as usual.

- **Routing details** arrive on the **first chunk**, in its `triage` object: the model, tier, reason and
  probabilities. The response headers carry them too.
- **Cost** arrives on the **last chunk**, the one with a `finish_reason`. Its `triage` object is complete,
  including `cost`, `strong_cost` and `saved`.
- The `X-Triage-Cost`, `X-Triage-Strong-Cost` and `X-Triage-Saved` headers are **not** sent on a stream, because
  headers go out before the answer is written. Read the last chunk, or `GET /v1/requests/{id}` after the stream ends.

Both chunks keep `choices[0]`, so code that reads `chunk.choices[0]` never breaks.

```python title="Python"
stream = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Explain recursion in two sentences."}],
    stream=True,
)
for chunk in stream:
    triage = getattr(chunk, "triage", None)
    if triage and chunk.choices[0].finish_reason is None:
        print(f"[{triage['model']}, {triage['tier']}]")
    print(chunk.choices[0].delta.content or "", end="")
    if triage and chunk.choices[0].finish_reason:
        print(f"\n[cost ${triage['cost']}, saved ${triage['saved']}]")
```

```js title="Node"
const stream = await client.chat.completions.create({
  model: "auto",
  messages: [{ role: "user", content: "Explain recursion in two sentences." }],
  stream: true,
});
for await (const chunk of stream) {
  const triage = chunk.triage;
  if (triage && !chunk.choices[0].finish_reason) console.log(`[${triage.model}, ${triage.tier}]`);
  process.stdout.write(chunk.choices[0]?.delta?.content ?? "");
  if (triage && chunk.choices[0].finish_reason) console.log(`\n[cost $${triage.cost}, saved $${triage.saved}]`);
}
```

```bash title="curl"
curl -N https://api.gettriage.dev/v1/chat/completions \
  -H "Authorization: Bearer trk_..." \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "stream": true, "messages": [{"role": "user", "content": "Explain recursion in two sentences."}]}'
```

**If a model fails.** Before the first word is sent, Triage quietly switches to a backup model. After text has been
sent it can't switch, so the stream ends with an error event (`stream_interrupted`), which the OpenAI SDKs raise as
an `APIError`.

**If you stop reading.** When your client disconnects part-way, Triage stops the model, so it doesn't keep
generating (and billing) the rest of the answer. The request is recorded as stopped, with an estimated cost.

Set `stream_options: {"include_usage": true}` to get token counts in a final chunk, as with OpenAI.

## 5. Optional answer checking

Send `X-Triage-Verify: sync` to have Triage check the answer before it reaches you. If a check says the answer is
bad, Triage asks a stronger model and sends that answer instead.

- A **cheap-model** answer gets a free format check: is it empty, cut off, a refusal, in a different language than
  the question, or not the JSON you asked for?
- A **medium-model** answer gets the format check and then an AI check by a separate model. The AI check is
  skipped when the prediction was already very confident.
- A **strong-model** answer is final and never checked.

`X-Triage-Check` tells you which check decided:

| Value | Meaning |
|---|---|
| `format` | The format check decided. |
| `ai_checker` | The AI check decided. |
| `ai_checker_error` | The AI check failed, so Triage played it safe and escalated. |
| `skipped_confident` | The AI check was skipped: the prediction was confident enough. |
| `skipped_no_key` | The AI check was skipped: it runs on Groq, and you have no Groq key. |
| `none` | No check ran (checking is off, or the strong model answered). |

```python title="Python"
reply = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "What is 17 × 23?"}],
    extra_headers={"X-Triage-Verify": "sync"},
)
```

```bash title="curl"
curl https://api.gettriage.dev/v1/chat/completions \
  -H "Authorization: Bearer trk_..." \
  -H "Content-Type: application/json" \
  -H "X-Triage-Verify: sync" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "What is 17 × 23?"}]}'
```

Checking works only with `model="auto"` and without streaming (the answer has to be checked before any of it is
sent). It isn't available in the free demo.

**When it's worth it.** Checking adds a little time and, for medium answers, the cost of the AI check. In Triage's
own locked test, the AI check did no better than random, so it's off by default. Turn it on when a wrong answer costs you more
than a slower, slightly pricier one: for example, answers shown to customers without review. Leave it off for chat
and drafts.

> Note: `X-Triage-P-Low`, `X-Triage-P-Medium` and `X-Triage-Tau` always describe the first routing decision, even
> when a check escalated the answer. `X-Triage-Model` and `X-Triage-Tier` describe the model that actually answered.

## 6. Using your own provider keys

Triage never pays for your requests: every request uses your own provider keys, and you pay OpenAI, Anthropic, Groq
or Gemini directly. Triage only routes to providers you have a key for.

### Saved in Settings

Save keys in **Settings → Provider keys**. Each key is checked with the provider (one free call), encrypted, and
afterwards shown only by its last 4 characters. A key marked valid was accepted by the provider; that doesn't prove
it has credit.

If a provider later rejects a saved key, Triage marks it invalid and stops using it. Save it again to fix it.

### Sent with a request

You can also send a provider key with a single request. It replaces your saved key for that provider, on that
request only, and is never stored.

| Header | Provider |
|---|---|
| `X-OpenAI-Key` | OpenAI |
| `X-Anthropic-Key` | Anthropic |
| `X-Groq-Key` | Groq |
| `X-Gemini-Key` | Gemini |

```python title="Python"
reply = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "What is 2 + 3?"}],
    extra_headers={"X-Anthropic-Key": "sk-ant-..."},
)
```

```js title="Node"
const reply = await client.chat.completions.create(
  { model: "auto", messages: [{ role: "user", content: "What is 2 + 3?" }] },
  { headers: { "X-Anthropic-Key": "sk-ant-..." } },
);
```

```bash title="curl"
curl https://api.gettriage.dev/v1/chat/completions \
  -H "Authorization: Bearer trk_..." \
  -H "Content-Type: application/json" \
  -H "X-Anthropic-Key: sk-ant-..." \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "What is 2 + 3?"}]}'
```

These headers also work without a Triage API key, with a lower daily limit (see [Limits](#8-limits)).

### Preferred provider

If you have keys for several providers, you can say which one to try first. Set a default in **Settings → Provider
keys**, or send `X-Triage-Prefer-Provider` (`openai`, `anthropic`, `groq` or `gemini`) to override it for one
request.

It changes which model answers within the chosen tier, not the tier itself. It applies only to `model="auto"` and
only to a provider you have a key for; otherwise it's ignored. Savings are then measured against that provider's
strongest model.

```python title="Python"
reply = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "What is 2 + 3?"}],
    extra_headers={"X-Triage-Prefer-Provider": "anthropic"},
)
```

```bash title="curl"
curl https://api.gettriage.dev/v1/chat/completions \
  -H "Authorization: Bearer trk_..." \
  -H "Content-Type: application/json" \
  -H "X-Triage-Prefer-Provider: anthropic" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "What is 2 + 3?"}]}'
```

## 7. Errors

Errors use OpenAI's format, so the OpenAI SDKs raise them as usual:

```json title="Error body"
{ "error": { "message": "...", "type": "invalid_request_error", "param": "X-Triage-Quality", "code": "invalid_value" } }
```

`param` names the field or header that was wrong. Messages never repeat what you sent, so a key can't leak into
your logs.

### Chat requests

| Status | Code | Meaning | What to do |
|---|---|---|---|
| 400 | `invalid_request` | The body is malformed (e.g. no messages, a wrong type). | Fix the field named in `param`. |
| 400 | `invalid_value` | A header has a value Triage doesn't accept. | Use a value listed in these docs. |
| 400 | `unsupported_parameter` | A field Triage doesn't support yet, or `n` above 1. | Remove that field. |
| 400 | `unsupported_combination` | `X-Triage-Verify: sync` with streaming or a named model. | Drop the header, or use `model="auto"` without streaming. |
| 400 | `no_provider_keys` | Your account has no usable provider key, and none was sent. | Add a key in Settings, or send one in a header. |
| 400 | `no_usable_model` | The keys you have reach no model. | Add a key for another provider. |
| 400 | `max_tokens_too_low` | A named reasoning model spent all its tokens thinking. | Send a higher `max_tokens`. |
| 400 | `demo_unsupported` | Answer checking isn't available in the free demo. | Use your own provider key. |
| 401 | `invalid_api_key` | The Triage API key is missing, unknown or revoked. | Check the key, or create a new one. |
| 401 | `invalid_provider_key` | A provider rejected your key. The message says which. | Update that key in Settings, or fix the header. |
| 403 | `model_not_allowed` | You named a model on a provider you have no key for. | Add that provider's key, or use `auto`. |
| 404 | `model_not_found` | No model has that name. | Use `auto` or a name from `GET /v1/models`. |
| 404 | `not_found` | The request id doesn't exist or isn't yours. | Check the id. |
| 413 | `demo_prompt_too_long` | A free demo prompt is over 4,000 characters. | Shorten it, or use your own provider key. |
| 429 | `account_rate_limited` | Your account reached its per-minute or per-day limit. | Wait `Retry-After` seconds. |
| 429 | `demo_limit_reached` | You've used today's free demo requests. | Use your own provider key, or wait until 00:00 UTC. |
| 429 | `demo_busy` | The free demo is busy. | Try again in a few seconds. |
| 429 | `demo_paused` | The free demo reached its daily total and is paused. | Use your own provider key, or wait until 00:00 UTC. |
| 429 | `public_limit_reached` | Your address used today's requests without a Triage API key. | Send a Triage API key, or wait until 00:00 UTC. |
| 500 | `internal_error` | Something went wrong on Triage's side. | Try again; report it if it keeps happening. |
| 502 | `model_error` | The model failed while answering. | Try again. |
| 503 | `provider_overloaded` | Every model tried was too busy. The message names the provider. | Try again soon, or add another provider's key. |
| 503 | `provider_unavailable` | Every model tried was busy or too slow. | Try again soon, or add another provider's key. |
| 503 | `all_models_failed` | No model could answer. | Try again. |
| 504 | `provider_timeout` | Every model tried ran out of time. | Try again, or shorten the request. |
| in a stream | `stream_interrupted` | The model failed after part of the answer was sent. | Retry the request. |

Every 429 carries a `Retry-After` header, in seconds.

### Account and keys

These come from the website's account pages (sign-up, log-in, Settings) and the account API behind them.

| Status | Code | Meaning | What to do |
|---|---|---|---|
| 400 | `invalid_value` | A field is outside its rules, e.g. a password that isn't 10 to 128 characters. `param` names it. | Fix that field. |
| 400 | `invalid_provider_key` | The provider rejected the key you tried to save. Nothing was stored. | Check the key. |
| 400 | `invalid_code` | A password reset code is wrong, expired or used up (5 wrong tries end a code). | Ask for a new code. |
| 400 | `invalid_reset_token` | A password reset link expired or was already used. | Start the reset again. |
| 401 | `invalid_credentials` | Wrong email or password, or an account locked after 5 wrong passwords in 15 minutes (the same error, on purpose). | Check the password, wait 15 minutes, or reset your password. |
| 401 | `invalid_session` | You're logged out (or the session expired). | Log in again. |
| 403 | `incorrect_password` | The current password was wrong. | Re-enter it. |
| 404 | `not_found` | That key doesn't exist or isn't yours. | Refresh the page. |
| 409 | `email_taken` | An account with this email already exists. | Log in instead. |
| 409 | `too_many_api_keys` | You have 20 active Triage API keys, the most allowed. | Revoke one first. |
| 409 | `key_unreadable` | A saved provider key can no longer be read. | Save it again. |
| 429 | `too_many_attempts` | Too many attempts in a short time (see [Limits](#8-limits)). | Wait `Retry-After` seconds. |
| 503 | `email_unavailable` | Password reset emails are unavailable right now. | Try again later. |
| 503 | `encryption_unavailable` | Saving provider keys is unavailable right now. | Try again later. |

## 8. Limits

| How you call Triage | Limit |
|---|---|
| **Account** (a Triage API key) | 60 requests per minute and 10,000 per day. No prompt length limit. |
| **Your own keys without a Triage API key** (provider key headers only) | 200 requests per day per visitor. |
| **Free demo** (no keys at all) | 10 requests per day per visitor; prompts up to 4,000 characters; answers up to 2,048 tokens; no answer checking. |

Days are UTC days and reset at 00:00 UTC. The free demo runs on Triage's own keys, so it also has a shared daily
total; when that is used up, the demo pauses until 00:00 UTC. A demo response says how many requests you have left
in `X-Triage-Demo-Remaining`, `X-Triage-Demo-Limit` and `X-Triage-Demo-Resets`, and `X-Triage-Max-Tokens` shows the
answer length it used.

Each message counts once, however many models Triage tries. Stopped answers count; if a request fails on Triage's
side (every model failed, or the answer broke off part-way), it doesn't count.

Account limits:

| What | Limit |
|---|---|
| Active Triage API keys | 20 per account |
| New Triage API keys | 10 per hour |
| Saving or re-checking provider keys | 20 per hour |
| Sign-up | 5 per hour from one IP address |
| Log-in | 10 per 15 minutes from one IP address |
| Wrong passwords | 5 per account per 15 minutes, counted across log-in, password change and account deletion |
| Password reset emails | 3 per hour per email address and 10 per hour per IP address; at most one per minute |
| Password reset codes | 10 checks per 15 minutes per IP address; 5 tries per code; 10 wrong codes per email address per day |
| Passwords | 10 to 128 characters |

Going over a limit is a 429 `too_many_attempts` with `Retry-After`, except in these cases:
- **Too many wrong passwords at log-in:** the same 401 `invalid_credentials` as a wrong password, so nobody can
  tell that an account is locked. A password change or account deletion in that state is a 429.
- **A second reset email within a minute:** the usual 202, but nothing is sent. The code you already have still
  works.
- **A 6th try on one reset code:** 400 `invalid_code`. The code no longer works; ask for a new one.
- **A Triage API key beyond the 20 active ones:** 409 `too_many_api_keys`.

> Tip: Identical requests at `temperature` 0 are answered from a cache for free (`X-Triage-Cache: hit`). Saved
> answers are kept for 24 hours and are only reused for requests made with the same provider key.

## 9. Privacy and security

Triage never stores your prompts or answers. It keeps routing details, token counts and costs for each request, so
your dashboard can show them. Your provider keys are encrypted and are only decrypted for the one request that
uses them; Triage API keys are stored only as a keyed hash, which is why they're shown once. Requests from
accounts are never used to train the router. The full details are in the [privacy policy](https://gettriage.dev/privacy).
