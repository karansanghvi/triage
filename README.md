# Triage

An LLM router that sends each request to the cheapest model that can handle it.

**Status:** Phases 1–4 complete: providers and reliability, data collection, the routing classifier, and an
OpenAI-compatible API with verification, logging and retraining. The sections below describe each part in more detail.

## Results so far (Phase 4)

Phase 4 tested whether checking the medium model's answers, and escalating only the bad ones, could match the strong
model for less. On the validation set it looked promising. The pre-registered test run (149 prompts, used once) did not
confirm it: selective checking got 139/149 right, about what random routing at the same cost gets, while predict-only
routing got 144/149 against 145 for always using the strong model, but saved only 12.4% of the strong model's cost
(28.4% on validation). The groq-small checker caught only 4 of the 12 bad
answers it saw. Predict-only routing stays the default, and checking stays opt-in. Full numbers, caveats and next
steps are in [reports/PHASE_4.md](reports/PHASE_4.md).

## What Phase 1 does

- **One function for every model.** `send_request(messages, model_config, stream=False)` works the same way for Groq, Gemini, Ollama, OpenRouter, OpenAI, and Anthropic, and always returns the same `Response`: the answer, input and output tokens, latency, actual cost, and realistic cost.
- **Streaming.** Answers can arrive piece by piece, with time-to-first-token measured.
- **A model registry in config.** `config/models.yaml` lists every model with its provider, tier, and price. Tiers are ordered by real price (low < medium < high), so routing a request down a tier always saves money. Free models carry a `price_as` price, so savings numbers stay realistic.
- **Honest costs.** Hidden reasoning ("thinking") tokens are counted as output, since providers bill them that way.
- **One error type.** Every provider's errors become a `ProviderError` with a kind: `rate_limit`, `server`, `overloaded`, `timeout`, `connection`, `client`, or `config`. Timeouts include 408/504 and each provider's deadline errors; overloaded includes Gemini 503, Anthropic 529, OpenAI 503 and Groq 498.
- **Reliability layer** (`app/reliability.py`):
  - a timeout on every call, longer for long inputs: `base + timeout_per_1k_input_tokens x input tokens / 1000`, at most `timeout_cap_seconds` (per model, also for the first-token limit)
  - a time budget per request across the whole chain (`request_time_budget_seconds`, 150): each call gets at most the time left, and when it runs out the rest are skipped
  - retries with exponential backoff and full jitter, only for rate limits (429) and server errors (5xx); timeouts and overloads move straight to the next model
  - fallback chains per tier in `config/routing.yaml`, with backups on a different provider
  - a circuit breaker: 3 failures in a row skip a model for 60 seconds (server errors, timeouts and rate limits are about one model); connection and auth errors skip every model on that provider
  - every fallback logged for the dashboard
- **Exact-match cache** (`app/cache.py`): repeated `temperature=0` requests are answered from a SQLite file in about a millisecond, for $0, with a 24-hour expiry and hit/savings tracking.

## Project structure

```
triage/
├── app/
│   ├── registry.py            # loads and validates models.yaml
│   ├── reliability.py         # retries, fallback chains, circuit breaker
│   ├── cache.py               # exact-match response cache
│   └── providers/
│       ├── base.py            # send_request() and Response
│       ├── common.py          # shared types and ProviderError
│       ├── openai_provider.py # adapter for OpenAI-format APIs
│       ├── groq_provider.py
│       ├── ollama_provider.py
│       ├── openrouter_provider.py
│       ├── gemini_provider.py
│       └── anthropic_provider.py
├── config/
│   ├── models.yaml            # models, tiers, and prices
│   └── routing.yaml           # fallback chains and reliability settings
├── scripts/                   # hands-on checks (see below)
└── tests/                     # unit tests (no network needed)
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env           # then add your API keys
```

Free keys: Groq (console.groq.com) and Gemini (aistudio.google.com). For local models, install Ollama and pull `llama3.2:3b` and `qwen2.5:3b`.

## Commands

```bash
python -m pytest                     # unit tests
python -m app.registry               # list models and example costs
python -m scripts.check_providers    # check every API key works
python -m scripts.try_models         # one real request to every model
python -m scripts.try_reliability    # retries, fallback, and a simulated outage
python -m scripts.try_cache          # cache hits and misses
python -m scripts.final_check        # end-to-end test with a real outage
```

## Design notes

- **Retries only where they can help.** A 429 or 5xx may succeed a moment later; a bad key or bad request never will.
- **Library retries are turned off** (`max_retries=0`), so the reliability layer is the only thing deciding when to retry.
- **Only temperature 0 is cached**, because only then is a saved answer a fair substitute. Empty or cut-off answers are never saved.
- **The cache wraps `send_request()`** instead of changing it, and plugs into the reliability layer through `send_fn`.
- **Prices live in config**, so a provider price change is a two-number edit, not a code change.
## The OpenAI-compatible API (Phase 4)

Triage speaks OpenAI's chat completions API, so any OpenAI client works by changing its base URL. Use
`model="auto"` to let Triage pick the cheapest model likely to pass, or a name from `GET /v1/models` to call
that model directly.

```bash
pip install -r requirements.txt
TRIAGE_API_KEYS=dev-key TRIAGE_ADMIN_KEYS=admin-key TRIAGE_HASH_SECRET="$(openssl rand -hex 32)" \
  uvicorn app.api.main:app --port 8000 --no-access-log
# TRIAGE_FAKE_MODELS=1 answers with fake models (no provider keys, no API calls); TRIAGE_CACHE=off disables the cache
# Fake mode keeps its own cache and database (data/cache/fake_responses.sqlite3, data/fake.sqlite3), never uses
# DATABASE_URL, and refuses the real cache, so a fake answer can never be served or logged as real
```

Python:

```python
from openai import OpenAI
print(OpenAI(base_url="http://localhost:8000/v1", api_key="dev-key").chat.completions.create(model="auto", messages=[{"role": "user", "content": "What is 2 + 3?"}]).choices[0].message.content)
```

JavaScript (`npm install openai`):

```js
import OpenAI from "openai";

const client = new OpenAI({ baseURL: "http://localhost:8000/v1", apiKey: "dev-key" });
const stream = await client.chat.completions.create({
  model: "auto",
  messages: [{ role: "user", content: "What is 2 + 3?" }],
  stream: true,
});
for await (const chunk of stream) process.stdout.write(chunk.choices[0]?.delta?.content ?? "");
```

**Request headers** (all optional):

| header | values | |
|---|---|---|
| `X-Triage-Quality` | `cheap`, `balanced`, `high`, or a number from 0 to 1 | the quality dial (tau); default `balanced` |
| `X-Triage-Task` | `qa`, `coding`, `reasoning`, `extraction`, `summarization`, `classification`, `creative`, `open_ended` | skips the task-type guess |
| `X-Triage-Verify` | `off`, `sync` | `sync` checks the answer before sending it and escalates a bad one (`model="auto"`, non-streaming only; 400 otherwise). Policy: `verification:` in `config/routing.yaml` |

Anything else is a 400 that names the header.

**Response headers:** `X-Triage-Request-Id`, `X-Triage-Model` (the model that answered), `X-Triage-Tier`,
`X-Triage-Reason`, `X-Triage-Fallback` (`primary->backup`, or `none`), `X-Triage-Cache` (`hit`, `miss`,
`bypass`), `X-Triage-Verified`, `X-Triage-Escalated`, `X-Triage-Check` (the check that decided: `format`,
`ai_checker`, `ai_checker_error` or `none`), and on non-streaming responses `X-Triage-Cost` and
`X-Triage-Saved` (US dollars at paid prices). The full list, including the routing probabilities, is in
"Accounts: a guide for developers" below. On a stream, the headers go out once the first token exists,
so model, tier and fallback are real, but cost isn't known yet. Those two headers are left out rather than
guessed. Fetch the real figures with `GET /v1/requests/<request id>` once the stream ends. `Saved` assumes the
strong model would have used the same number of tokens, so it's an estimate.

**Other endpoints:** `GET /v1/models` (with prices), `GET /v1/stats` (savings for your key),
`GET /v1/requests/{id}` (one of your own requests), and for admin keys `GET`/`PUT /v1/routing-config`, which
changes the quality presets and backup models without a restart. Those changes are kept in memory only. The
first model of each tier can't be changed, because the classifier predicts for that exact model.

**Settings:** `temperature` and `max_tokens` are honoured. The classifier learned from temperature 0 and
max_tokens 4000, so `auto` requests with other values are logged with `settings_match_training: false`.

**Reliability:** a failing model falls back to the tier's backup, but only before the first token is sent.
After that, the stream ends with an error event (the openai SDKs raise `APIError`). A stream whose first token
doesn't arrive within the limit in `routing.yaml` (`streaming:`) fails over. A long answer that streams steadily
is never cut off.

## Public access: the free demo and BYOK (Phase 5)

With `TRIAGE_PUBLIC=on`, `POST /v1/chat/completions` also accepts requests without a Triage key. Each request
gets a mode, sent back as `X-Triage-Mode`:

| Mode | Triage key | Provider key headers | Keys used | Limits |
|---|---|---|---|---|
| `api` | valid | none | the server's | none (as before) |
| `byok` | valid or none | `X-OpenAI-Key`, `X-Anthropic-Key`, `X-Groq-Key`, `X-Gemini-Key` | the user's, only their providers | 200/day per visitor without a Triage key |
| `demo` | none | none | the server's | `config/demo.yaml` |

An `Authorization` header that is sent but invalid is always a 401. Every other endpoint still needs a Triage key.

**Demo limits** (`config/demo.yaml`, UTC days):
- 10 requests per visitor, and 300 requests, 150,000 tokens and 600 model calls per day for all visitors together.
- 4 requests per minute for all visitors together.
- 8,000 prompt characters. `max_tokens` is lowered to 2,048, and the response says so in `X-Triage-Max-Tokens`.
- No sync verification.

Refusals are 429s with `Retry-After`: `demo_limit_reached` and `demo_busy` suggest adding a provider key, and
`demo_paused` means the demo is paused until 00:00 UTC. A prompt over the limit is a 413 `demo_prompt_too_long`.

Demo responses carry `X-Triage-Demo-Remaining`, `-Limit` and `-Resets`. `GET /v1/demo/status` (public) returns
the visitor's allowance and whether the demo is paused, so a website can show it before the first request.

**Visitors** are identified by IP. Set `TRIAGE_TRUSTED_PROXY_HOPS` to the number of proxies in front of the
server; the default 0 ignores `X-Forwarded-For`, which any client can fake. IPs are stored only as an HMAC with
`TRIAGE_HASH_SECRET` (IPv6 grouped by /64), and the request log only ever sees `key_id` `demo` or `public`.

Demo and BYOK requests are never used for retraining and get no background labels.

**Deploying:**
- `TRIAGE_SERVER_PROVIDERS=groq,gemini` drops Ollama on a host that can't run it. The low tier then falls back
  from groq-allam to gemini-flash-lite.
- `TRIAGE_CORS_ORIGINS` lists the website's origins.
- The demo counters live in the database with the request log (`DATABASE_URL`; see "Database"), so on Postgres
  they survive restarts and redeploys and every instance shares them. The per-minute limit is kept in memory by
  each instance. Without `DATABASE_URL` they're in the local `data/triage.sqlite3`, which an ephemeral disk loses
  on every redeploy. The old `data/demo/usage.sqlite3` and `TRIAGE_DEMO_DB` are no longer used; the counters
  weren't copied (they're per day and kept 7 days).

## Accounts: a guide for developers

With accounts on (`TRIAGE_ACCOUNTS=on`; see "Operating Triage"), you can sign up on the Triage website or API,
save your own provider keys, create Triage API keys, and call Triage from your apps with only a Triage key.
Triage routes each request to the cheapest model it expects to answer well, using **your** provider keys, so
you pay your providers directly.

### Quick start

```bash
B=https://your-triage-server; CT="Content-Type: application/json"

# 1. Sign up (or log in with POST /v1/auth/login). The session token is shown only here.
curl -s -X POST $B/v1/auth/signup -H "$CT" -d '{"email":"you@example.com","password":"a long password"}'
S="Authorization: Bearer trs_..."             # the session_token from the response

# 2. Save at least one provider key (checked with the provider first; never shown again)
curl -s -X PUT $B/v1/account/provider-keys/groq -H "$S" -H "$CT" -d '{"key":"gsk_..."}'

# 3. Create a Triage API key for your app. Copy "key" now: it's shown only once.
curl -s -X POST $B/v1/account/api-keys -H "$S" -H "$CT" -d '{"name":"my app"}'

# 4. Call Triage with it, like the OpenAI API
curl -s $B/v1/chat/completions -H "Authorization: Bearer trk_..." -H "$CT" \
  -d '{"model":"auto","messages":[{"role":"user","content":"What is 2 + 3?"}]}'
```

Python (`pip install openai`):

```python
from openai import OpenAI

client = OpenAI(base_url="https://your-triage-server/v1", api_key="trk_...")
reply = client.chat.completions.create(
    model="auto",                                     # or a model name from GET /v1/models
    messages=[{"role": "user", "content": "What is 2 + 3?"}],
    extra_headers={"X-Triage-Quality": "balanced"},   # optional; see the headers below
)
print(reply.choices[0].message.content)
```

Node (`npm install openai`):

```js
import OpenAI from "openai";

const client = new OpenAI({ baseURL: "https://your-triage-server/v1", apiKey: "trk_..." });
const reply = await client.chat.completions.create(
  { model: "auto", messages: [{ role: "user", content: "What is 2 + 3?" }] },
  { headers: { "X-Triage-Prefer-Provider": "groq" } },   // optional
);
console.log(reply.choices[0].message.content);
```

Streaming (`stream: true`) works the same way.

### Which keys a request uses

- **Your saved provider keys** with status `valid` or `unverified`, decrypted for that one request and then
  dropped. Triage never keeps them in memory, a cache or a background job.
- **An `X-OpenAI-Key`, `X-Anthropic-Key`, `X-Groq-Key` or `X-Gemini-Key` header** replaces the saved key for
  that provider on that request.
- Triage only routes to providers you have a key for. With none, the request is 400 `no_provider_keys`.
- If a provider rejects a saved key (401/403), Triage marks it `invalid` (the reason is in `status_detail`) and
  stops using it, even if another provider then answered. If nothing answered, you get 401
  `invalid_provider_key` saying which saved key to update. Save it again or re-check it with
  `POST /v1/account/provider-keys/{provider}/verify`.
- A saved key's status `valid` means the provider **accepted** it (one free list-models call); it doesn't prove
  credit or quota.

### Preferred provider

Set a default with `PATCH /v1/account {"preferred_provider": "anthropic"}` (`null` clears it), or per request with
`X-Triage-Prefer-Provider`, which overrides it. It applies only to `model: "auto"` and only to a provider you
have a key for (otherwise it's ignored). The tier decision doesn't change: within each tier, that provider's model
is tried first (its first model in the chain; a last-resort model such as `gemini-flash-lite` at the end of medium
and high stays last). Savings are then measured against that provider's strongest model.

### Small talk

With `model: "auto"`, a message that is only a greeting or pleasantry ("hi", "Thanks!", "Good morning 👋") goes
to the cheapest model without asking the classifier. The whole latest user message must be one of the phrases in
`config/routing.yaml` (`small_talk.phrases`, editable), after trimming, lowercasing and removing trailing
punctuation and emoji; "hi, can you help with X" is routed as usual. Fallbacks and answer checking work as
always. These requests say `small_talk` in `X-Triage-Routed-By`, the response's `triage.routed_by` and the
request log, and are left out of classifier training data, background labels and the `GET /v1/stats` numbers
(counted on their own as `small_talk_requests`).

### Headers

**Request** (all optional):

| Header | Values | Effect |
|---|---|---|
| `X-Triage-Quality` | `cheap`, `balanced`, `high`, or a number from 0 to 1 | The quality bar (tau). Higher means stronger, pricier models. Default `balanced` |
| `X-Triage-Task` | `qa`, `coding`, `reasoning`, `extraction`, `summarization`, `classification`, `creative`, `open_ended` | Skips the task-type guess |
| `X-Triage-Verify` | `off`, `sync` | `sync` checks the answer before sending it and escalates a bad one (`model: "auto"`, non-streaming only) |
| `X-Triage-Prefer-Provider` | `openai`, `anthropic`, `groq`, `gemini` | Try that provider's models first (see above) |
| `X-OpenAI-Key`, `X-Anthropic-Key`, `X-Groq-Key`, `X-Gemini-Key` | a provider key | Use it instead of your saved key for that provider |

**Response** (also exposed to browsers through CORS):

| Header | Meaning |
|---|---|
| `X-Triage-Request-Id` | Look the request up later with `GET /v1/requests/{id}` |
| `X-Triage-Mode` | `account` for your requests (`api`, `byok`, `demo` otherwise) |
| `X-Triage-Model`, `X-Triage-Tier` | The model that answered and its tier (`low`, `medium`, `high`) |
| `X-Triage-Reason` | Why, in words, e.g. `cheap 0.400 < tau 0.860; medium 0.950 >= tau` |
| `X-Triage-P-Low`, `X-Triage-P-Medium`, `X-Triage-Tau` | The predicted pass chances and the bar used (6 decimals; `model: "auto"` only, not for small talk) |
| `X-Triage-Routed-By` | What decided the tier: `classifier`, `small_talk` (see below) or `requested_model` |
| `X-Triage-Preferred-Provider` | The preferred provider applied, or `none` |
| `X-Triage-Fallback` | `primary->backup` when a backup answered, else `none` |
| `X-Triage-Cache` | `hit`, `miss` or `bypass` (only `temperature: 0`, non-streaming answers are cached) |
| `X-Triage-Verified`, `X-Triage-Check`, `X-Triage-Escalated` | Sync verification: on or off, which check decided, whether a stronger model was used |
| `X-Triage-Cost`, `X-Triage-Strong-Cost`, `X-Triage-Saved` | US dollars at list prices: this answer, the strong model at the same tokens, and the difference (non-streaming only; left out when unknown) |

Non-streaming responses also carry the same facts as a `triage` object in the body.

### Limits

| What | Limit | Over it |
|---|---|---|
| Chat | 60 requests a minute and 10,000 a day per account (`config/accounts.yaml`) | 429 `account_rate_limited` with `Retry-After` |
| Saving or re-checking provider keys | 20 per account per hour | 429 `too_many_attempts` with `Retry-After` |
| Triage API keys | 20 active per account; 10 new per hour | 409 `too_many_api_keys`; 429 `too_many_attempts` |
| Sign-up | 5 per IP per hour | 429 `too_many_attempts` |
| Log-in | 10 per IP per 15 minutes; 5 wrong passwords per account per 15 minutes | 429; for wrong passwords, the same 401 as a wrong password |
| Password reset codes | 3 requests per email and 10 per IP per hour; one email per address per 60 seconds | 429 `too_many_attempts`; inside the 60 seconds, the usual 202 but no email |
| Checking reset codes | 10 per IP per 15 minutes; 5 tries per code; 10 wrong codes per email per day | 429 `too_many_attempts`; after 5 wrong tries the code is dead (request a new one) |

### Error codes

Errors use OpenAI's shape, `{"error": {"message", "type", "param", "code"}}`.

| Status | `code` | When |
|---|---|---|
| 400 | `invalid_request`, `invalid_value`, `unsupported_parameter`, `unsupported_combination` | A malformed body or header; `param` names it. Errors never echo what was sent |
| 400 | `no_provider_keys` | Account chat with no usable saved key and no key header |
| 400 | `no_usable_model` | Your keys reach no model |
| 400 | `max_tokens_too_low` | A named reasoning model spent every token thinking; send a higher `max_tokens` |
| 400 | `invalid_provider_key` | Saving a key the provider rejected (nothing is stored) |
| 400 | `invalid_code` | A password reset code that's wrong, expired, used or dead, or an email with no account (always the same message) |
| 400 | `invalid_reset_token` | A password reset token that's unknown, expired or already used |
| 401 | `invalid_api_key` | A missing, unknown or revoked Triage key or session |
| 401 | `invalid_provider_key` | A provider rejected your key during chat |
| 401 | `invalid_credentials` | Log-in failed (always the same message) |
| 401 | `invalid_session` | Account endpoints need a session token: log in again |
| 403 | `incorrect_password` | The current password was wrong (password change, account deletion) |
| 403 | `model_not_allowed` | A named model on a provider you have no key for |
| 403 | `admin_key_required` | Admin endpoints (`/v1/routing-config`) |
| 404 | `model_not_found`, `not_found` | Unknown model; a request, key or provider key that isn't yours or doesn't exist |
| 409 | `email_taken`, `too_many_api_keys`, `key_unreadable` | Sign-up with a used email; the key cap; a saved key that can no longer be decrypted (save it again) |
| 429 | `account_rate_limited`, `too_many_attempts` | See "Limits"; wait `Retry-After` seconds |
| 500 | `internal_error` | A bug on our side; the response never contains internal details |
| 502 | `model_error` | The model failed while answering |
| 503 | `all_models_failed` | No model could answer; try again |
| 503 | `email_unavailable` | Password reset: the server has no email settings |
| 503 | `provider_overloaded` | Every model that was tried was too busy (e.g. Gemini's "high demand" 503); the message names the provider |
| 503 | `provider_unavailable` | Every model that was tried was busy or too slow (a mix of the two cases above and below) |
| 504 | `provider_timeout` | Every model that was tried ran out of time (its timeout in `config/routing.yaml`); the message names the provider |
| (in a stream) | `stream_interrupted` | The model failed after text was sent |

### Reference: account endpoints

All of these except sign-up, log-in and password reset need a **session token** (`Authorization: Bearer trs_...`) and accept
nothing else, not even a Triage API key.

| Endpoint | Body | Result |
|---|---|---|
| `POST /v1/auth/signup` | `email`, `password`, `display_name?` | 201 `{user, session_token, expires_at}` |
| `POST /v1/auth/login` | `email`, `password` | 200, same shape |
| `POST /v1/auth/logout` | | 204, ends this session |
| `GET /v1/auth/me` | | `{id, email, display_name, preferred_provider, created_at}` |
| `PATCH /v1/account` | `display_name?`, `preferred_provider?` | the user; only the fields sent change, `null` clears one |
| `POST /v1/account/password` | `current_password`, `new_password` | 204; every other session is ended |
| `DELETE /v1/account` | `password` | 204; deletes the user, sessions, saved keys and API keys (request rows stay, unlinked) |
| `POST /v1/auth/password-reset/request` | `email` | 202 `{message}`, the same whether or not the account exists |
| `POST /v1/auth/password-reset/verify` | `email`, `code` | 200 `{reset_token, expires_at}` |
| `POST /v1/auth/password-reset/complete` | `reset_token`, `new_password` | 200 `{user, session_token, expires_at}`, as log-in |
| `GET /v1/account/provider-keys` | | `[{provider, last4, status, status_detail, verified_at, created_at, updated_at}]` |
| `PUT /v1/account/provider-keys/{provider}` | `key` | the summary (checked with the provider; replaces any saved key) |
| `POST /v1/account/provider-keys/{provider}/verify` | | the summary after checking again |
| `DELETE /v1/account/provider-keys/{provider}` | | 204 |
| `GET /v1/account/api-keys` | | `[{id, name, public_id, created_at, last_used_at, revoked_at}]`, newest first |
| `POST /v1/account/api-keys` | `name` | 201 `{id, name, public_id, created_at, key, note}`: the only time `key` is shown |
| `DELETE /v1/account/api-keys/{id}` | | 204, revokes it |

### Resetting a forgotten password

Three steps, no session needed. The answer to step 1 is always the same 202, whether or not the email has an
account, and the email is sent after the answer, so neither the answer nor its timing tells anyone which emails
have accounts.

```bash
B=http://localhost:8000; CT="Content-Type: application/json"
# 1. Ask for a code. Its email has a 6-digit code that expires in 10 minutes.
curl -s -X POST $B/v1/auth/password-reset/request -H "$CT" -d '{"email":"you@example.com"}'
# 2. Trade the code for a reset token (rst_..., 15 minutes, single use). 5 wrong tries kill the code.
curl -s -X POST $B/v1/auth/password-reset/verify -H "$CT" -d '{"email":"you@example.com","code":"123456"}'
# 3. Set the new password. Every session is logged out, and a new one comes back, so you're logged in again.
curl -s -X POST $B/v1/auth/password-reset/complete -H "$CT" \
  -d '{"reset_token":"rst_...","new_password":"a new long password"}'
```

A new code replaces an earlier unused one, but asking again within 60 seconds sends nothing and keeps the code
you already have. A wrong code, an expired one and an email with no account all get the same 400 `invalid_code`.
Your Triage API keys keep working after a reset; revoke them separately if you think they leaked.

With an account key or session, `GET /v1/models` works, and `GET /v1/stats` and `GET /v1/requests/{id}` show only
your own account's requests, whichever of your keys made them.

**How your data is kept:**
- **Passwords:** 10 to 128 characters, hashed with argon2id.
- **Password reset codes and tokens:** only keyed hashes are stored, never the code, token or your email.
- **Sessions:** last 30 days from log-in, or 7 days unused.
- **Provider keys:** encrypted with AES-256-GCM; only their last 4 characters are visible.
- **Triage API keys:** only a keyed hash is stored, so a lost key can't be shown again. Revoke it and create a new one.
- **Never stored:** your prompts and answers (only an HMAC of the prompt). Account requests are never used to
  train the router.

## Retraining the classifier (Phase 4)

`python -m scripts.retrain` retrains the live classifier with the background labels in the request log. It
adds them to the original fit set, keeps any validation or test prompt out (matched by HMAC, so it needs the
API's `TRIAGE_HASH_SECRET`), and trains a candidate as the next model version. It then compares the candidate
with the live model on the validation set. The candidate replaces the live model only if it's no worse on quality
and clearly cheaper, or better on quality and no more expensive. Every run is recorded in
`reports/retrain_history.jsonl`, and old model files are never overwritten. `--dry-run` trains and compares but
changes nothing except the candidate's model files.

**A new version is only used after a restart.** The API server loads the classifier once at startup. When
retraining points `config/routing.yaml` at a new version, restart the server (for example, stop and rerun
`uvicorn app.api.main:app`) to start routing with it. Until then, the running server keeps the old model.

## Database

Triage keeps its request log, the demo counters and accounts in one database through SQLAlchemy Core, with the
schema managed by Alembic (`migrations/`): SQLite locally and in tests (no network needed), Postgres in
production (see "Operating Triage"). Check-and-count limits (demo, sign-in, account and API-key caps) are atomic
across instances: SQLite runs each in a `BEGIN IMMEDIATE` transaction, Postgres under a transaction-level advisory
lock. Fake mode (`TRIAGE_FAKE_MODELS=1`) always uses its own `data/fake.sqlite3` and ignores `DATABASE_URL`, so
fake requests never reach the real log.

## Operating Triage

### Environment variables

`.env.example` lists every variable with a comment; copy it to `.env` (never commit `.env`). Secrets are empty
there on purpose: the server refuses to start rather than run with a guessable value.

| Variable | Needed | Default | What it does |
|---|---|---|---|
| `TRIAGE_API_KEYS` | yes (or admin keys) | none | Server API keys for your own apps, comma-separated |
| `TRIAGE_ADMIN_KEYS` | for `/v1/routing-config` | none | Keys that may also read and change the live routing config |
| `TRIAGE_HASH_SECRET` | yes, with the database log | none | Keys every HMAC Triage stores (see "Rotating secrets") |
| `DATABASE_URL` | production | `data/triage.sqlite3` | The database: Postgres in production |
| `GROQ_API_KEY`, `GEMINI_API_KEY` | for server-key and demo traffic | none | The server's own provider keys (only providers in `server_providers`) |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | no | none | Benchmarks only; live traffic uses users' own keys for these |
| `OPENROUTER_API_KEY` | no | none | Data-collection scripts only |
| `OLLAMA_BASE_URL` | for `ollama-llama` | `http://localhost:11434/v1` | A local Ollama server |
| `TRIAGE_PUBLIC` | no | `off` | `on`: the free demo and BYOK without a Triage key |
| `TRIAGE_ACCOUNTS` | no | `off` | `on`: user accounts and account chat |
| `TRIAGE_ENCRYPTION_KEYS` | with accounts | none | Encrypts saved provider keys: `k2:<base64>,k1:<base64>` |
| `RESEND_API_KEY`, `TRIAGE_EMAIL_FROM` | for emails | none | Resend API key and sender address; without both, the reset endpoints answer 503 and no welcome email is sent |
| `TRIAGE_EMAIL_LOGO_URL` | no | none | An `https://` URL of a hosted PNG logo for emails; empty or not serving an image: a plain blue square |
| `TRIAGE_APP_URL` | for email links | none | The website (e.g. `https://gettriage.dev`, locally `http://localhost:5173`); the only place email links go. Empty: emails without links |
| `TRIAGE_PUBLIC_API_URL` | no | `https://api.gettriage.dev` | The API address shown in the welcome email's code sample |
| `TRIAGE_DEV_EMAIL_LOG` | no | `off` | `on`: print reset codes to the console instead of emailing (local development only) |
| `TRIAGE_CORS_ORIGINS` | for a website | none | Browser origins allowed to call the API |
| `TRIAGE_TRUSTED_PROXY_HOPS` | behind a proxy | `0` | Proxies that append to `X-Forwarded-For` (visitor IPs) |
| `TRIAGE_SERVER_PROVIDERS` | no | `routing.yaml`'s | e.g. `groq,gemini` on a host without Ollama |
| `TRIAGE_FAKE_MODELS` | no | `0` | `1`: fake models, no API calls (development) |
| `TRIAGE_FAKE_DB` | no | `data/fake.sqlite3` | Fake mode's own database (it never uses `DATABASE_URL`) |
| `TRIAGE_CACHE`, `TRIAGE_CACHE_PATH` | no | `on`, `data/cache/responses.sqlite3` | The response cache (local to each instance) |
| `TRIAGE_LOG` | no | `sql` | `memory`: keep the request log in memory |
| `TRIAGE_TEST_DATABASE_URL` | tests only | none | A throwaway Postgres for `pytest -m postgres` |
| `HF_HUB_OFFLINE` | no | `1` (set by the server) | The embedding model must already be in the local Hugging Face cache |

Limits live in `config/demo.yaml` and `config/accounts.yaml`; routing in `config/routing.yaml`.

### Generating secrets

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"            # a TRIAGE_API_KEYS / TRIAGE_ADMIN_KEYS key
python3 -c "import secrets; print(secrets.token_hex(32))"                # TRIAGE_HASH_SECRET
python3 -c "import base64, secrets; print(base64.b64encode(secrets.token_bytes(32)).decode())"   # an encryption key
```

Keep them in your host's secret settings (e.g. Render's environment), never in the database or the repo.

### Deploying

1. Install exactly the tested versions: `pip install -r requirements.txt`. Every dependency is pinned; to upgrade
   one, change its pin, run the tests (including `pytest -m postgres`), then commit.
2. **Migrate before the new code starts:** `DATABASE_URL=... venv/bin/alembic upgrade head`, as your host's
   pre-deploy step if it has one, or at the front of the start command (`alembic upgrade head && uvicorn ... --no-access-log`).
   The server never migrates Postgres itself, and refuses to start if the database isn't at the latest
   migration. A local SQLite file is migrated automatically when opened.
3. Start: `uvicorn app.api.main:app --host 0.0.0.0 --port $PORT --no-access-log`. `--no-access-log` is required:
   uvicorn's access log prints each client's IP address, and the privacy policy says Triage logs none (its own
   request lines carry no IP). With `TRIAGE_ACCOUNTS=on` it refuses to start without valid `TRIAGE_ENCRYPTION_KEYS`.
4. Housekeeping runs by itself, at startup and then hourly: expired cache answers (older than 24 hours) are deleted,
   and so are request log rows older than `TRIAGE_REQUEST_LOG_RETENTION_DAYS` (default 365) with their labels.

Request logs from before the database layer (`data/logs/requests.sqlite3`) aren't opened any more; copy them in
with `venv/bin/python scripts/import_sqlite_log.py <file>` (safe to re-run; the file is only read).

### Emails: password reset and welcome (Resend)

Reset codes and the welcome email are sent with [Resend](https://resend.com) (`POST https://api.resend.com/emails`,
[API reference](https://resend.com/docs/api-reference/emails/send-email)).

1. In Resend, verify the domain you'll send from (DNS records), and create an API key with sending access.
   Until a domain is verified, Resend only delivers to your own address and refuses the rest (403).
2. Set `RESEND_API_KEY` and `TRIAGE_EMAIL_FROM` (e.g. `Triage <no-reply@your-domain>`). The sender must be
   `email@domain` or `Name <email@domain>`. Surrounding whitespace and one matching pair of quotes are removed;
   anything else (such as a quote left on one end) is refused at startup with a warning naming the variable, and
   the reset endpoints answer 503 until it's fixed.
3. Optional: host the logo as a PNG (72x72, shown at 36x36) and set `TRIAGE_EMAIL_LOGO_URL` to its `https://` URL.
   At startup Triage sends one HEAD request (5 s timeout). If the URL is unusable, doesn't answer 200 with an
   `image/*` content-type, or can't be reached, it's ignored with one startup warning and emails show the plain
   blue square instead of a broken image. Restart after fixing the hosting. In Resend's
   domain settings, keep open and click tracking off: the email has no tracking of its own.
4. Set `TRIAGE_APP_URL` to the website (e.g. `https://gettriage.dev`; locally `http://localhost:5173`): every link
   in the emails goes to one of its pages (`/settings/get-started`, `/settings/provider-keys`,
   `/settings/api-keys`, and the site itself in the footer). `https://` is required, except `http://` for
   localhost. Unset or invalid: emails are sent without links or buttons, and startup logs one warning. Set
   `TRIAGE_PUBLIC_API_URL` to the API's public address for the welcome email's code sample (default
   `https://api.gettriage.dev`; a trailing `/v1` is removed).
5. Check the server's log line for each request: `{"event":"password_reset_request","outcome":"sent"}`, or
   `send_failed` with a `detail` such as `resend_403`. Nothing else about the email is logged. A failed send
   isn't retried; the user can ask again after 60 seconds.

**Welcome email.** After each successful signup (`POST /v1/auth/signup`), a background task sends one welcome
email: three setup steps (add a provider key, create an API key, point the OpenAI client at Triage) and a button to
the settings. Signup never waits for it or fails because of it. The log line is
`{"event":"welcome_email","outcome":"sent"}`, or `send_failed` with a short `detail`; a failed send isn't retried.
It's sent only at signup, once per account. Without Resend settings nothing is sent and signup works as usual;
with `TRIAGE_DEV_EMAIL_LOG=on` the console shows only `welcome email for user <id>`.

Without both variables the server still starts, logs a warning naming what's missing, and the three reset
endpoints answer 503 `email_unavailable`. Fake mode (`TRIAGE_FAKE_MODELS=1`) never emails.

**Local development:** `TRIAGE_DEV_EMAIL_LOG=on` prints `reset code for user <id>: 123456` to the server console
instead of sending anything (it also works in fake mode). The server refuses to start with it if anything looks
like a deployment: a non-SQLite `DATABASE_URL`, `TRIAGE_PUBLIC=on`, a `TRIAGE_CORS_ORIGINS` entry that isn't
localhost, or `TRIAGE_TRUSTED_PROXY_HOPS` above 0.

### Neon

- **Production:** use Neon's **pooled** connection string (its host contains `-pooler`) in `DATABASE_URL`, as
  `postgresql+psycopg://USER:PASSWORD@HOST/DB?sslmode=require`. Neon suspends idle databases: the first query
  after a pause is slower, and the server re-checks pooled connections before using them.
- **Migrations:** run `alembic upgrade head` with the **direct** connection string (the same host without
  `-pooler`); schema changes are safest without the pooler in between.
- **Tests:** point `TRIAGE_TEST_DATABASE_URL` at a separate **test branch**, never production, and run
  `venv/bin/python -m pytest -m postgres`. Each test works in its own throwaway schema. The tests switch a pooled
  host to its direct host automatically, because the pooler rejects the `search_path` option they use. A plain
  `pytest` never connects to it.

### Rotating the encryption key (saved provider keys)

1. Generate a new key and put it **first**: `TRIAGE_ENCRYPTION_KEYS="k2:<new>,k1:<old>"`. Deploy. New saves use
   `k2`; old rows still decrypt with `k1`.
2. Re-encrypt every saved key with `k2` (same environment):
   `venv/bin/python scripts/rotate_provider_keys.py --dry-run`, then without `--dry-run`. It first checks that
   every row decrypts; if any doesn't, it changes nothing and exits with an error. It works in batches (one
   transaction each), is safe to re-run, and prints only counts.
3. When it reports every row on `k2` (`by_key_after`), remove `k1`: `TRIAGE_ENCRYPTION_KEYS="k2:<new>"`. Deploy.

If every key that encrypted a row is lost, that row can't be decrypted: the user sees 409 `key_unreadable` and
must save the key again.

### Rotating TRIAGE_HASH_SECRET

Only rotate it if it leaked: it keys several HMACs at once, and changing it invalidates them all.

| Affected | Effect |
|---|---|
| Account API keys (`trk_`) | **All stop working** (401). Users must create new keys |
| Prompt HMACs in the request log | New rows don't match old ones. **Retraining:** run `scripts/retrain.py` with the secret the rows were logged under, or a validation or test prompt sent before the change won't be recognized and could leak into training |
| Response cache entries for BYOK and account requests | Become misses (they're fingerprinted per key) |
| Demo visitor ids and sign-in rate-limit counters | Start over |
| Password reset codes already emailed | Stop working (`invalid_code`); users ask for a new one |

Sessions (`trs_`), passwords and saved provider keys don't use it and keep working.

## Known limitations

- **Keys rejected during a sync-verify escalation aren't marked invalid yet.** Only rejections while routing a
  request mark a saved key `invalid`; the escalation's rejected attempts just fall back.
- No email verification: an account's email isn't confirmed at sign-up (password reset sends only to the
  address on the account).
- The server's own log shows each reset request's outcome (`sent`, `no_account`, `cooldown`), so whoever can read
  the server logs can tell whether a reset was for a real account. Clients can't.
- Session tokens are bearer tokens the website keeps in the browser, so the website must avoid third-party
  scripts (a strict Content-Security-Policy).
- The per-minute limits (demo and account) are kept in memory by each server instance.
- Streams don't report their cost in the response; fetch it with `GET /v1/requests/{id}` once the stream ends.
- `PUT /v1/routing-config` changes are kept in memory only, not persisted.

## Privacy: what the request log keeps

Every API request writes one row to the database and one JSON line to the server log. The database is
`DATABASE_URL` (Postgres in production, e.g. Neon) or, by default, the local SQLite file `data/triage.sqlite3`
(not committed). See "Database" above.

**Stored:**
- Routing and cost data: the tier and model used, predicted pass probabilities, fallbacks, tokens, costs,
  latency, status and error kind.
- `key_id`: the first 8 hex characters of the API key's SHA-256, so requests can be grouped by key.
- `prompt_hmac`: an HMAC-SHA256 of the conversation, keyed with `TRIAGE_HASH_SECRET`. It shows when the same
  prompt comes back. Without the secret, nobody can test a guessed prompt against it, as they could with a
  plain hash. The server refuses to start with the database log if the secret isn't set. Keep the secret out of
  the database and out of version control. Changing it means new rows can't be matched with old ones.
- For `model="auto"` requests that answered: the classifier's inputs, which are the 14 handcrafted features
  (`FEATURE_VERSION`) and the 384-number sentence embedding (`all-MiniLM-L6-v2`). Step 8 retrains on rows
  marked `training_eligible`.

**Never stored:** prompt text, answer text, API keys, or provider error messages. That covers the database,
the JSON lines and `GET /v1/requests/{id}`, which leaves out the features and embedding. The tests search the
raw database files and the log lines for keys, prompt text and answer text.

**Treat the log as sensitive anyway.** An embedding isn't the text, but it isn't anonymous either. Similar
prompts get similar vectors, so anyone with the database can group requests by topic. They can also find
which stored requests resemble a sentence they choose, and with enough effort recover parts of a prompt's
meaning. The handcrafted features reveal length, format and task type. Restrict who can read the file, back it
up like user data, and delete it when it's no longer needed. A later demo mode will store raw text, and only
behind a visible notice to the user. That isn't built yet.
