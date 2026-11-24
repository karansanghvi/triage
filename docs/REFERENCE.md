# Triage reference

The details behind the [README](../README.md) for running and operating Triage: the demo and bring-your-own-key
modes, the account endpoints, retraining, the database, and running Triage yourself (environment variables,
secrets, emails, key rotation). Deploying is in [DEPLOY.md](../DEPLOY.md).

**Using the API from an app?** The [developer docs](DOCS.md) (also at
[gettriage.dev/docs](https://gettriage.dev/docs)) are the reference for every request and response header, the
`triage` object, streaming, answer checking, provider keys, every error code a caller can get, and the limits.
This file links there instead of repeating them, and adds what an operator needs on top.

## The OpenAI-compatible API

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

**Request headers** (all optional): `X-Triage-Quality`, `X-Triage-Task`, `X-Triage-Verify`,
`X-Triage-Prefer-Provider` and the provider key headers; their values are in the
[developer docs](DOCS.md#2-choosing-quality). A value Triage doesn't accept is a 400 that names the header; headers
Triage doesn't know are ignored. The quality presets are `quality:` in `config/routing.yaml`, the answer-checking
policy is `verification:` in the same file.

**Response headers and the `triage` object:** listed with their meanings in the
[developer docs](DOCS.md#3-reading-the-response). The `triage` object is built from the headers, so the two can't
disagree. It has `request_id`, `model`, `tier`, `p_low`, `p_medium`, `tau`, `reason`, `cost`, `strong_cost`, `saved`,
`cache`, `fallback`, `mode`, `routed_by` and `preferred_provider`, with `null` for a header that was left out. It
does **not** have the answer-checking results: `X-Triage-Verified`, `X-Triage-Check` and `X-Triage-Escalated` are
headers only. `Saved` assumes the strong model would have used the same number of tokens, so it's an estimate.
It's only sent for `model: "auto"`: when you name a model, the choice was yours, so Triage claims no saving for it
(here or in any stats).

**Streams report their cost.** On a stream, the headers go out once the first token exists, so model, tier and
fallback are real, but cost isn't known yet: `X-Triage-Cost`, `-Strong-Cost` and `-Saved` are left out rather than
guessed. The cost arrives in the stream itself: the `triage` object has the routing on the first chunk and is
complete (with `cost`, `strong_cost` and `saved`) on the `finish_reason` chunk. Both chunks keep `choices[0]`, and
the openai SDKs expose it as `chunk.triage`. The same figures are in `GET /v1/requests/<request id>` once the
stream ends. If the client disconnects part-way (a Stop button), Triage closes the model's stream so it stops
generating, and logs an estimated cost for what it produced (`tokens_estimated: true`, status `stopped`). Stopped
requests count in spend and savings, but not as errors.

**Cache:** every `temperature` 0 request is cached (the default temperature is 0), **streamed or not**: streamed
and non-streamed requests share entries, so either can be answered from the other's saved answer. A hit says
`X-Triage-Cache: hit` and costs nothing; `bypass` means `temperature` above 0. Entries expire after 24 hours, and
BYOK and account answers are kept per provider key (an HMAC of it), never shared between keys. `TRIAGE_CACHE=off`
turns it off.

**Answer checking (`X-Triage-Verify: sync`):** for the tier that answered, `verification.sync.tiers` in
`config/routing.yaml` lists the checks and where a bad answer escalates: low gets the free format check, medium
the format check and then the AI checker (`config/checker.yaml`, which runs on `groq-small`), and high is never
checked. `X-Triage-Check` says which check decided:

| Value | When |
|---|---|
| `format` | The format check decided |
| `ai_checker` | The AI checker decided |
| `ai_checker_error` | The AI checker failed or gave no verdict; `on_error: escalate` sent it to a stronger model |
| `skipped_confident` | Medium's `skip_check_above` (0.90): the classifier's `p_medium` was at least that, so the AI check was skipped |
| `skipped_short` | Medium's `min_answer_tokens`: the answer was shorter (currently 0, so never) |
| `skipped_no_key` | A BYOK or account request with **no Groq key**: the AI checker runs on Groq, so it's skipped (the format check and escalation still run) |
| `none` | No check ran: verification off, or the high tier answered |

**Other endpoints:** `GET /v1/models` (with prices; only models a caller can reach: every model on a provider that takes
the caller's own key, plus the server's own providers in `server_providers`, so no Ollama model on a host without
it. Models marked `retired: true` in `config/models.yaml` stay there for evaluation but are neither listed nor
served: naming one is a 404 `model_not_found`), `GET /v1/stats` (savings for your key),
`GET /v1/stats/public` (no key: live totals across everyone's routed requests, once a range has 100 of them),
`GET /v1/requests/{id}` (one of your own requests; with public access on, a request made without a key can be
read without one, by its id, for 10 minutes (`TRIAGE_KEYLESS_LOOKUP_MINUTES`), but only its outcome and cost), and for admin keys `GET`/`PUT /v1/routing-config`, which
changes the quality presets and backup models without a restart. Those changes are kept in memory only. The
first model of each tier can't be changed, because the classifier predicts for that exact model. Admin keys can also
read `GET /v1/admin/forwarded-for`: their own request's proxy headers, to set `TRIAGE_TRUSTED_PROXY_HOPS` behind a
host that doesn't document them (DEPLOY.md, section 8).

**Settings:** `temperature` and `max_tokens` are honoured. The classifier learned from temperature 0 and
max_tokens 4000, so `auto` requests with other values are logged with `settings_match_training: false`.

**Reliability:** a failing model falls back to the tier's backup, but only before the first token is sent.
After that, the stream ends with an error event (the openai SDKs raise `APIError`). A stream whose first token
doesn't arrive within the limit in `routing.yaml` (`streaming:`) fails over. A long answer that streams steadily
is never cut off.

## Public access: the free demo and BYOK

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
- 4,000 prompt characters (only the demo: your own provider keys, accounts and Triage API keys have no limit).
  `max_tokens` is lowered to 2,048, and the response says so in `X-Triage-Max-Tokens`.
- No sync verification.

**What counts:** one request per message sent, however many model calls it takes (retries, fallbacks; those count
toward the 600 model calls). Cache hits and stopped answers count. A request that fails on the server's side is
given back: every model failed, timed out or was overloaded (503/504), a model failed mid-answer (502, or a stream
broken part-way), or no model could run it (`no_usable_model`, `model_not_allowed`, `model_not_found`, `truncated`).
Its model calls and tokens still count toward the demo-wide budgets. Requests refused before they start (400, 413,
429) never count, and neither do `/healthz`, `/readyz` or `GET /v1/demo/status`.

Refusals are 429s with `Retry-After`: `demo_limit_reached` and `demo_busy` suggest adding a provider key, and
`demo_paused` means the demo is paused until 00:00 UTC. A prompt over the limit is a 413 `demo_prompt_too_long`,
and `X-Triage-Verify: sync` in the demo is a 400 `demo_unsupported`. Public BYOK (provider key headers, no Triage
key) over its 200 a day per visitor is a 429 `public_limit_reached`.

Demo responses carry `X-Triage-Demo-Remaining`, `-Limit` and `-Resets`. `GET /v1/demo/status` (public) returns
the visitor's allowance and whether the demo is paused, so a website can show it before the first request.

**Visitors** are identified by IP. Set `TRIAGE_TRUSTED_PROXY_HOPS` to the number of proxies in front of the
server (Render: `3`, measured in DEPLOY.md section 8); the default 0 ignores `X-Forwarded-For`, which any client
can fake. IPs are stored only as an HMAC with `TRIAGE_HASH_SECRET` (IPv6 grouped by /64), and the request log
only ever sees `key_id` `demo` or `public`.

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
    extra_headers={"X-Triage-Quality": "balanced"},   # optional; see docs/DOCS.md
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

### Headers, limits and error codes

The headers an account request can send and gets back, the limits and every error code are in the
[developer docs](DOCS.md): [headers](DOCS.md#3-reading-the-response),
[provider keys](DOCS.md#6-using-your-own-provider-keys), [errors](DOCS.md#7-errors) and
[limits](DOCS.md#8-limits). Account requests say `X-Triage-Mode: account`. Errors use OpenAI's shape,
`{"error": {"message", "type", "param", "code"}}`, and never echo what was sent.

Where the limits live: chat (60 a minute, 10,000 a day per account) in `config/accounts.yaml`; sign-up, log-in,
password reset and key changes in `app/accounts/ratelimit.py`; 20 active Triage API keys in
`app/accounts/api_keys.py`. The per-minute chat limit is kept in memory by each instance, the rest in the database.

Codes the developer docs don't list, because only an operator meets them:

| Status | `code` | When |
|---|---|---|
| 400 | `invalid_json`, `invalid_config` | `PUT /v1/routing-config` with a body that isn't JSON, or a change that fails validation (nothing changes) |
| 403 | `admin_key_required` | Admin endpoints (`/v1/routing-config`, `/v1/admin/forwarded-for`) without an admin key; account keys are never admins |
| 503 | `encryption_unavailable` | Saving a provider key on a server without encryption keys. With `TRIAGE_ACCOUNTS=on` the server refuses to start without valid `TRIAGE_ENCRYPTION_KEYS`, so a correctly started server never returns it |

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

## Retraining the classifier

`python -m scripts.retrain` retrains the live classifier with the background labels in the request log. It
adds them to the original fit set, keeps any validation or test prompt out (matched by HMAC, so it needs the
API's `TRIAGE_HASH_SECRET`), and trains a candidate as the next model version. It then compares the candidate
with the live model on the validation set. The candidate replaces the live model only if it's no worse on quality
and clearly cheaper, or better on quality and no more expensive. Every run is recorded in
`reports/retrain_history.jsonl`, and old model files are never overwritten. `--dry-run` trains and compares but
changes nothing except the candidate's model files.

**A new version is only used after a restart.** The API server loads the classifier once at startup. When
retraining points `config/routing.yaml` at a new version, restart the server (for example, stop and rerun
`uvicorn app.api.main:app --no-access-log`) to start routing with it. Until then, the running server keeps the old model.

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
| `TRIAGE_HASH_SECRET` | yes, with the database log | none | Keys every HMAC Triage stores (see "Rotating TRIAGE_HASH_SECRET") |
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
| `TRIAGE_KEYLESS_LOOKUP_MINUTES` | no | `10` | How long a keyless (demo or public BYOK) request's outcome and cost can be read by id without a key |
| `TRIAGE_PUBLIC_STATS_MIN_REQUESTS` | no | `100` | Routed requests a range needs before `GET /v1/stats/public` shows figures. Keep 100 in production; lower it locally (e.g. `1`) to see the full Overview tab |
| `TRIAGE_TEST_DATABASE_URL` | tests only | none | A throwaway Postgres for `pytest -m postgres` |
| `HF_HUB_OFFLINE` | no | `1` (set by the server) | The embedding model must already be in the local Hugging Face cache |

Limits live in `config/demo.yaml` and `config/accounts.yaml`; routing in `config/routing.yaml`.

### Generating secrets

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"            # a TRIAGE_API_KEYS / TRIAGE_ADMIN_KEYS key
python3 -c "import secrets; print(secrets.token_hex(32))"                # TRIAGE_HASH_SECRET
python3 -c "import base64, secrets; print(base64.b64encode(secrets.token_bytes(32)).decode())"   # an encryption key
```

Keep them in your host's secret settings (the server's `.env`, see [DEPLOY.md](../DEPLOY.md)), never in the database or the repo.

### Deploying

The API runs on Render's free plan (`render.yaml`): GitHub Actions builds and tests the Docker image for x86_64 and
ARM on every push to `main`, publishes it to GitHub's container registry, then has Render deploy exactly that image.
Render handles HTTPS; the free plan sleeps when idle, and the website wakes it. Your own VM (a Google Cloud e2-micro or
an Oracle Cloud ARM VM, with Caddy for HTTPS) is the documented alternative. The whole setup, every variable, updates and rollback are in
[DEPLOY.md](../DEPLOY.md). What any host needs:

1. Install exactly the tested versions: `pip install -r requirements.lock` on Linux (every package pinned; only
   what the server runs, so no torch: the embedding model runs on onnxruntime). `requirements.txt` lists the
   server's direct dependencies and `requirements-dev.txt` adds training, evaluation and tests. To upgrade one,
   change its pin, run the tests (including `pytest -m postgres`), regenerate the lock (DEPLOY.md), then commit.
2. **Migrations run themselves at startup, only when needed:** the server reads `alembic_version` with one query
   and compares it with the latest revision (recorded in `migrations/HEAD` when the image is built). If the
   database is behind, it upgrades it before serving (Postgres on `DATABASE_URL_UNPOOLED`, the direct
   connection); if it's newer than the code, it refuses to start. `alembic upgrade head` still works by hand.
3. Start: `uvicorn app.api.main:app --host 0.0.0.0 --port $PORT --no-access-log`. `--no-access-log` is required:
   uvicorn's access log prints each client's IP address, and the privacy policy says Triage logs none (its own
   request lines carry no IP). With `TRIAGE_ACCOUNTS=on` it refuses to start without valid `TRIAGE_ENCRYPTION_KEYS`.
4. Housekeeping runs by itself, at startup and then on a timer: expired cache answers (older than 24 hours) are
   deleted hourly, and request log rows older than `TRIAGE_REQUEST_LOG_RETENTION_DAYS` (default 365) are deleted with
   their labels once a day (each run wakes a serverless database, so not more often).

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
- **Migrations:** set `DATABASE_URL_UNPOOLED` to the **direct** connection string (the same host without
  `-pooler`): the server migrates on it when it starts behind, and `alembic upgrade head` by hand uses it too.
  Schema changes are safest without the pooler in between.
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
