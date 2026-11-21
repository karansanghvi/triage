# Deploying Triage

| Piece | Where | Config in this repo |
|---|---|---|
| API (FastAPI) | Render, Python web service, Standard plan (1 CPU, 2 GB), region Oregon | `render.yaml`, `requirements.lock` |
| Database | Neon Postgres (pooled connection for the app, direct for migrations) | `migrations/` (Alembic) |
| Website (React) | Vercel, root directory `web/` | `web/vercel.json` |
| Email | Resend | environment variables only |

Never put real values in this file, in `render.yaml` or in the repo. Every secret below goes into the Render or
Vercel dashboard.

---

## 1. Neon (database)

1. Create a project in **AWS US West 2 (Oregon)**, the same region as the Render service.
2. From **Connect**, copy two connection strings for the same database and role:
   - **Pooled** (host contains `-pooler`): this is `DATABASE_URL`.
   - **Direct** (no `-pooler`): this is `DATABASE_URL_UNPOOLED`, used only for migrations.
3. Change the scheme from `postgresql://` to `postgresql+psycopg://` in both, and keep `?sslmode=require`.

**Prepared statements are off automatically on the pooled host.** Neon's pooler (PgBouncer, transaction mode) can
send each transaction to a different server connection, where psycopg's prepared statements don't exist. Triage
turns them off when the host contains `-pooler` (`TRIAGE_DB_PREPARED_STATEMENTS=auto`, the default; `on` or `off`
overrides it). Migrations run on the direct connection, as Neon recommends.

## 2. Render (API)

1. **New > Blueprint**, pick this repository. Render reads `render.yaml`: build command, start command, health
   check (`/healthz`), region and every variable name. It asks for each `sync: false` variable; fill them in from
   section 4.
2. The blueprint sets the **Standard** plan (1 CPU, 2 GB); section 3 explains why.
3. Deploys are manual (`autoDeploy: false`): deploy from the dashboard when `main` is ready.

What the blueprint does:

- **Build:** `pip install -r requirements.lock` (every package at the exact tested version, CPU-only torch), then
  `python -m scripts.fetch_embedding_model`, which downloads the embedding model (pinned revision, ~90 MB) into
  `.hf-cache/` inside the app and proves it loads with the network off.
- **Start:** `alembic upgrade head` (on `DATABASE_URL_UNPOOLED`), then uvicorn with **one worker**. The demo,
  sign-in and per-minute limits, the circuit breakers and the response cache live in the process's memory; more
  workers or instances would each keep their own. `--no-access-log` keeps client IPs out of the logs, as the privacy
  policy promises.
- **Runtime:** `HF_HUB_OFFLINE=1`: the server never downloads anything. If the model files or the classifier files
  are missing, it refuses to start with a message naming what's missing and how to fix it.
- **Health check:** `GET /healthz` returns 200 when the classifier and embedding model are loaded and the database
  answered `SELECT 1` within 2 seconds, else 503. The database result is cached for 60 seconds, so frequent checks
  don't each hit Neon and a brief network blip doesn't flip the service up and down. The body holds only status
  words; it never calls a model provider.

  Note: a check every 60 seconds still counts as database activity, so Neon's compute won't scale to zero while the
  API is running (Neon suspends after 5 idle minutes by default).

## 3. Memory

Measured in Linux containers with the production setup (`requirements.lock`, one worker, `MALLOC_ARENA_MAX=2`, one
torch thread, network off), startup plus 8 routed requests including two 18,000-character prompts:

| Platform | After startup | Peak |
|---|---|---|
| Linux aarch64 (native) | 620 MB | 659 MB |
| Linux x86_64 (emulated on Apple silicon; includes emulator overhead) | 694 MB | 745 MB |

Where it goes (Linux aarch64): the app's own imports (FastAPI, SQLAlchemy, provider SDKs) ~130 MB, torch ~200 MB,
transformers ~25 MB, the embedding model ~160 MB, embedding a long prompt ~60 MB.

**Triage deploys on Render's Standard plan (2 GB).** The 512 MB plans don't fit.

Running the embedding model on ONNX Runtime instead of torch would cut about 200 MB, but that still leaves the peak
too close to 512 MB for the Starter plan. It's an optional future optimization, not a path to a smaller plan.

Already in place: CPU-only torch, one torch thread (`TRIAGE_TORCH_THREADS`, `OMP_NUM_THREADS`), `MALLOC_ARENA_MAX=2`,
long prompts embedded 4 chunks at a time, the classifier run as numpy instead of scikit-learn, and the embedding
model run on plain `transformers` instead of sentence-transformers (which loads scikit-learn, scipy, pandas and
datasets). Tests check both against the originals: classifier probabilities within 1e-9, embeddings within 1e-6,
and identical routing decisions at every quality preset.

## 4. Environment variables

### Render

| Name | Purpose | Example format | Notes |
|---|---|---|---|
| `DATABASE_URL` | The app's database: Neon **pooled** string | `postgresql+psycopg://USER:PASSWORD@ep-xxx-pooler.us-west-2.aws.neon.tech/neondb?sslmode=require` | Secret |
| `DATABASE_URL_UNPOOLED` | Migrations only: Neon **direct** string | `postgresql+psycopg://USER:PASSWORD@ep-xxx.us-west-2.aws.neon.tech/neondb?sslmode=require` | Secret |
| `TRIAGE_DB_PREPARED_STATEMENTS` | Prepared statements | `auto` (default), `on`, `off` | Leave empty |
| `TRIAGE_HASH_SECRET` | Keys prompt fingerprints, cache keys, visitor ids and account API key hashes | 64 hex characters | **Generate fresh.** Changing it later invalidates every account API key |
| `TRIAGE_ENCRYPTION_KEYS` | Encrypts saved provider keys (AES-256-GCM) | `k1:<base64 of 32 bytes>` | **Generate fresh.** Required with accounts on. Rotation: `k2:<new>,k1:<old>`, then `scripts/rotate_provider_keys.py` |
| `TRIAGE_API_KEYS` | Bearer keys for your own apps | `key1,key2` | **Generate fresh.** Optional |
| `TRIAGE_ADMIN_KEYS` | Keys that may read and change the live routing config | `key1` | **Generate fresh.** Optional; keep separate from API keys |
| `GROQ_API_KEY` | The server's Groq key (demo and server-key requests) | `gsk_...` | Secret. A production key, not your development one |
| `GEMINI_API_KEY` | The server's Gemini key | `AIza...` | Secret |
| `TRIAGE_PUBLIC` | The free demo and bring-your-own-key chat without a Triage key | `on` | |
| `TRIAGE_ACCOUNTS` | User accounts | `on` | Needs `TRIAGE_ENCRYPTION_KEYS` |
| `TRIAGE_CORS_ORIGINS` | Browser origins allowed to call the API | `https://triage.example.com,https://www.triage.example.com` | Exactly the website's origins, no trailing slash |
| `TRIAGE_TRUSTED_PROXY_HOPS` | How many proxies append to `X-Forwarded-For` | `1` | See section 6 |
| `TRIAGE_APP_URL` | The website, for links in emails | `https://triage.example.com` | https only |
| `TRIAGE_PUBLIC_API_URL` | The API address shown in the welcome email | `https://api.triage.example.com` | https only |
| `RESEND_API_KEY` | Resend, sending access | `re_...` | Secret |
| `TRIAGE_EMAIL_FROM` | Sender, on a domain verified in Resend | `Triage <no-reply@triage.example.com>` | No quotes around the value |
| `TRIAGE_EMAIL_LOGO_URL` | Optional logo in emails | `https://triage.example.com/email/logo.png` | |
| `TRIAGE_REQUEST_LOG_RETENTION_DAYS` | Request-log retention | empty (365) | Keep the default; the privacy policy says 12 months |
| `TRIAGE_PUBLIC_STATS_MIN_REQUESTS` | Overview tab threshold | empty (100) | Keep the default in production |
| `TRIAGE_KEYLESS_LOOKUP_MINUTES` | How long a keyless request's cost can be read back | empty (10) | |
| `PYTHON_VERSION`, `HF_HOME`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `MALLOC_ARENA_MAX`, `OMP_NUM_THREADS`, `TRIAGE_TORCH_THREADS`, `TRIAGE_SERVER_PROVIDERS` | Build and runtime mechanics | set in `render.yaml` | Not secret; don't change |

Never set in production: `TRIAGE_FAKE_MODELS`, `TRIAGE_FAKE_DB`, `TRIAGE_DEV_EMAIL_LOG`, `TRIAGE_TEST_DATABASE_URL`,
`TRIAGE_CACHE_PATH`, `TRIAGE_LOG`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `OLLAMA_BASE_URL`
(paid providers are used only with a user's own key; Ollama doesn't run on Render).

### Vercel

| Name | Purpose | Example format | Notes |
|---|---|---|---|
| `VITE_TRIAGE_URL` | The API's address, without `/v1` | `https://api.triage.example.com` | Read at **build** time: redeploy after changing it |

Project settings: **Root Directory** `web`; keep **"Include files outside the root directory in the Build Step"**
on (the build copies `CASE_STUDY.md`, `docs/` and `reports/` images from the repo root); Node.js 22 or newer.
`web/vercel.json` sets the build and sends every route to `index.html`, so `/chat/<id>`, `/settings/api-keys` and
`/case-study` load on refresh; real files (`/assets/...`, `/case-study-media/...`) are served first.

### Generating the secrets

Run each on your own machine and paste the output into the dashboard. Generate them fresh for production; never
reuse a development value.

```bash
# TRIAGE_HASH_SECRET
python3 -c "import secrets; print(secrets.token_hex(32))"

# TRIAGE_ENCRYPTION_KEYS (prefix the output with "k1:")
python3 -c "import base64, secrets; print(base64.b64encode(secrets.token_bytes(32)).decode())"

# TRIAGE_API_KEYS and TRIAGE_ADMIN_KEYS (one per key)
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

Keep a copy of `TRIAGE_HASH_SECRET` and `TRIAGE_ENCRYPTION_KEYS` somewhere safe (a password manager): losing the
encryption keys makes every saved provider key unreadable.

## 5. Resend (email)

1. Add and verify the sending domain (the DNS records Resend lists).
2. Create an API key with sending access only: `RESEND_API_KEY`.
3. Set `TRIAGE_EMAIL_FROM` to an address on that domain. A bad sender is caught at startup with a warning naming
   the variable.

## 6. Client IPs behind Render's proxy

The demo's per-visitor limit and sign-in rate limits use the visitor's IP (stored only as an HMAC). Behind a proxy,
the server sees the proxy's address, and the client's arrives in `X-Forwarded-For`. `TRIAGE_TRUSTED_PROXY_HOPS=N`
takes the N-th address from the right, so a value a client adds itself (further left) can't choose the address.

Render's documentation doesn't state exactly how its proxy builds the header, so **start with `1`** and check it
after the first deploy:

1. **Not too high:** send one demo chat message from your computer, so your allowance is below the maximum. Then
   compare `curl -s https://API/v1/demo/status` with
   `curl -s -H "X-Forwarded-For: 203.0.113.7" https://API/v1/demo/status`. If the second shows a fresh, full
   allowance, the made-up address was trusted: lower the value. Both should show the same `remaining`.
2. **Not too low:** use one demo request from your computer, then open the demo from a phone on mobile data. If the
   phone's allowance is already reduced, every visitor shares one address (Render's): raise the value.

`0` is never right behind a proxy: every visitor would share the proxy's allowance.

## 7. Smoke test after each deploy

- [ ] `GET https://API/healthz` returns 200 with `"status": "ok"`.
- [ ] Render logs: no "refusing to start", no warnings about the email sender or logo.
- [ ] Website loads; refresh `/case-study`, `/chat` and `/settings/api-keys` directly: each loads (no 404).
- [ ] Browser console on the website: no CORS errors.
- [ ] Demo chat (logged out): an answer streams in, the routing panel shows model, tier and cost.
- [ ] Press Stop during a long answer: it stops and later shows an estimated cost.
- [ ] Refresh a long answer with a table: it renders fully (no "stopped part-way").
- [ ] Sign up: the welcome email arrives, its links point at the website.
- [ ] Log out, "Forgot password": the code arrives and resets the password.
- [ ] Save a provider key in Settings: it's checked and shown by its last 4 characters; chat uses it.
- [ ] Create a Triage API key and call the API with the OpenAI SDK (`model="auto"`): 200, `X-Triage-*` headers.
- [ ] Dashboard: Benchmark shows the test results; My usage shows the requests just made.
- [ ] Client-IP check (section 6).
- [ ] Neon: the tables exist (`alembic_version` at the latest revision).

## 8. Maintenance

**Updating dependencies.** Change a pin in `requirements.txt`, reinstall locally, run the tests
(`set -o pipefail; venv/bin/python -m pytest` and `-m postgres`), then regenerate `requirements.lock` on Linux
x86_64 with your tested versions as constraints:

```bash
venv/bin/pip freeze --exclude pip | grep -v -i -E "^(torch|setuptools)==" > /tmp/constraints.txt
docker run --rm --platform linux/amd64 -v "$PWD:/src" -v /tmp/constraints.txt:/c.txt python:3.14.7-slim sh -c '
  pip install -q torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu &&
  pip install -q -r /src/requirements.txt -c /c.txt && pip check &&
  pip freeze --exclude pip --exclude setuptools' | sort -f
```

Put that output under the header lines of `requirements.lock` (they keep the CPU index line).

**Retraining.** A retrained classifier gets new `.joblib`/`.json`/`.npz` files. Commit all three for the version
`config/routing.yaml` names, and un-ignore its `.joblib` in `.gitignore` the way v2's are.

**npm audit.** `npm audit --omit=dev` in `web/` reports 6 moderate findings and nothing high or critical. They come
from two build-time tools, not code shipped to browsers: `minimatch` (through the shadcn CLI) and
`postcss-selector-parser` (through `@tailwindcss/typography`). No action needed; recheck when updating those packages.
