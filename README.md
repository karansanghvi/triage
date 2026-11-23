# Triage

**Every LLM request, sent to the cheapest model that can answer it well.**

Triage is an OpenAI-compatible LLM router. For every request it predicts how likely a cheap, a medium and a strong
model are to answer correctly, then picks the cheapest one that clears a quality bar you set. Apps that already use
the OpenAI SDK switch to it by changing two lines.

- **Website and free demo:** [gettriage.dev](https://gettriage.dev)
- **Developer docs:** [gettriage.dev/docs](https://gettriage.dev/docs) (source: [docs/DOCS.md](docs/DOCS.md))

## Why

Teams using LLMs usually pick one model and send it everything. A strong model overpays for the many requests a
small model could handle ("What is 2 + 3?" doesn't need a frontier model); a cheap one drops quality on the requests
that needed more. On Triage's own data the cheapest model answered 44.6% of prompts correctly, the medium model 87.3%
and the strongest 97.2%, while always using the cheapest costs about a tenth of always using the strongest.

The ideal is to send each request to the cheapest model that will still get it right. The hard part is knowing that
in advance, from the prompt alone, and intuition is a poor guide: when 50 prompts were labelled by hand before any
answers were seen, the guess matched the real outcome only 34% of the time. So Triage learns routing from data.

## Results

From a pre-registered evaluation on a locked test set of 149 prompts (details in the
[case study](CASE_STUDY.md#6-evaluation)):

| | |
|---|---|
| Labelled dataset | 762 prompts, every one answered by all 3 tiers |
| Default setting | 144/149 correct (always-strongest: 145): **99.3% of the quality at 12.4% lower cost** |
| Cheap setting | 92.4% of the quality at **57.4% lower cost** |
| Perfect-router ceiling | 149/149 correct at 57% lower cost |
| Routing overhead | 2.2–3.6 ms per request |

The gap between Triage and the perfect router is in prediction, not in the routing system, so a better classifier
converts directly into savings. An AI answer checker that looked promising on validation did no better than random
routing on the locked test set, so it ships as opt-in rather than the default.

## How it works

For each request, Triage:

1. turns the prompt into features and a sentence embedding, and predicts the chance that the cheap and the medium
   model will answer well;
2. picks the cheapest model whose chance clears the **quality dial** (`X-Triage-Quality: cheap | balanced | high`,
   or a number from 0 to 1);
3. calls that model, retrying only errors that can succeed later and falling back to a backup on another provider
   if it fails (with circuit breakers, timeouts and a per-request time budget);
4. reports what it did in response headers: `X-Triage-Model`, `X-Triage-Tier`, `X-Triage-Reason`,
   `X-Triage-Cost`, `X-Triage-Saved` and more.

Asking for a specific model instead of `auto` also works: Triage calls exactly that model, with retries but no
routing, and claims no savings for it. Identical repeated requests at temperature 0 (the default) are answered from
a cache for free, streamed or not. Prompts and answers are never stored: the usage log keeps routing details, tokens and costs only.

| Component | What it does |
|---|---|
| API server | The OpenAI-compatible chat endpoint (streaming and not), usage statistics, accounts and demo status |
| Router | Features, a sentence embedding and a classifier that predicts pass chances, then the quality dial |
| Provider layer | One way to call Groq, Gemini, OpenAI and Anthropic, with their errors translated into one vocabulary |
| Reliability layer | Fallback chains, targeted retries, circuit breakers, timeouts and a per-request time budget |
| Accounts | Sign-up, password reset, encrypted provider keys, Triage API keys and rate limits |
| Web app | Landing page, chat with a routing explanation under each answer, settings and a usage dashboard |
| Evaluation pipeline | Dataset building, grading, training, simulation and pre-registered tests |

## Using Triage

There are three ways in:

| Mode | Who | Whose provider keys | Limits |
|---|---|---|---|
| Free demo | anyone, no sign-up | Triage's own Groq and Gemini keys | 10 requests per visitor per day |
| Your own keys | anyone sending their provider keys in headers | yours, only those providers | 200 requests per visitor per day |
| Account | signed-in users and their API keys | your saved, encrypted keys | 60 requests per minute, 10,000 per day |

You bring your own provider keys (OpenAI, Anthropic, Groq or Gemini) and pay your providers directly.

### With an account

1. Sign up at [gettriage.dev](https://gettriage.dev).
2. In **Settings**, save at least one provider key. It's checked with the provider, encrypted, and afterwards shown
   only by its last 4 characters.
3. Create a Triage API key for your app. Copy it when it appears: it's shown only once.
4. Point your OpenAI client at Triage and ask for `model="auto"`.

Python (`pip install openai`):

```python
from openai import OpenAI

client = OpenAI(base_url="https://api.gettriage.dev/v1", api_key="trk_...")
reply = client.chat.completions.create(
    model="auto",                                     # or a model name from GET /v1/models
    messages=[{"role": "user", "content": "What is 2 + 3?"}],
    extra_headers={"X-Triage-Quality": "balanced"},   # optional: cheap, balanced or high
)
print(reply.choices[0].message.content)
```

JavaScript (`npm install openai`):

```js
import OpenAI from "openai";

const client = new OpenAI({ baseURL: "https://api.gettriage.dev/v1", apiKey: "trk_..." });
const stream = await client.chat.completions.create({
  model: "auto",
  messages: [{ role: "user", content: "What is 2 + 3?" }],
  stream: true,
});
for await (const chunk of stream) process.stdout.write(chunk.choices[0]?.delta?.content ?? "");
```

A stream carries the routing details on its first chunk and the cost on its last one (`chunk.triage`); the cost
headers are only on non-streaming responses.

**Your own provider keys, per request.** Instead of saved keys (or without an account, up to 200 requests a day),
send them as headers: `X-OpenAI-Key`, `X-Anthropic-Key`, `X-Groq-Key`, `X-Gemini-Key`. A header replaces the saved
key for that provider on that request and is never stored.

**Preferred provider.** With keys for several providers, pick the one tried first in each tier: in **Settings →
Provider keys**, or per request with `X-Triage-Prefer-Provider: openai | anthropic | groq | gemini`. It changes which
model answers within the chosen tier, never the tier, and applies only to `model="auto"`.

Your **Dashboard** shows spend, savings per API key and recent requests. Every header, the `triage` object,
streaming, answer checking, errors and limits are in the [developer docs](https://gettriage.dev/docs)
([docs/DOCS.md](docs/DOCS.md)); running and operating Triage is in [docs/REFERENCE.md](docs/REFERENCE.md).

## Running it locally

You need Python 3.14 (what the Docker image and tests use) and Node 22.12 or newer.

### API

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt        # the server's packages plus training, evaluation and tests
cp .env.example .env                       # then fill in what you need (every variable has a comment)
python -m scripts.fetch_embedding_model    # the router's embedding model, once
```

Try it with fake models first: no provider keys and no API calls.

```bash
TRIAGE_FAKE_MODELS=1 TRIAGE_API_KEYS=dev-key TRIAGE_HASH_SECRET="$(openssl rand -hex 32)" \
  uvicorn app.api.main:app --port 8000 --no-access-log
```

```bash
curl -s http://localhost:8000/v1/chat/completions -H "Authorization: Bearer dev-key" \
  -H "Content-Type: application/json" -d '{"model":"auto","messages":[{"role":"user","content":"What is 2 + 3?"}]}'
```

For real answers, put a free Groq key ([console.groq.com](https://console.groq.com)) and a free Gemini key
([aistudio.google.com](https://aistudio.google.com)) in `.env`, and run without fake models. `groq,gemini` skips
the local Ollama models; install Ollama and pull `llama3.2:3b` and `qwen2.5:3b` if you want them.

```bash
TRIAGE_SERVER_PROVIDERS=groq,gemini TRIAGE_API_KEYS=dev-key TRIAGE_HASH_SECRET="$(openssl rand -hex 32)" \
  uvicorn app.api.main:app --port 8000 --no-access-log
```

### Website

The website talks to the API at `VITE_TRIAGE_URL` (default `http://localhost:8000`). Start the API with the
public demo on and the website's origin allowed:

```bash
TRIAGE_FAKE_MODELS=1 TRIAGE_PUBLIC=on TRIAGE_CORS_ORIGINS=http://localhost:5173 TRIAGE_API_KEYS=dev-key \
  TRIAGE_HASH_SECRET="$(openssl rand -hex 32)" uvicorn app.api.main:app --port 8000 --no-access-log
```

Then, in another terminal:

```bash
cd web
npm install
npm run dev                                # http://localhost:5173
```

Accounts (`TRIAGE_ACCOUNTS=on`) also need `TRIAGE_ENCRYPTION_KEYS`; see `.env.example` and
[docs/REFERENCE.md](docs/REFERENCE.md#operating-triage).

### Tests

```bash
python -m pytest                           # no network needed: providers are simulated
cd web && npm test && npm run build
```

## Project structure

```
triage/
├── app/            # the API: providers, reliability, router, cache, accounts, answer checker
├── config/         # models and prices, routing and fallbacks, demo and account limits
├── eval/           # dataset building, grading, training and evaluation
├── migrations/     # database schema (Alembic)
├── models/         # the trained routing classifiers
├── scripts/        # hands-on checks and maintenance
├── tests/          # Python tests
├── web/            # the website: React, TypeScript, Vite, Tailwind CSS
├── deploy/         # image smoke test, update and rollback scripts
├── CASE_STUDY.md
├── DEPLOY.md
└── docs/            # DOCS.md (the developer docs, also at /docs) and REFERENCE.md (operating Triage)
```

## Tech stack

| Layer | Technology |
|---|---|
| API | Python, FastAPI |
| Machine learning | scikit-learn, onnxruntime (all-MiniLM-L6-v2 sentence embeddings), NumPy |
| Providers | the official OpenAI, Anthropic and Google SDKs, plus OpenAI-compatible APIs for Groq |
| Data | SQLAlchemy, Alembic, SQLite locally, Postgres in production |
| Security | argon2 password hashing, AES-GCM encryption, HMAC fingerprints |
| Web | React, TypeScript, Vite, Tailwind CSS, shadcn/ui, Recharts |
| Hosting | Render (API, a Docker image built and tested by GitHub Actions), Vercel (website), Neon (database), Resend (email) |

## License

[MIT](LICENSE)
