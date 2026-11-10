# Triage

An LLM router that sends each request to the cheapest model that can handle it.

**Status:** Phase 1 complete (providers, reliability, and caching). Routing comes in Phase 2.

## What Phase 1 does

- **One function for every model.** `send_request(messages, model_config, stream=False)` works the same way for Groq, Gemini, Ollama, OpenRouter, OpenAI, and Anthropic, and always returns the same `Response`: the answer, input and output tokens, latency, actual cost, and realistic cost.
- **Streaming.** Answers can arrive piece by piece, with time-to-first-token measured.
- **A model registry in config.** `config/models.yaml` lists every model with its provider, tier, and price. Tiers are ordered by real price (low < medium < high), so routing a request down a tier always saves money. Free models carry a `price_as` price, so savings numbers stay realistic.
- **Honest costs.** Hidden reasoning ("thinking") tokens are counted as output, since providers bill them that way.
- **One error type.** Every provider's errors become a `ProviderError` with a kind: `rate_limit`, `server`, `timeout`, `connection`, `client`, or `config`.
- **Reliability layer** (`app/reliability.py`):
  - a timeout on every call
  - retries with exponential backoff and full jitter, only for rate limits (429) and server errors (5xx)
  - fallback chains per tier in `config/routing.yaml`, with backups on a different provider
  - a circuit breaker: 3 failures in a row skip a provider for 60 seconds (rate limits skip only that model, since limits are per model)
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
  uvicorn app.api.main:app --port 8000
# TRIAGE_FAKE_MODELS=1 answers with fake models (no provider keys, no API calls); TRIAGE_CACHE=off disables the cache
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
| `X-Triage-Verify` | `off` | `sync` arrives in a later step (501 for now; 400 with `stream=true`) |

Anything else is a 400 that names the header.

**Response headers:** `X-Triage-Request-Id`, `X-Triage-Model` (the model that answered), `X-Triage-Tier`,
`X-Triage-Reason`, `X-Triage-Fallback` (`primary->backup`, or `none`), `X-Triage-Cache` (`hit`, `miss`,
`bypass`), `X-Triage-Verified`, `X-Triage-Escalated`, and on non-streaming responses `X-Triage-Cost` and
`X-Triage-Saved` (US dollars at paid prices). On a stream, the headers go out once the first token exists,
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

## Privacy: what the request log keeps

Every API request writes one row to a SQLite file (`TRIAGE_LOG_DB`, default `data/logs/requests.sqlite3`,
not committed) and one JSON line to the server log.

**Stored:**
- Routing and cost data: the tier and model used, predicted pass probabilities, fallbacks, tokens, costs,
  latency, status and error kind.
- `key_id`: the first 8 hex characters of the API key's SHA-256, so requests can be grouped by key.
- `prompt_hmac`: an HMAC-SHA256 of the conversation, keyed with `TRIAGE_HASH_SECRET`. It shows when the same
  prompt comes back. Without the secret, nobody can test a guessed prompt against it, as they could with a
  plain hash. The server refuses to start with the SQLite log if the secret isn't set. Keep the secret out of
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
