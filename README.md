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