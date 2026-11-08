# Phase 1: Providers, reliability, and caching

## Summary

Phase 1 built the plumbing every later phase depends on: one function that can call any model, a reliability layer that retries, falls back and skips unhealthy providers, and an exact-match cache. The work ran from 2026-09-29 to 2026-10-12 and ended with commit `77112a7` ("Done with Phase 1 with end-to-end check and README file").

Both "done when" goals are met in code. `send_request(messages, model_config, stream=False)` calls models on six providers through one interface and returns the same `Response` from each. Automatic fallback when a provider fails is covered by unit tests that simulate failures. The live outage demo (`scripts/final_check.py`, which asks you to quit Ollama mid-run) exists, but its output was never saved and it was not re-run for this report, so the live demo itself cannot be confirmed here. The Phase 1 test files had 36 test functions; today the same four files have 47 test cases, all passing, and the full suite passes 397.

## What was built

| Piece | File | Purpose |
|---|---|---|
| Environment setup | `.env.example`, `.gitignore`, `scripts/check_setup.py` | Slots for Groq, Gemini, OpenRouter, OpenAI and Anthropic keys plus the Ollama URL; `.env` is never committed |
| Provider check | `scripts/check_providers.py` | Confirms each configured key works |
| Model registry | `config/models.yaml`, `app/registry.py` | Each model's provider, model ID, tier, actual price and `price_as` paid price; rejects unknown tiers, duplicate names and leftover placeholders |
| Unified interface | `app/providers/base.py`, `app/providers/common.py` | `send_request()` returns text, model name and ID, provider, tokens, latency, actual cost, realistic cost, finish reason and time to first token; every provider error becomes one `ProviderError` with a kind |
| Adapters | `app/providers/openai_provider.py` and five small adapters | One OpenAI-format adapter shared by OpenAI, Groq, Ollama and OpenRouter (the last three are 13-line wrappers); separate adapters for Gemini and Anthropic |
| Reliability | `app/reliability.py`, `config/routing.yaml` | Timeouts, retries with backoff and jitter, per-tier fallback chains, circuit breaker, fallback log |
| Cache | `app/cache.py` | SQLite exact-match cache for temperature-0 requests, with expiry and hit tracking |
| Demos | `scripts/try_models.py`, `try_reliability.py`, `try_cache.py`, `final_check.py` | Hands-on checks with real requests |

The Phase 1 lineup had six models. The low tier was `llama3.2:3b` and `qwen2.5:3b` on local Ollama plus `gpt-oss-20b` on Groq, the medium tier was `gpt-oss-120b` on Groq, and the high tier was Gemini 3.8 Flash with Groq's preview Qwen3.8 27B as a backup. Tiers were ordered by paid price. Free and local models carried a `price_as` price so savings figures would reflect what the same traffic would cost on a paid plan.

## How reliability works

A request names a tier, and `ReliableClient.send()` walks that tier's fallback chain from `routing.yaml`. In Phase 1 the chains were `[ollama-llama, groq-small, ollama-qwen]` for low, `[groq-large, gemini-flash]` for medium and `[gemini-flash, groq-qwen]` for high, with each backup on a different provider from the model before it. Every call gets a 30-second timeout unless the caller sets one.

Only rate limits (429) and server errors (5xx) are retried, because a bad key or a malformed request will never succeed on a second try; those move straight to the next model. A model gets up to three tries. Before each retry the client waits a random time between zero and a limit that doubles from 0.5 seconds and is capped at 8 seconds ("full jitter"), so many failing requests don't retry at the same moment. The OpenAI and Anthropic client libraries have their own retries turned off (`max_retries=0`), so this layer alone decides when to retry.

The circuit breaker counts consecutive failures. Server errors, timeouts and connection failures count against the whole provider, while rate limits count against the single model, because Groq limits each model separately. After three failures in a row, that provider or model is skipped for 60 seconds. When the cooldown ends, one request is let through: success resets the count, and another failure trips the breaker again. Cache hits never count as successes, since they say nothing about the provider's health. Every fallback is written to the log and kept in the client's `fallback_events` list. A stream that fails before its first piece also falls back.

## How caching works

The cache wraps `send_request()` rather than changing it, and plugs into the reliability layer through its `send_fn` hook. Only temperature-0, non-streaming requests are cached, since only then is a saved answer a fair substitute for a fresh one. The key is a SHA-256 hash of everything that affects the answer: a cache version number, the messages, the model, temperature and token limit. Entries live in a SQLite file under `data/cache/` and expire after 24 hours. Empty answers and answers cut off by the token limit are never saved. A hit returns the saved answer with zero cost, the lookup time as its latency, and the avoided cost recorded as `cache_saved_cost`. Hits, misses and money saved are counted per process and logged.

## Evidence it works

| Test file | Test functions at Phase 1 | Test functions today |
|---|---|---|
| `tests/test_registry.py` | 8 | 8 |
| `tests/test_send_request.py` | 6 | 11 |
| `tests/test_reliability.py` | 12 | 12 |
| `tests/test_cache.py` | 10 | 11 |

Today these files run 47 test cases (some tests are parametrized), and all pass. None of them need a network connection. The reliability tests cover each behaviour described above with simulated failures: a rate limit retried until it succeeds, a client error falling back without a retry, fallback after retries run out, the breaker skipping a provider and recovering, the breaker skipping every model on a provider, a rate limit skipping only one model, all models failing, capped and randomized backoff, and a stream falling back before its first piece. The functions added since Phase 1 cover options introduced in Phase 2.

The cache demo was re-run for this report against the medium tier, which is now `gpt-oss-20b`. The first temperature-0 request was a miss (486 ms, $0.000027 at paid prices), and the identical second request was a hit (0.1 ms, $0). Two requests at temperature 0.7 both went to the model, as designed. That's one hit and one miss among the cacheable requests, $0.000027 saved.

`scripts/final_check.py` is the end-to-end demo for the "done when" goal. It sends a request to every model and every tier, then asks you to quit Ollama, sends four low-tier requests that must all be answered by another provider, and finally checks that Ollama is used again after the 60-second cooldown. It was not run for this report because it is interactive and calls every model, and no saved output exists. It is also out of date: the low tier now starts with a Groq model, so its recovery check, which expects Ollama to answer again, would fail even though the fallback works.

## Limitations

The registry's `avg_latency_ms` field is 0 for every model, so the latency figures the plan asked for were never recorded. Fallback events, breaker state and cache counters live only in memory and in log lines; nothing persists them for a dashboard. Only Groq, Gemini and Ollama are configured on this machine. The OpenRouter, OpenAI and Anthropic adapters exist, but there's no evidence in the repo that they ever served a live request, and the unit tests use fake Groq and Gemini clients. The cache matches exact text only, so a reworded question is always a miss. Everything ran on free tiers, whose limits turned out to shape later phases more than anything in this layer.

## Decisions

| Decision | Choice | Reason |
|---|---|---|
| What to retry | Only 429 and 5xx | Other errors won't succeed on a second try |
| Backoff | Exponential with full jitter, capped at 8 s | Spreads out retries from many failing requests |
| Breaker scope | Per provider for outages, per model for rate limits | Groq rate-limits each model separately |
| Library retries | Off | One layer decides retries, so attempts aren't multiplied |
| What to cache | Temperature 0, non-streaming, complete answers only | Only then is a saved answer a fair substitute |
| Cache placement | Wraps `send_request()` via `send_fn` | Keeps the provider code unchanged |
| Cost reporting | `price_as` paid price for free models; reasoning tokens counted as output | Keeps savings figures realistic; providers bill reasoning tokens |

## What later phases changed

Phase 2 (commit `69907fd`, 2026-10-19) changed the lineup for data collection. Llama 3.1 8B, Groq's small model, had been shut off for free accounts on 2026-08-16, so `allam-2-7b` was added as `groq-allam`, the smallest non-reasoning text model left on Groq's free tier. Its paid price is listed as pending, so `price_as` borrows Llama 3.1 8B's last price ($0.05 / $0.08 per million tokens). The gpt-oss prices were corrected to Groq's published rates, from $0.10 / $0.50 to $0.075 / $0.30 for the 20B model, and from $0.15 / $0.75 to $0.15 / $0.60 for the 120B model. Collection used three Groq models only: Gemini's free tier is too small (5 requests a minute, 20 a day), and `gemini-flash`, `groq-qwen` and `ollama-qwen` were blocked from collection. `send_request()` also gained `reasoning_effort` and `response_format` options, which enter the cache key only when set, so existing cache entries stayed valid.

Phase 3 brought live routing in line with those tiers. Commit `1a41f31` (2026-11-04) rebuilt the fallback chains so each tier starts with its Phase 2 model, `[groq-allam, ollama-llama]`, `[groq-small, gemini-flash]` and `[groq-large, gemini-flash]`, and removed `groq-qwen` and `ollama-qwen` from routing. Gemini went from the high tier's primary model to a backup. Commit `cbc79a1` (2026-11-07) re-tiered `models.yaml` to match: `groq-small` moved from low to medium and `groq-large` from medium to high. It also lowered the local llama 3B model's `price_as` from $0.10 / $0.50 to $0.05 / $0.08, and kept the unused models in the file marked as no longer used. The retry, breaker, timeout and cache settings from Phase 1 are unchanged.
