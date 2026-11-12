# Phase 4

## Results

Every number below says which split it comes from. **Validation** (117 prompts) was used for all tuning and choices. **Test** (149 prompts) was used once, for a pre-registered check at the end.

### Summary

Phase 4 asked one question: can Triage get the strong model's quality for less by answering with the medium model and checking the answer, escalating to the strong model only when the check fails? To find out, it built an AI answer checker, an OpenAI-compatible API with streaming, sync and background verification, a SQLite request log, and safe retraining. On validation, selective checking matched the strong model's quality (113/117) at about the same cost as predict-only routing, and looked cheaper once the strong model costs more than about 2× the medium one. The pre-registered test run did not confirm this. On test, selective checking got 139/149 right, no better than random routing at the same cost, while predict-only routing got 144/149 (always-high: 145) but saved only 12.4% (28.4% on validation). **The checker does not add measurable value at Groq's prices, so predict-only routing stays the default.**

### What was built

- **Step 0, validation split:** the 598 labeled train prompts became 481 fit / 117 validation. Classifier v2 is trained on fit only, and live settings match Phase 2.
- **Step 1, AI checker v3:** groq-small judges whether an answer is good, without a reference answer. It runs after the free format check (`app/verifier/ai_checker.py`).
- **Step 2, calibration:** groq-small and groq-large were measured as checkers on a 341-answer fit-set sample (`reports/checker_calibration.json`).
- **Step 3, checker routing simulation:** checker strategies were scored on validation with the real verdicts and every call charged (`eval/checker_routing.py`), plus a price-gap analysis (`eval/price_sensitivity.py`).
- **Step 3, selective checking:** the check is skipped when the classifier is already confident (p_medium ≥ skip_above). This works in the simulation and in sync mode.
- **Step 4, pre-registered final test:** one test-set run, with the strategy and the 100 prompts to check fixed in advance (`eval/checker_test_run.py`, `eval/final_test_phase4.py`).
- **Step 5, OpenAI-compatible API:** FastAPI with streaming, API keys, a fallback header, request lookups, and a fake-model mode for tests (`app/api/`).
- **Step 6, one-line integration:** tests that call Triage with the official OpenAI Python SDK (sync and async, with and without streaming), plus Python and JavaScript examples in the README. Built together with Step 5.
- **Step 7, sync and background verification:** `X-Triage-Verify: sync` checks the answer before replying and escalates if it's bad. Background labels compare a sample of answers with the strong model, under a daily cap.
- **Step 7, per-model circuit breaker:** server errors, timeouts and rate limits skip one model. Only connection and auth errors skip a whole provider.
- **Step 8, stored classifier inputs:** auto requests keep their features and embedding for retraining, with a `training_eligible` flag.
- **Step 9, safe retraining:** `scripts/retrain.py` trains a candidate on fit plus eligible live labels, guards against validation and test leaks, and replaces the live model only if it does better on validation.
- **Step 10, SQLite request log:** restart-safe logging. Prompts are stored only as an HMAC keyed by `TRIAGE_HASH_SECRET`; no prompt or answer text is kept.

### Checker calibration (fit-set sample, `reports/checker_calibration.json`)

These are medium answers: 241 of the 341 sampled, checker v3. "Catch" means a bad answer judged not good. "False alarm" means a good answer judged not good. "Trust" means the share of "not good" verdicts that were right, reweighted to the real share of bad answers (15%).

| checker | catch rate (95% CI) | false-alarm rate | trust rate (95% CI) | repeat-run flips |
|---|---|---|---|---|
| groq-small | 52% (40–64%) | 3.2% | 75% (60–91%) | 2 of 20 |
| groq-large | 56% (43–67%) | 4.5% | 69% (53–87%) | 3 of 20 |

**Why groq-small was chosen.** The deciding result was on the non-comparable (open-ended) slice, where the checker actually runs, since comparable prompts use agreement instead. There, groq-small caught 64% of bad medium answers against groq-large's 53% (23 vs 19 of 36; the intervals overlap), though it also raised more false alarms there (6.4% vs 2.1%). Overall, the two catch rates overlap almost completely. groq-small raised fewer false alarms, had a slightly higher trust rate, flipped less on repeat runs, and costs half as much as groq-large ($0.075 / $0.30 vs $0.15 / $0.60 per million input/output tokens). One point went the other way: on exact-match items, groq-large caught more bad medium answers (62% vs 35%).

### Validation results (117 prompts, `reports/pareto_val.json`)

Real Groq prices. "Escalated" is the share of prompts whose final answer comes from high. "Checked" is the share that get an AI check.

| strategy (validation) | correct | quality kept | cost saved | escalated | checked |
|---|---|---|---|---|---|
| always medium | 106 | 93.8% | 64.7% | 0% | 0% |
| always high | 113 | 100.0% | 0% | 100% | 0% |
| predict-only, balanced (τ 0.86) | 113 | 100.0% | 28.4% | 42% (routed to high) | 0% |
| medium then check (on_error escalate) | 113 | 100.0% | 21.3% | 7% | 100% |
| selective checking (skip_above 0.90) | 113 | 100.0% | 27.9% | 7% | 68% |
| oracle | 116 | 102.7% | 57.0% | 9% | – |

- **Medium's mistakes:** medium was wrong on 11 of 117 prompts. The checker flagged 6 of them, missed 4, and returned no verdict on 1 (an invalid-JSON reply, escalated). It raised 1 false alarm. That is a catch rate of about 60% on the answers it judged.
- **No strategy beat always-medium.** Every one that got more right also cost more.
- **Best selective setting:** skip_above 0.90 kept all of full checking's catches while skipping 32% of the checks. Below 0.90, quality started to drop.

### Price-gap analysis (validation, `reports/price_sensitivity_val.json`)

Only the high tier's price was scaled, as R = high price / medium price; Groq's real R is 2. For each strategy, the cheapest setting reaching 113/117 correct, as a percentage of always-high's cost at that R:

| R (validation) | 2 | 3 | 5 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| predict-only (best τ) | 71.6% | 65.8% | 61.2% | 57.7% | 56.0% | 55.4% |
| selective checking (skip_above 0.90) | 72.1% | 53.8% | 39.1% | 28.1% | 22.7% | 20.8% |

On validation, the two broke even at R ≈ 2.03 for 113 correct, and at R ≈ 2.5 for 111. Above that, checking looked clearly cheaper. These are validation numbers, chosen and scored on the same 117 prompts, so they flatter the result.

### Pre-registered final test (149 prompts, `reports/final_test_phase4.json`)

**What was registered.** Before any test-set check was made, `reports/PHASE4_PREREGISTRATION.json` fixed the following:
- the strategy: medium_then_check_selective, skip_above 0.90, min_answer_tokens 0, on_error escalate;
- the checker: groq-small v3, with its prompt hash;
- the classifier, v2;
- the 100 test prompts the strategy would check.

**What was verified before scoring.** The scoring script confirmed that all four still matched the registration, and that every registered check had finished on the exact answer. Nothing was tuned on test.

| strategy (test, real prices) | correct | quality kept | cost saved | escalated | checked |
|---|---|---|---|---|---|
| always medium | 133 | 91.7% | 61.0% | 0% | 0% |
| always high | 145 | 100.0% | 0% | 100% | 0% |
| predict-only, balanced (τ 0.86) | 144 | 99.3% | 12.4% | 55% (routed to high) | 0% |
| **selective checking (pre-registered)** | **139** | 95.9% | 29.6% | 6.7% | 67.1% |
| oracle | 149 | 102.8% | 57.0% | 9.4% | – |

**The validation finding did not hold.**
- **The checker caught far fewer mistakes.** Medium was wrong on 16 of 149 test prompts. The format check caught 2, 2 were skipped as confident, and of the 12 the checker saw, it **caught 4 and missed 8 (33%, against about 60% on validation)**. It also raised 4 false alarms. All 6 escalated mistakes were fixed by high.
- **Selective checking matched random routing at the same cost.** Mixing medium and high at random to spend the same amount would get about 139.2 right; selective checking got 139. Predict-only got 144 while saving 12.4%, about 1.4 above random at its cost and 1 below always-high.
- **The price ratio doesn't rescue it.** With both settings fixed, selective checking is cheaper than predict-only at every ratio tested (70.4% vs 87.6% of always-high's cost at R = 2, 18.1% vs 78.4% at R = 20). But it is 5 correct answers behind at every ratio, so on test it never reaches predict-only's quality.
- **Nothing beat always-medium** at any ratio, on test or on validation.

### Why the checker didn't pay off

- **Medium is already right about 90% of the time** (106/117 on validation, 133/149 on test). Most checks confirm an answer that was fine.
- **A check costs about three-quarters of a medium answer** (validation: $0.089 of checks per 1,000 requests against $0.120 for medium's answers, when every answer is checked). Checking nearly doubles medium's cost to fix a few prompts.
- **Groq's strong model costs only 2× the medium one,** so escalating directly, as predict-only does, is cheap. Checking only saves money when escalation is expensive.
- **groq-small checking its own answers misses too many mistakes.** It is the same model as the medium tier, and on test it caught a third of the bad answers it saw.

### Decisions

- **Predict-only routing stays the default: near always-high quality, but modest savings at Groq's prices.** On test it got 144/149 (always-high: 145) and saved 12.4% (28.4% on validation).
- **Sync verification stays opt-in** (`X-Triage-Verify: sync`, auto non-streaming requests only). When used, the medium check is skipped at p_medium ≥ 0.90 (`config/routing.yaml`, chosen on validation).

### Bugs found and fixed

- **Circuit breaker scope:** one failing model used to block every model on its provider for 60 s. Now only connection and auth errors are provider-wide (Step 7).
- **Groq JSON-mode rejections:** a `json_validate_failed` 400 was retried forever as a provider error. It is now a finished `invalid_json` verdict (Step 3).
- **Fake-mode cache leak:** fake-model runs wrote answers into the real response cache, and one fake answer was served in real mode. Fake mode now has its own cache and log, each mode refuses the other's files, and the poisoned entry and the incident row were removed.
- **Cache hits counted as new data:** cached repeats are no longer training-eligible or sampled for background labels.

### Limitations

- **Few mistakes to learn from:** medium was wrong only 11 times on validation and 16 times on test, so one or two prompts move every conclusion.
- **Second use of the test set:** Phase 3 used it first, and its result was known when Phase 4 began. Nothing in Phase 4 was tuned on test.
- **Groq only:** all three tiers and the checker are Groq models. The price-gap analysis varies price but not the quality gap or answer lengths.
- **No real traffic yet:** every result comes from the labeled benchmark prompts.
- **Free-tier capacity:** groq-small allows 200K tokens a day, which slowed every checker run. A busy deployment would need paid tiers.

### Next steps (Phase 5)

- **A more accurate checker:** a different, stronger model than the one that wrote the answer, or a task-specific check.
- **Stronger cheap models,** so more prompts can stay on the low and medium tiers.
- **Multi-provider routing with BYOK,** where the price gaps between tiers are larger and checking might pay.
- **Deployment and real traffic,** to grow the live label set that retraining uses.

## Step 0 summary

Step 0 prepared the ground for tuning: a validation set to tune on, live calls that match the data the classifier learned from, timeouts based on measured latency, and a classifier retrained without the validation prompts. No API calls were made and the locked test set was not read.

- **Validation split.** The 598 labeled train prompts were split 481 fit / 117 validation, stratified by (source, task_type) with seed 42, using the same method as the train/test split (`eval/make_validation_split.py`, `data/splits/val_ids.txt`, `val_info.json`). Classifiers now train on the fit set by default (`--include-val` for a final refit).
- **Task-type guesser: reverted.** Extra coding keywords ("option D") gained 2 prompts on fit (67.7% → 68.1%) but lost 2 on validation (73.5% → 71.8%), so the rules are back to the Phase 3 behaviour (`reports/guesser_agreement.json`).
- **Live settings match Phase 2.** `config/routing.yaml` now sets `temperature: 0`, `max_tokens: 4000` and `reasoning_effort: low` for groq-small and groq-large, exactly as in `config/collection.yaml`; a test fails if they drift apart. `reasoning_effort` is applied per model, so the gemini-flash fallback never receives it.
- **Timeouts from Phase 2 latency** (`reports/latency_p95.json`, train split, cache hits and errors excluded): default 8 s (groq-large p95 6.65 s, rounded up, + 1 s), groq-large 10 s (p99 8.71 s + 1 s; no Phase 2 call exceeded it), ollama-llama 32 s (p95 30.8 s + 1 s), gemini-flash 30 s (no data; last fallback).
- **Classifier v2 chosen.** `classifier_c_logreg_cv` v2 (hand-tagged task types, fit set, FEATURE_VERSION 3) is live in `routing.yaml`. The predictor also now refuses to start if the version recorded inside the model file differs from the one requested.

### Validation baseline: the gap Steps 1–4 must close

| strategy (validation, 117 prompts) | quality kept | cost saved |
|---|---|---|
| always medium | 93.8% | 65% |
| Triage, balanced (tau 0.86, v1's preset) | 100.0% | 27% |
| oracle (perfect router) | 102.7% | 57% |

Triage keeps all of the strong model's quality but saves far less than simply always using medium. At tau 0.86 it never routes to the cheap tier.

## Decision: classifier v2 uses hand-tagged task types

Two candidates were trained on the fit set: one on hand-tagged task types (v2) and one on task types from the keyword guesser (`classifier_c_logreg_cv_guessed` v1). On validation they tied: identical quality (96.6%, 100% of the strong model's), log-loss differences of 0.006 or less, and a cost-savings gap (27–28% vs 31%) of only a few prompts (`reports/candidates_val.json`). Under the phase rule, a challenger must clearly win on validation to replace the current approach, so hand tags stay. Hand-tag training also keeps the classifier independent of the guesser.

Finding: task type barely affects routing, so the embedding-based task classifier follow-up is dropped. The guessed model files and `data/features_guessed.npz` are kept for the record.

## NOTES (for later steps)

- **Circuit breaker scope (FIXED in Step 7).** The breaker used to count timeouts, server errors and connection errors per *provider*, so one failing model (e.g. slow groq-large calls, or a groq-small checker returning 5xx) skipped every Groq model for 60 s. It came up three times: in the Step 5 API tests, and in Step 7, where it sent sync escalations to gemini-flash instead of groq-large. Now server errors (5xx), timeouts and rate limits trip a breaker for that *model* only; only connection errors and auth errors (401/403, a new `auth` error kind) trip the provider-wide breaker (`app/reliability.py`). The tests that worked around it now use server errors and check that groq-large is still used after groq-small trips.
- **Streaming timeouts (Step 5).** Streaming needs a time-to-first-token timeout, not a total-time timeout: a long answer that is streaming steadily should not be cut off, but a stream that never starts should fail over quickly.
- **Split option for simulations (before Step 3).** `eval/pareto.py` and `eval/cascade.py` still score `load_eval_rows("train")`, which includes the validation prompts. `routing_sim.load_eval_rows` now accepts `fit` and `val`; add the option to those two scripts before tuning on validation.
- **Tau presets (Step 3).** The cheap / balanced / high presets in `config/routing.yaml` were picked for classifier v1 on out-of-fold probabilities over all 598 train prompts. Re-pick them on the validation set using v2.

## Step 1: AI answer checker (smoke tests)

`app/verifier/ai_checker.py` asks a model (groq-small, gpt-oss-20b) whether an answer is good, with no reference answer; `app/verifier/pipeline.py` runs it last, after the free format checker and the agreement check. Results are logged in `data/checks/checks.jsonl` (v1, v2 and v3 records all kept). Twelve real calls were made in total, all on fit-set prompts; they check plumbing and cost only, not accuracy.

| run | calls | result | report |
|---|---|---|---|
| v1 smoke (5 categories) | 5 | the right low-tier answer gsm8k_0014 was failed for formatting (the format trap) | `reports/checker_smoke.json` |
| v2 format traps | 3 | 1 of 3 right answers judged good; gsm8k_0177 and gsm8k_0065 still failed for formatting | `reports/checker_smoke_format_trap.json` |
| v3 format traps + wrong answer | 4 | 2 of 3 format traps judged good (gsm8k_0177 still failed for formatting); the wrong answer gsm8k_0076 was still judged not good | `reports/checker_smoke_v3.json` |

No call was truncated, unparseable or retried. Checks took about 0.4–0.6 s and used at most 220 output tokens against the 1,024 allowed.

### NOTES (for Steps 2 and 3)

- **Don't run the AI checker on low-tier answers.** In the smoke tests, checking a low-tier answer cost 2–6× the answer itself (a check is about $0.00005–0.0001 at Groq's paid prices; a groq-allam answer about $0.00001–0.00003). Step 3 should not run the AI checker on low-tier answers. Questions with a final answer (an "Answer:" line or JSON) should use agreement first.
- **Decision rule: stop prompt tuning.** If v3 still fails format traps, stop tuning the prompt and measure the wrong-fail rate (right answers judged not good) on the fit set in Step 2 instead. v3 still fails one of the three (gsm8k_0177), so the checker stays at v3 and Step 2 measures how often this happens.
