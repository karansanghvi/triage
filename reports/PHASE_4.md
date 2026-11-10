# Phase 4

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

- **Circuit breaker scope.** The breaker counts timeouts, server errors and connection errors per *provider*, so a run of slow strong-tier calls (groq-large) could trip it and skip every Groq model (low, medium and high) for 60 s. Consider a per-model breaker for timeouts if this shows up in live logs.
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
