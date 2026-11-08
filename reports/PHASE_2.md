# Phase 2: The outcome-labeled dataset

## Summary

Phase 2 built the dataset the router learns from. Every prompt was answered by every tier, each answer was scored, and each prompt was labeled with the cheapest tier whose answer actually passed, rather than with anyone's opinion of how hard it is. The dataset has 762 prompts from four sources. 747 of them are labeled: 333 low, 329 medium and 85 high. The other 15 were excluded for documented reasons. Answers were collected on free tiers between 2026-10-19 and 2026-10-22, and the open-ended answers were judged between 2026-10-24 and 2026-11-02.

The "done when" condition is met. `data/outcomes.jsonl` has one line per prompt, with every tier's pass/fail, method and score, and the final label or the reason it has none. The most useful finding for the router is that the cheap model passes 45% of prompts and the medium model 87%: most of the room for saving lies in telling those two apart. The labels are only as good as their grading. Exact-match labels are reliable, while judge labels inherit the judge's known weaknesses, measured later in Phase 3 (see Judge reliability below).

## What was built

| Piece | File | Purpose |
|---|---|---|
| Prompt schema and validator | `data/schema.md`, `scripts/validate_prompts.py` | Record format, task types, scoring methods, constraints; validation of every record |
| Own prompts | `data/own_prompts.jsonl` | 162 hand-written app-style prompts with references and checkable constraints |
| Benchmark loaders | `eval/benchmarks.py`, `data/benchmarks/` | GSM8K, MMLU and Arena samples at pinned dataset revisions, merged into `data/prompts.jsonl` |
| Collector | `eval/build_dataset.py`, `config/collection.yaml` | Every prompt to every tier, paced within free limits; resumable append-only log |
| Automatic scorers | `app/verifier/scorers.py`, `eval/score_answers.py` | Exact match, label match, field checks and final-answer extraction; writes `data/scores.jsonl` |
| LLM judge | `app/verifier/judge.py`, `app/verifier/constraints.py`, `eval/run_judge.py`, `config/judge.yaml` | 1–5 rubric against the strong answer, code-checked countable constraints, option C runner |
| Outcomes | `eval/build_outcomes.py` | Combines grades into one label per prompt in `data/outcomes.jsonl` |
| Locked split | `eval/make_split.py`, `eval/splits.py`, `data/splits/` | Stratified 80/20 split, hash-checked, with a leak guard for training code |
| Hand labels | `eval/hand_labels.py`, `data/hand_labels/` | 50 blind labels and a comparison with real outcomes |
| Review log | `data/review_decisions.md` | Every exclusion, rewording and keep decision, with its reason |

## Data

The plan called for 600 to 1,000 prompts from three kinds of source; the final set has 762.

| Source | Prompts | Task types | Scoring |
|---|---|---|---|
| Own (hand-written) | 162 | 27 each of extraction, classification, summarization, qa, reasoning, creative | 27 field check, 27 label match, 52 exact match, 56 judge |
| GSM8K (test split) | 200 | reasoning | final answer (number) |
| MMLU (test split, all 57 subjects) | 200 | qa | final answer (letter) |
| Chatbot Arena (`arena-human-preference-140k`) | 200 | 80 qa, 47 coding, 38 open_ended, 25 creative, 9 reasoning, 1 classification | judge |

The own prompts vary deliberately in length and tidiness (53 are marked messy, written the way real users type), and each has explicit constraints the answer can be checked against. GSM8K and MMLU give known correct answers. MMLU was spread over its 57 subjects, with 3 or 4 questions each. Arena supplies real user requests, and it is where two task types not in the original plan, coding and open_ended, came from. Arena prompts were keyword-filtered and length-limited (5 to 1,200 words) into 300 candidates, and 200 survived a hand review. Every dataset is pinned to a fixed Hugging Face revision, so rebuilding gives identical files.

The three tier models in `config/collection.yaml` are all on Groq, each a fixed model with no fallback. If a different model answered, the label would no longer describe that tier.

| Tier | Model | Why |
|---|---|---|
| Low | ALLaM 2 7B (`groq-allam`) | Smallest non-reasoning text model left on Groq's free tier after Llama 3.1 8B was shut off for free accounts |
| Medium | gpt-oss-20b (`groq-small`), reasoning effort low | Mid-size reasoning model |
| High | gpt-oss-120b (`groq-large`), reasoning effort low | Strongest free model with workable limits |
| Extra (not used for labels) | llama3.2:3b on local Ollama | Free extra data |

Gemini Flash, the Phase 1 high-tier model, was blocked from collection because its free tier allows only 5 requests a minute and 20 a day. Reasoning effort was set to low after pilot runs, so gpt-oss would not spend its entire 4,000-token budget on hidden reasoning. All calls used temperature 0.

The collector paces every model against its own limits: for both gpt-oss models, 30 requests and 8,000 tokens per minute, and 1,000 requests and 200,000 tokens per rolling 24 hours. ALLaM's limits (7,000 requests a day, 6,000 tokens a minute) aren't in Groq's published table and were read from response headers. When a daily limit is reached, the collector waits it out using wall-clock time, so a sleeping laptop doesn't stretch the wait. It records the resume time for the `status` command, and cache hits don't count against the budget. Answers are appended to a log in which the newest record per prompt and tier wins, so a crash never repeats finished work. The final log holds 3,048 current answers, 762 per tier.

## Scoring

Each prompt has one scoring method, chosen so that ground truth is used wherever it exists. Exact match (52 own prompts) compares a short answer after normalizing case, spacing, one trailing period, and Unicode dashes and spaces. Label match (27 own classification prompts) works the same way on a single label. Field check (27 own extraction prompts) parses the JSON answer, allowing one code fence. It requires exactly the expected keys, compares numbers by value, and compares designated fuzzy text fields loosely and fuzzy lists as unordered sets. Final answer (400 GSM8K and MMLU prompts) reads the line after the last `Answer:`; numbers are compared by value and letters must be A to D. Truncated, empty or rejected answers fail without being scored.

The remaining 256 prompts (all Arena, plus 56 own summarization and creative prompts) are judge-scored. The judge is gpt-oss-120b in JSON mode at temperature 0. It compares a candidate answer with the high tier's answer, rates it 1 to 5, and passes it at 4 or above (`config/thresholds.yaml`). The judge also checks each listed constraint, and breaking any one caps the score at 3. Countable constraints (word, character, sentence and bullet limits) are counted in code rather than by the judge, since models miscount; the code result overrides the judge, and this turned 8 judge passes into fails. The high tier's own answer passes automatically when complete, since it is the reference. When it was truncated, the prompt has no reference and no label.

To fit the judge into the free limits, it ran under "option C". The cheap answer is judged first, the medium answer only if the cheap one failed, and otherwise medium is assumed to pass. That skipped 50 medium judge calls. The assumption was checked on the auto-graded prompts, where every tier is graded: of 283 prompts the cheap model passed, medium failed only 10 (3.5%). Option C applies to 50 labeled prompts.

The judge went through three versions. Version 2 added the code-checked constraints, meaning-based constraint checks ("18 x 12 inches" satisfies "18 x 12 in") and a rule that a factual error in the main point caps the score at 3. Version 3 told the judge that the reference is a quality bar, not the source of truth, after version 2 penalized a candidate for disagreeing with a year the reference had made up. The judge was then frozen at version 3, and further quirks are documented rather than fixed by re-judging. Version 3 made 414 judge calls using 1,114,163 tokens, with no unparseable verdicts. The older judgments (201 from version 1, 213 from version 2) are kept but unused.

## Labels

| Source | Low | Medium | High | Labeled |
|---|---|---|---|---|
| GSM8K | 119 (60%) | 75 | 6 | 200 |
| MMLU | 124 (62%) | 60 | 16 | 200 |
| Own | 48 (30%) | 92 | 22 | 162 |
| Arena | 42 (23%) | 102 | 41 | 185 |
| **All** | **333 (45%)** | **329 (44%)** | **85 (11%)** | **747** |

| Task type | Low | Medium | High | Labeled |
|---|---|---|---|---|
| classification | 21 | 5 | 2 | 28 |
| extraction | 10 | 16 | 1 | 27 |
| summarization | 2 | 15 | 10 | 27 |
| qa | 149 | 117 | 38 | 304 |
| reasoning | 121 | 103 | 12 | 236 |
| creative | 12 | 29 | 9 | 50 |
| coding | 6 | 27 | 7 | 40 |
| open_ended | 12 | 17 | 6 | 35 |

Across labeled prompts, the cheap tier passes 44.6%, medium 87.3% (including the 50 assumed passes) and high 97.2%. Eleven prompts were failed by every tier and are labeled high under the rule: 10 MMLU questions and one own SLA-deadline reasoning prompt. Before Phase 3 training they were re-reviewed. All 11 answer keys are correct except two ambiguous MMLU questions (`mmlu_0030`, `mmlu_0079`), and none were excluded.

Fifteen prompts have no label. Fourteen Arena prompts had a truncated high answer, so the judge had no reference. `arena_0067` was excluded because the judge request for its medium answer is 9,523 tokens, over the free tier's 8,000-token-per-minute limit, so it fails on every attempt. The earlier reference review, done before judging, excluded and replaced 12 benchmark questions whose answer keys were wrong or ambiguous: 4 GSM8K and 8 MMLU, plus a second replacement for `mmlu_0030` whose first replacement was also broken. It also reworded 2 own prompts and kept 14 questions after checking them. The standing rule is that benchmark records are never edited: a broken one is replaced from the same subject, and nothing is excluded just because models failed it.

## Split

The labeled data is split once, 80/20, stratified by source and task type with seed 42: 612 train and 150 test prompts, of which 598 and 149 are labeled. The SHA-256 hash of the sorted test IDs is stored in `data/splits/split_info.json`, and `load_split()` refuses to load if the files no longer match it. Training and tuning code must call `assert_no_test_ids()`, which raises an error if any test prompt reaches it. Later exclusions and replacements kept their IDs and slots, so the split never had to be redone.

## Findings

The cheap model's failures are often about form rather than content. Of its 17 extraction failures, 12 broke the requested format: 5 answers weren't valid JSON, 3 used a different key structure, and 4 wrote values in the wrong form, such as `04/22/1957` for a date, `12,480.00` as a string, or a spaced-out tracking number. Some of these also got values wrong; the other 5 extracted the wrong content. The same pattern shows up elsewhere. In 51 of its 157 GSM8K and MMLU failures, the cheap model left out the `Answer:` line or put no option letter after it. In 9 of its 43 exact-match failures, the right answer sat inside a full sentence. Medium, by contrast, passed 26 of 27 extraction prompts. The cheap model also answered 22 prompts mostly in Arabic, a side effect of its bilingual training.

Intuition was a poor guide. On the 162 own prompts, the written-in difficulty guess barely tracked the outcome: prompts marked "easy" needed medium as often as low (25 versus 24), and only 7 of 54 "hard" prompts went to the strong model's tier. A separate blind test, with 50 own prompts labeled by hand before any outcomes were seen, matched the real label 17 times (34%). Twenty-seven prompts were guessed low, but only 10 really were. Most misses overrated the cheap model on reasoning, extraction and qa, and underrated it on classification. Summarization was the hardest task type, with only 2 of 27 prompts going to low, largely because short word limits are strict and checked in code.

## Judge reliability

Phase 3 measured how far to trust the judge (`reports/judge_calibration.json`, `reports/PHASE_3.md`). The unchanged version-3 judge was run on 150 training-split GSM8K and MMLU answers whose truth is known from exact match. It agreed with the truth 70% of the time. It passed 9 wrong answers (13% of wrong answers) and failed 36 correct ones (44% of correct answers). The causes are specific. All 36 strict failures were cheap answers capped for breaking the literal format rule "ends with `Answer: …`". All 9 lenient passes came from the 14 cases where the strong model's reference was itself wrong and the candidate repeated it; when the reference was right, the judge never passed a wrong answer. For the 241 judge-scored labels, this means content judgments are trustworthy, but a cheap answer may be failed for a minor format slip, which pushes some labels from low up to medium. It also means a wrong reference can let a wrong answer through. The 506 auto-graded labels don't depend on the judge.

## Limitations

Each prompt was answered once per tier, so labels carry some sampling noise even at temperature 0. The 50 option C prompts assume medium passed. Judge labels depend on the strong model's answer as the reference, which is occasionally wrong, and the judge enforces format rules literally. One known case, `own_summarization_022`, failed medium because of a year the reference invented, even under version 3. Costs are estimates at published paid prices: ALLaM's price is still pending, so it borrows Llama 3.1 8B's last price. The local llama model was priced like gpt-oss-20b during collection, which overstated its cost; `models.yaml` was corrected in Phase 3, but the saved answers keep the old figure. Coding prompts, the most common developer request, number only 47, and there's no system design category. All three tiers are on Groq, so the labels describe those specific models.

## Decisions

| Decision | Choice | Reason |
|---|---|---|
| Tier models | ALLaM 2 7B / gpt-oss-20b / gpt-oss-120b, all on Groq | Free limits large enough to collect everything; Llama 3.1 8B no longer free |
| Fallback during collection | Off | A label must describe one fixed model |
| Reasoning effort | Low for both gpt-oss models | Default effort exhausted the token budget on reasoning |
| Judge pass line | 4 of 5 | Matches the plan; "fully acceptable for the user" |
| Judge strategy | Option C | Fewer judge calls within free limits; assumption held on 96.5% of auto-graded cases |
| Judge version | Frozen at 3 | Stop re-judging; document further quirks instead |
| Broken benchmark items | Exclude and replace from the same subject | Benchmark records are never edited |
| Unjudgeable prompts | Exclude (14 no reference, `arena_0067`) | No fair way to label them |
| Split | 80/20 stratified by source and task type, seed 42, hash-locked | Every reported result comes from an untouched test set |

## What Phase 3 used it for

Phase 3 trained its "will this tier pass?" classifiers on the 598 labeled training prompts, using `low_pass` and `medium_pass` from `outcomes.jsonl` as targets. It evaluated routing without any new API calls, by replaying each prompt's recorded pass/fail, paid-price cost and latency for whichever tier a strategy chose. The locked 149-prompt test set was used once, for the final results. The same answers powered the cascade experiments and the judge calibration, and the 45% cheap pass rate set the ceiling on how much routing could save.
