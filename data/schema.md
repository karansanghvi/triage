# Prompt dataset schema

Prompts for Phase 2 live in JSONL files (one JSON object per line):

- `data/seed_prompts.jsonl`: hand-written examples that set the style.
- `data/own_prompts.jsonl`: the custom dataset (162 records, seeds included).
- Public benchmark prompts (GSM8K, MMLU, and later Arena) use the same format, with a few extra rules
  described under [Benchmark records](#benchmark-records).

Each prompt is later sent to cheap, medium, and strong models, and the outcomes
(which tier passes) train the routing classifier.

Check a file with:

```bash
python scripts/validate_prompts.py                       # data/own_prompts.jsonl
python scripts/validate_prompts.py data/seed_prompts.jsonl
python scripts/validate_prompts.py data/seed_prompts.jsonl data/own_prompts.jsonl  # IDs unique across both
```

## Record format

```json
{
  "id": "own_extraction_007",
  "source": "own",
  "task_type": "extraction",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "reference": {"order_id": "A-77812"},
  "scoring": "field_check",
  "difficulty_guess": "hard",
  "constraints": ["valid JSON"],
  "notes": "partial date must be null"
}
```

| Field | Type | Rules |
|---|---|---|
| `id` | string | Unique. Own records: `own_<task_type>_<3 digits>`, and the task type in the ID must match `task_type`. Benchmark records: `<source>_<4 digits>` (`gsm8k_0001`, `mmlu_0001`, `arena_0001`), and the prefix must match `source`. |
| `source` | string | `own` for prompts written for this project; `gsm8k`, `mmlu`, or `arena` for public benchmark prompts. |
| `task_type` | string | `extraction`, `classification`, `summarization`, `qa`, `reasoning`, `creative`, `coding`, `open_ended` |
| `messages` | list | Non-empty. Each item is `{"role", "content"}` with role `system`, `user`, or `assistant` and non-empty content. An optional system message comes first (include one in roughly half the records). The last message is from the user. |
| `reference` | varies | The expected answer, or `null`. See the per-type rules. |
| `scoring` | string | `field_check`, `label_match`, `exact_match`, `final_answer`, `judge` |
| `fuzzy_fields` | list of strings | Optional, `field_check` only. Reference fields compared after normalizing case, whitespace, and punctuation. See below. |
| `meta` | object | Required for `gsm8k`, `mmlu`, and `arena` records, where it traces the record back to its source row. See [Benchmark records](#benchmark-records). Optional for `own` records, with no required fields. |
| `messy` | boolean | Optional. `true` when the input is messy: typos, informal writing, forwarded chains, or copy-paste artifacts (pasted tables, PDF headers, raw commit titles). Left unset on `arena` records. |
| `difficulty_guess` | string | `easy`, `medium`, `hard`. Benchmark records may also use `unrated`. |
| `constraints` | list of strings | Explicit, checkable rules stated in the prompt (`"exactly 3 bullets"`, `"under 50 words"`, `"valid JSON"`). Empty list if none. |
| `notes` | string | Short note on what makes the prompt easy or hard. For hard prompts, name the trap. |

## Per-type rules

| task_type | scoring | reference |
|---|---|---|
| extraction | `field_check` (`judge` allowed for `arena`) | The exact expected JSON object (a dict). Missing or ambiguous fields are `null`. Arena records judged instead have reference `null`. |
| classification | `label_match` (`judge` allowed for `arena`) | The correct label as a string. The prompt must list the allowed labels. Arena records judged instead have reference `null`. |
| summarization | `judge` | `null`. Use constraints for length, bullet counts, audience, and must-include points. |
| qa | `exact_match`, `final_answer`, or `judge` | `exact_match` / `final_answer`: one short, unambiguous answer as a string. `judge`: `null`. |
| reasoning | `exact_match` or `final_answer` (preferred), or `judge` | `exact_match` / `final_answer`: the single numeric or short answer as a string. `judge` only when no exact answer is possible. |
| creative | `judge` | `null`. |
| coding | `judge` | `null`. Writing, explaining, or debugging code. |
| open_ended | `judge` | `null`. Advice, opinions, recommendations, or casual conversation with no single correct answer. |

In general: `exact_match` and `final_answer` need a non-null string reference, and `judge` always has a `null`
reference.

## Exact-match normalization

Before comparing an `exact_match` answer with its reference, both are normalized:

- compared case-insensitively
- leading and trailing whitespace trimmed
- one trailing period ignored (`"14."` matches `"14"`)
- spaces around `+`, `:`, `-`, `/`, and `,` ignored (`"Ctrl + Shift + E"` matches `"Ctrl+Shift+E"`, and `"O-101, O-102"` matches `"O-101,O-102"`)

Nothing else is normalized. In particular, dates and times must still match the format the prompt asks
for: `"2026-05-25"` does not match `"May 25, 2026"`, and `"08:50"` does not match `"8:50"`.

## Final answer (final_answer scoring)

For `final_answer`, the model may reason freely and then ends its response with a line `Answer: <value>`.
Only that value is graded, so the working is never checked. A user message must tell the model to end this way
and must contain the literal text `Answer:` (the validator checks this), for example:

> Solve the problem step by step, then end with a final line in the form "Answer: <number>".

Scoring (`app/verifier/scorers.py`, `score_final_answer`):

1. **Find the answer.** Take the **last** occurrence of `Answer:` in the output, in any case (`answer:` and
   `ANSWER:` count, and so does `Final Answer:`). The value is the rest of that line. If nothing is left on
   that line (for example `**Answer:**` with the value below it), the value is the next non-empty line
   instead; no later lines are searched. Markdown emphasis around the value is dropped (`**18**` reads as
   `18`). If there is no `Answer:` at all, or nothing follows it, the record fails.
2. **Numeric references** (the whole reference is one number, optionally with `$` and thousands commas, e.g.
   `"18"`, `"-0.25"`, `"1,800"`, `"1545.00"`): remove `$` from the value and take the **first** number in it,
   allowing a minus sign, thousands commas, and decimals. Units and words around it are ignored. The two are
   compared as numbers, so `18`, `18.0`, `$18`, `18 apples`, and `about 18 in total` all match `"18"`, and
   `1,800` matches `"1800"`. If the value has no number, the record fails.
3. **Multiple-choice references** (a single letter `A`–`D`, used by MMLU): the first standalone capital `A`–`D`
   in the value is graded, so `B`, `B)`, `(B)`, `B) Paris`, `**B**`, and `Option B` all count as `B` (the `B` in
   `Bob` is not standalone). A lone lowercase letter such as `b` or `(b)` also counts. No letter means a fail.
   Only the answer value is searched, never the reasoning above it.
4. **Other references** (for example a date): compared as text using the
   [exact-match normalization](#exact-match-normalization) rules.

Numbers must be equal; there is no tolerance, so `18.5` does not match `"18"`. Because the first number wins,
`Answer: 3 boxes, so 18` is graded as `3`.

## Fuzzy fields (field_check only)

By default, `field_check` compares every field strictly. `fuzzy_fields` lists the free-text fields (names,
addresses, titles, descriptions) that should instead be compared after lowercasing, collapsing whitespace,
and stripping punctuation, so `"Harbor Line Printing Co."` matches `"harbor line printing co"`.
IDs, codes, flight and tracking numbers, dates, amounts, booleans, and enum values stay strict.

Each entry is a dotted path into the reference. A path that passes through a list applies to every item
in that list:

```json
"reference": {
  "new_address": {"street": "77 Harrow Court, Unit 12", "zip": "53711"},
  "line_items": [{"description": "Window display design", "quantity": 1}],
  "required_skills": ["Go", "PostgreSQL"]
},
"fuzzy_fields": ["new_address.street", "line_items.description", "required_skills"]
```

When a fuzzy field is a list of strings, it is compared as an unordered set after normalization, so
`["Python", "SQL"]` matches `["sql", "python"]`.

Every path must exist in the reference and point to a string (or `null`), or to a list of strings.
Omit the field, or use an empty list, when everything is strict.

## Benchmark records

Records with `source` set to `gsm8k`, `mmlu`, or `arena` follow every rule above, plus:

- The ID is the source plus 4 digits, numbered within that source: `gsm8k_0001`, `mmlu_0001`, `arena_0001`.
- `difficulty_guess` may be `unrated` (benchmark rows aren't hand-rated; their difficulty is measured by which
  tier passes). Own records must use `easy`, `medium`, or `hard`.
- The near-duplicate check only compares records from the same source.
- Arena prompts come with no reference answers, so `arena` records of type `classification` or `extraction` may
  use `judge` scoring with a `null` reference. Own records keep the stricter rules.
- A `meta` object records exactly where the prompt came from:

| Field | Type | Rules |
|---|---|---|
| `dataset` | string | Hugging Face dataset name, e.g. `"openai/gsm8k"`, `"cais/mmlu"`. |
| `config` | string | Dataset config, e.g. `"main"` for GSM8K. |
| `split` | string | Split the row came from, e.g. `"test"`. |
| `index` | integer | The row's original position in that split (0-based, non-negative). |
| `subject` | string | MMLU only, and required there (e.g. `"high_school_biology"`). Not allowed on `gsm8k` or `arena`. |
| `is_code` | boolean | Arena only, optional: the dataset's own `is_code` flag. Not allowed on `gsm8k` or `mmlu`. |
| `category_tag` | object | Arena only, optional: the dataset's own `category_tag` annotations, copied as-is. Not allowed on `gsm8k` or `mmlu`. |

These rules apply only to benchmark sources. An `own` record may include `meta` for its own notes; it must be an
object, but no fields are required.

GSM8K example:

```json
{
  "id": "gsm8k_0001",
  "source": "gsm8k",
  "task_type": "reasoning",
  "messages": [
    {"role": "user", "content": "<GSM8K question text>\n\nSolve the problem step by step, then end with a final line in the form \"Answer: <number>\"."}
  ],
  "reference": "42",
  "scoring": "final_answer",
  "difficulty_guess": "unrated",
  "constraints": ["ends with a line 'Answer: <number>'"],
  "notes": "multi-step arithmetic word problem",
  "meta": {"dataset": "openai/gsm8k", "config": "main", "split": "test", "index": 117}
}
```

MMLU example:

```json
{
  "id": "mmlu_0001",
  "source": "mmlu",
  "task_type": "qa",
  "messages": [
    {"role": "user", "content": "<MMLU question text>\n\nA. ...\nB. ...\nC. ...\nD. ...\n\nAnswer with the letter only."}
  ],
  "reference": "B",
  "scoring": "exact_match",
  "difficulty_guess": "unrated",
  "constraints": ["single letter only"],
  "notes": "factual recall with plausible distractors",
  "meta": {"dataset": "cais/mmlu", "config": "all", "split": "test", "index": 2045, "subject": "high_school_biology"}
}
```

These examples are illustrative: the question text, answer, and index are placeholders, not real rows.

Benchmark questions that need working (such as GSM8K) use [`final_answer`](#final-answer-final_answer-scoring)
scoring, so models can reason before answering.

## Difficulty

About 8 easy, 9 medium, and 8 hard per batch of 25.

- **Easy:** one clear instruction, clean input, one obvious answer.
- **Medium:** a few constraints, longer input, some irrelevant information.
- **Hard:** a specific trap that small models tend to fail, such as nested or multi-entity JSON, missing fields that must be `null`, mixed date or currency formats, distractor details, corrections later in the input, multi-step reasoning, or strict formatting combined with content rules. Describe the trap in `notes`.

## Content rules

These apply to `own` records. Benchmark records keep the benchmark's original wording.

- Write prompts like real app requests (an instruction plus pasted content), not quiz questions.
- Vary domains, length, and tone. At least a third of prompts should have messy input (typos, informal writing, forwarded chains, copy-paste artifacts).
- Keep each prompt under ~1,200 words.
- Use invented names, emails, addresses, and order numbers only.
- No near-duplicates (the validator warns when user content from the same source is more than 85% similar).
