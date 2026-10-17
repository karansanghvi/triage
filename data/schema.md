# Prompt dataset schema

Prompts for Phase 2 live in JSONL files (one JSON object per line):

- `data/seed_prompts.jsonl`: hand-written examples that set the style.
- `data/own_prompts.jsonl`: the generated dataset (~150 records).

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
| `id` | string | `own_<task_type>_<3 digits>`, unique. The task type in the ID must match `task_type`. |
| `source` | string | `"own"` for prompts written for this project. |
| `task_type` | string | `extraction`, `classification`, `summarization`, `qa`, `reasoning`, `creative` |
| `messages` | list | Non-empty. Each item is `{"role", "content"}` with role `system`, `user`, or `assistant` and non-empty content. An optional system message comes first (include one in roughly half the records). The last message is from the user. |
| `reference` | varies | The expected answer, or `null`. See the per-type rules. |
| `scoring` | string | `field_check`, `label_match`, `exact_match`, `judge` |
| `fuzzy_fields` | list of strings | Optional, `field_check` only. Reference fields compared after normalizing case, whitespace, and punctuation. See below. |
| `messy` | boolean | Optional. `true` when the input is messy: typos, informal writing, forwarded chains, or copy-paste artifacts (pasted tables, PDF headers, raw commit titles). |
| `difficulty_guess` | string | `easy`, `medium`, `hard` |
| `constraints` | list of strings | Explicit, checkable rules stated in the prompt (`"exactly 3 bullets"`, `"under 50 words"`, `"valid JSON"`). Empty list if none. |
| `notes` | string | Short note on what makes the prompt easy or hard. For hard prompts, name the trap. |

## Per-type rules

| task_type | scoring | reference |
|---|---|---|
| extraction | `field_check` | The exact expected JSON object (a dict). Missing or ambiguous fields are `null`. |
| classification | `label_match` | The correct label as a string. The prompt must list the allowed labels. |
| summarization | `judge` | `null`. Use constraints for length, bullet counts, audience, and must-include points. |
| qa | `exact_match` or `judge` | `exact_match`: one short, unambiguous answer as a string. `judge`: `null`. |
| reasoning | `exact_match` (preferred) or `judge` | `exact_match`: the single numeric or short answer as a string. `judge` only when no exact answer is possible. |
| creative | `judge` | `null`. |

In general: `exact_match` needs a non-null string reference, and `judge` always has a `null` reference.

## Exact-match normalization

Before comparing an `exact_match` answer with its reference, both are normalized:

- compared case-insensitively
- leading and trailing whitespace trimmed
- one trailing period ignored (`"14."` matches `"14"`)
- spaces around `+`, `:`, `-`, `/`, and `,` ignored (`"Ctrl + Shift + E"` matches `"Ctrl+Shift+E"`, and `"O-101, O-102"` matches `"O-101,O-102"`)

Nothing else is normalized. In particular, dates and times must still match the format the prompt asks
for: `"2026-05-25"` does not match `"May 25, 2026"`, and `"08:50"` does not match `"8:50"`.

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

## Difficulty

About 8 easy, 9 medium, and 8 hard per batch of 25.

- **Easy:** one clear instruction, clean input, one obvious answer.
- **Medium:** a few constraints, longer input, some irrelevant information.
- **Hard:** a specific trap that small models tend to fail, such as nested or multi-entity JSON, missing fields that must be `null`, mixed date or currency formats, distractor details, corrections later in the input, multi-step reasoning, or strict formatting combined with content rules. Describe the trap in `notes`.

## Content rules

- Write prompts like real app requests (an instruction plus pasted content), not quiz questions.
- Vary domains, length, and tone. At least a third of prompts should have messy input (typos, informal writing, forwarded chains, copy-paste artifacts).
- Keep each prompt under ~1,200 words.
- Use invented names, emails, addresses, and order numbers only.
- No near-duplicates (the validator warns when user content is more than 85% similar).
