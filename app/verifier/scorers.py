"""Scorers: grade a model's output against a prompt's reference (see data/schema.md)."""
import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Optional

# "Answer:" in any case; the last one in the output is the one that counts
ANSWER_MARKER = re.compile(r"answer:", re.IGNORECASE)

# Whitespace and markdown emphasis stripped from the edges of an answer value ("**18**" -> "18")
MARKDOWN_EDGES = " \t\r*_`"

# A number, optionally negative, with optional thousands commas and decimals: -1,234.5 / 18 / .75
NUMBER = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|-?\.\d+")

# A standalone capital A-D: "B", "B)", "(B)", "**B**", "Option B", but not the B in "Bob"
CHOICE_LETTER = re.compile(r"(?<![A-Za-z0-9])[A-D](?![A-Za-z0-9])")

# Applied by every scorer before comparing: Unicode spaces become a normal space and Unicode
# hyphens, dashes and minus signs become "-". Models (gpt-oss especially) often write dates and
# numbers like "2026\u201110\u201119\u202f15:30" with non-breaking hyphens and narrow spaces.
UNICODE_SPACES = "\u00a0\u202f\u2009"                       # no-break, narrow no-break, thin
UNICODE_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2212"  # hyphen, non-breaking hyphen, figure dash, en, em, minus
UNICODE_TRANSLATION = str.maketrans({**{c: " " for c in UNICODE_SPACES}, **{c: "-" for c in UNICODE_DASHES}})

# Spaces around these characters are ignored when comparing text answers (exact-match rules)
TIGHT_PUNCTUATION = re.compile(r"\s*([+:\-/,])\s*")


def normalize_unicode(text: str) -> str:
    """Map Unicode spaces to " " and Unicode hyphens and dashes to "-"."""
    return text.translate(UNICODE_TRANSLATION)


@dataclass(frozen=True)
class ScoreResult:
    passed: bool
    extracted: Optional[str]  # what was graded, or None if nothing could be pulled out
    reason: str


def extract_final_answer(output: str) -> Optional[str]:
    """Return the value after the last "Answer:", or None if there is no "Answer:" at all.

    The value is the rest of that line. If nothing is left there (models often write "**Answer:**"
    and put the value below it), the next non-empty line is used instead. Markdown emphasis around
    the value is dropped, so "**18**" reads as "18". Returns "" when nothing follows at all.
    """
    matches = list(ANSWER_MARKER.finditer(output))
    if not matches:
        return None
    lines = output[matches[-1].end():].split("\n")
    candidates = (line.strip(MARKDOWN_EDGES) for line in lines)
    return next((value for value in candidates if value), "")


def parse_number(text: str) -> Optional[Decimal]:
    """Return the first number in text, ignoring $ signs, thousands commas, and surrounding units."""
    # Drop $ so "-$18" and "$-18" both read as -18; Unicode minus signs become a plain hyphen
    cleaned = normalize_unicode(text).replace("$", "")
    match = NUMBER.search(cleaned)
    if not match:
        return None
    try:
        return Decimal(match.group().replace(",", ""))
    except InvalidOperation:
        return None


def is_choice_reference(reference: str) -> bool:
    """True if the reference is a single multiple-choice letter A-D."""
    return len(reference.strip()) == 1 and reference.strip().upper() in "ABCD"


def extract_choice_letter(value: str) -> Optional[str]:
    """The first standalone capital A-D in an answer value, else a lone letter like "b" or "(b)"."""
    match = CHOICE_LETTER.search(value)
    if match:
        return match.group()
    bare = value.strip(" .()[]")
    if len(bare) == 1 and bare.upper() in "ABCD":
        return bare.upper()
    return None


def reference_number(reference: str) -> Optional[Decimal]:
    """The reference as a number, if the whole reference is one number (with optional $ and commas)."""
    cleaned = reference.strip().replace("$", "")
    if not NUMBER.fullmatch(cleaned):
        return None
    return Decimal(cleaned.replace(",", ""))


def normalize_text(text: str) -> str:
    """Exact-match normalization: Unicode spaces/dashes, case, outer whitespace, one trailing period,
    spaces around + : - / ,"""
    text = normalize_unicode(text).strip().lower()
    if text.endswith("."):
        text = text[:-1].rstrip()
    return TIGHT_PUNCTUATION.sub(r"\1", text)


def score_final_answer(output: str, reference: str) -> ScoreResult:
    """Grade only the value on the model's last "Answer:" line.

    Multiple-choice references (A-D) match the first standalone A-D letter in the value.
    Numeric references are compared as numbers, so "18", "18.0", "$18", and "18 apples" all match 18.
    Other references are compared as text using the exact-match normalization rules.
    Only the answer value is looked at, never the reasoning before it.
    """
    output, reference = normalize_unicode(output), normalize_unicode(reference)
    answer = extract_final_answer(output)
    if answer is None:
        return ScoreResult(False, None, "no 'Answer:' line")

    if is_choice_reference(reference):
        letter = extract_choice_letter(answer)
        if letter is None:
            return ScoreResult(False, answer, "no A-D letter after 'Answer:'")
        expected_letter = reference.strip().upper()
        if letter == expected_letter:
            return ScoreResult(True, letter, "letters match")
        return ScoreResult(False, letter, f"expected {expected_letter}, got {letter}")

    expected = reference_number(reference)
    if expected is not None:
        got = parse_number(answer)
        if got is None:
            return ScoreResult(False, answer, "no number after 'Answer:'")
        if got == expected:
            return ScoreResult(True, str(got), "numbers match")
        return ScoreResult(False, str(got), f"expected {expected}, got {got}")

    if not answer:
        return ScoreResult(False, answer, "nothing after 'Answer:'")
    if normalize_text(answer) == normalize_text(reference):
        return ScoreResult(True, answer, "text matches")
    return ScoreResult(False, answer, f"expected {reference!r}, got {answer!r}")


# ---------- exact_match / label_match ----------

def score_exact_match(output: str, reference: str) -> ScoreResult:
    """The whole output must equal the reference after exact-match normalization (schema.md)."""
    got = normalize_text(output)
    if got == normalize_text(reference):
        return ScoreResult(True, got, "text matches")
    return ScoreResult(False, got, f"expected {reference!r}, got {output.strip()[:80]!r}")


def score_label_match(output: str, reference: str) -> ScoreResult:
    """The output must be exactly the reference label, using the exact-match normalization."""
    got = normalize_text(output)
    if got == normalize_text(reference):
        return ScoreResult(True, got, "label matches")
    return ScoreResult(False, got, f"expected label {reference!r}, got {output.strip()[:80]!r}")


# ---------- field_check ----------

# A single surrounding ```json ... ``` fence is allowed around the JSON
JSON_FENCE = re.compile(r"^\s*```(?:json)?[ \t]*\n?(.*?)\n?[ \t]*```\s*$", re.DOTALL | re.IGNORECASE)


def parse_json_output(output: str):
    """The JSON value in a model's output, allowing one surrounding code fence. Raises ValueError."""
    text = output.strip()
    fenced = JSON_FENCE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    return json.loads(text)


def fuzzy_text(text: str) -> str:
    """Fuzzy-field normalization: lowercase, punctuation to spaces, whitespace collapsed."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def _normalize_strings(value):
    """normalize_unicode applied to every string value in a JSON structure (keys are left alone)."""
    if isinstance(value, str):
        return normalize_unicode(value)
    if isinstance(value, list):
        return [_normalize_strings(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize_strings(v) for k, v in value.items()}
    return value


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _compare_fuzzy(expected, actual, path: str, problems: list) -> None:
    if isinstance(expected, list):
        if not (isinstance(actual, list) and all(isinstance(v, str) for v in actual)):
            problems.append(f"{path}: expected a list of strings")
        elif {fuzzy_text(v) for v in expected} != {fuzzy_text(v) for v in actual}:
            problems.append(f"{path}: expected {expected!r}, got {actual!r}")
    elif expected is None:
        if actual is not None:
            problems.append(f"{path}: expected null, got {actual!r}")
    elif not isinstance(actual, str) or fuzzy_text(actual) != fuzzy_text(expected):
        problems.append(f"{path}: expected {expected!r}, got {actual!r}")


def _compare(expected, actual, path: str, fuzzy: set, problems: list) -> None:
    """Record every place where actual differs from expected. List items share their list's path."""
    label = path or "(root)"
    if path in fuzzy:
        _compare_fuzzy(expected, actual, label, problems)
    elif isinstance(expected, dict):
        if not isinstance(actual, dict):
            problems.append(f"{label}: expected an object, got {type(actual).__name__}")
            return
        missing, extra = set(expected) - set(actual), set(actual) - set(expected)
        if missing:
            problems.append(f"{label}: missing keys {sorted(missing)}")
        if extra:
            problems.append(f"{label}: unexpected keys {sorted(extra)}")
        for key in expected:
            if key in actual:
                _compare(expected[key], actual[key], f"{path}.{key}" if path else key, fuzzy, problems)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            problems.append(f"{label}: expected a list of {len(expected)} items, got {actual!r}"[:200])
            return
        for item_expected, item_actual in zip(expected, actual):
            _compare(item_expected, item_actual, path, fuzzy, problems)
    elif isinstance(expected, bool) or expected is None:
        if actual is not expected:
            problems.append(f"{label}: expected {json.dumps(expected)}, got {actual!r}")
    elif _is_number(expected):
        if not _is_number(actual) or actual != expected:
            problems.append(f"{label}: expected {expected!r}, got {actual!r}")
    elif actual != expected:
        problems.append(f"{label}: expected {expected!r}, got {actual!r}")


def score_field_check(output: str, reference: dict, fuzzy_fields=()) -> ScoreResult:
    """Every field must match the reference: fuzzy_fields after normalization, everything else exactly.

    The output must be one JSON object (optionally inside a ```json fence) with exactly the
    reference's keys at every level. Lists keep their order, except fuzzy lists of strings,
    which are compared as unordered sets. Numbers compare by value (58000 == 58000.0).
    """
    try:
        actual = parse_json_output(output)
    except ValueError:
        return ScoreResult(False, None, "output is not valid JSON")
    problems = []
    # Unicode spaces and dashes are normalized in every string, strict fields included
    _compare(_normalize_strings(reference), _normalize_strings(actual), "", set(fuzzy_fields or ()), problems)
    if problems:
        return ScoreResult(False, json.dumps(actual, ensure_ascii=False)[:300], "; ".join(problems)[:500])
    return ScoreResult(True, json.dumps(actual, ensure_ascii=False)[:300], "all fields match")


# ---------- Dispatch ----------

def score_record(record: dict, output: str) -> Optional[ScoreResult]:
    """Grade an output with the record's scoring method. None for judge records (no automatic scorer)."""
    scoring = record["scoring"]
    if scoring == "judge":
        return None
    if scoring == "final_answer":
        return score_final_answer(output, record["reference"])
    if scoring == "exact_match":
        return score_exact_match(output, record["reference"])
    if scoring == "label_match":
        return score_label_match(output, record["reference"])
    if scoring == "field_check":
        return score_field_check(output, record["reference"], record.get("fuzzy_fields", []))
    raise ValueError(f"unknown scoring method {scoring!r}")
