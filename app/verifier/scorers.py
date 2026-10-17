"""Scorers: grade a model's output against a prompt's reference (see data/schema.md)."""
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

# Spaces around these characters are ignored when comparing text answers (exact-match rules)
TIGHT_PUNCTUATION = re.compile(r"\s*([+:\-/,])\s*")


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
    # Drop $ so "-$18" and "$-18" both read as -18; Unicode minus becomes a plain hyphen
    cleaned = text.replace("$", "").replace("−", "-")
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
    """Exact-match normalization: case, outer whitespace, one trailing period, spaces around + : - / ,"""
    text = text.strip().lower()
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
