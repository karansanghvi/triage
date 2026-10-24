"""Code-checked constraints: exact counts the judge model can't be trusted to do.

LLM judges miscount words, characters, bullets and sentences. For constraints that can be parsed with
confidence and that apply to the whole answer, check_constraint() returns True (met) or False (broken),
and the judge's own verdict for that constraint is overridden. Anything else returns None and stays with
the judge: scoped constraints ("each bullet under 12 words", "Customer: exactly 2 sentences"), fuzzy ones
("plain language"), and answers whose structure makes a count unreliable (sentences in a bulleted list,
nested bullets). The counting rules are documented in data/schema.md ("Code-checked constraints").
"""
import re
from typing import Optional

from app.verifier.scorers import normalize_unicode

NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                "nine": 9, "ten": 10}
NUM = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"

# A whitespace-separated token is a word if it contains at least one letter or digit
WORD = re.compile(r"[^\W_]", re.UNICODE)
# A list item: "-", "*", "•", "+" or "1." / "1)" at the start of a line
BULLET_LINE = re.compile(r"^(\s*)(?:[-*•+]|\d+[.)])\s+\S")
# Abbreviations whose period doesn't end a sentence
ABBREVIATIONS = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e|U\.S|U\.K|a\.m|p\.m|No|Inc|Ltd|Co)\.",
                           re.IGNORECASE)
SENTENCE_END = re.compile(r"[.!?]+(?=[\"')\]]*(?:\s|$))")


def _n(text: str) -> int:
    return int(text) if text.isdigit() else NUMBER_WORDS[text.lower()]


# ---------- Counting ----------

def count_words(answer: str) -> int:
    """Whitespace-separated tokens that contain a letter or digit. "18-inch" is one word; "-" or "**" alone
    are not words. Markdown markers attached to a word don't matter ("**Answer:**" is one word)."""
    return sum(1 for token in normalize_unicode(answer).split() if WORD.search(token))


def count_characters(answer: str) -> int:
    """Characters in the answer with leading and trailing whitespace removed, spaces inside included."""
    return len(answer.strip())


def bullet_lines(answer: str) -> Optional[list]:
    """The bullet lines, or None when the list structure isn't reliable (nested bullets or code fences)."""
    if "```" in answer:
        return None
    lines = [m for m in (BULLET_LINE.match(line) for line in answer.splitlines()) if m]
    if any(m.group(1) for m in lines):          # an indented bullet: a nested list
        return None
    return lines


def count_sentences(answer: str) -> Optional[int]:
    """Sentences in plain prose, or None when the answer isn't plain prose (bullets, headings, code, several
    paragraphs). A sentence ends at . ! or ? followed by whitespace or the end; periods in abbreviations
    (e.g., Dr.) and decimals (3.5) don't end a sentence."""
    text = normalize_unicode(answer).strip()
    if not text or "```" in text or bullet_lines(text) or re.search(r"^\s*#", text, re.MULTILINE):
        return None
    if re.search(r"\n\s*\n", text):
        return None
    text = ABBREVIATIONS.sub(lambda m: m.group(0).replace(".", ""), text)
    pieces = [p for p in SENTENCE_END.split(text) if WORD.search(p)]
    return len(pieces)


# ---------- Parsing constraints ----------

def _check_count(value: Optional[int], op: str, a: int, b: Optional[int] = None) -> Optional[bool]:
    if value is None:
        return None
    if op == "<":
        return value < a
    if op == "<=":
        return value <= a
    if op == "==":
        return value == a
    return a <= value <= b                      # range


# (pattern, what to count, comparison). Patterns must match the WHOLE constraint, so scoped constraints
# like "each bullet under 12 words" or "Customer: exactly 2 sentences" never match.
RULES = [
    (rf"under {NUM} words", "words", "<"),
    (rf"fewer than {NUM} words", "words", "<"),
    (rf"(?:max|maximum|at most) {NUM} words", "words", "<="),
    (rf"{NUM} words or fewer", "words", "<="),
    (rf"{NUM}\s*[-–]\s*{NUM} words", "words", "range"),
    (rf"under {NUM} characters(?: including spaces)?", "characters", "<"),
    (rf"(?:max|maximum|at most) {NUM} characters(?: including spaces)?", "characters", "<="),
    (rf"(?:single sms, )?{NUM} characters or fewer(?: including (?:spaces|the url))?", "characters", "<="),
    (rf"exactly {NUM} bullets?(?: points?)?", "bullets", "=="),
    (rf"at most {NUM} bullets?(?: points?)?", "bullets", "<="),
    (rf"exactly {NUM} sentences?", "sentences", "=="),
    (rf"(one) sentence", "sentences", "=="),
    (rf"(?:max|maximum|at most) {NUM} sentences?", "sentences", "<="),
    (rf"{NUM}\s*[-–]\s*{NUM} sentences?", "sentences", "range"),
]


def parse_constraint(constraint: str) -> Optional[tuple]:
    """(what, op, a, b) if this constraint is one we check in code, else None."""
    text = normalize_unicode(constraint).strip().lower()
    for pattern, what, op in RULES:
        match = re.fullmatch(pattern, text)
        if match:
            numbers = [_n(g) for g in match.groups()]
            return what, op, numbers[0], numbers[1] if len(numbers) > 1 else None
    return None


def check_constraint(constraint: str, answer: str) -> Optional[bool]:
    """True/False if the constraint can be checked exactly in code, None to leave it to the judge."""
    parsed = parse_constraint(constraint)
    if parsed is None:
        return None
    what, op, a, b = parsed
    if what == "words":
        value = count_words(answer)
    elif what == "characters":
        value = count_characters(answer)
    elif what == "bullets":
        lines = bullet_lines(answer)
        value = None if lines is None else len(lines)
    else:
        value = count_sentences(answer)
    return _check_count(value, op, a, b)


def describe(constraint: str, answer: str) -> str:
    """A short note for the judgment record, e.g. 'words=88 (limit < 80)'."""
    parsed = parse_constraint(constraint)
    if parsed is None:
        return ""
    what, op, a, b = parsed
    counts = {"words": count_words, "characters": count_characters,
              "bullets": lambda t: (lambda l: None if l is None else len(l))(bullet_lines(t)),
              "sentences": count_sentences}
    limit = f"{a}-{b}" if op == "range" else f"{op} {a}"
    return f"{what}={counts[what](answer)} (limit {limit})"


def cap_note(constraint: str, answer: str) -> str:
    """A plain-English note for a broken countable constraint, e.g. "151 words, limit under 150"."""
    parsed = parse_constraint(constraint)
    if parsed is None:
        return constraint
    what, op, a, b = parsed
    value = {"words": count_words, "characters": count_characters, "sentences": count_sentences,
             "bullets": lambda t: (lambda l: None if l is None else len(l))(bullet_lines(t))}[what](answer)
    limit = {"<": f"under {a}", "<=": f"at most {a}", "==": f"exactly {a}"}.get(op, f"{a}-{b}")
    return f"{value} {what}, limit {limit}"
