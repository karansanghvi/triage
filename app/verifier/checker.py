"""Answer checker for the cascade: is this answer obviously bad?

It sees only what a live request has: the conversation and the model's answer. It NEVER uses a reference
answer (no GSM8K number, MMLU letter or expected JSON fields), because live requests don't have one.
So it can't tell if an answer is right; it only catches answers that are clearly not usable:

    error / empty / cut off       the call failed, returned nothing, or hit its token limit
    wrong language                an English conversation answered mostly in another script
    refusal                       a short "I can't help with that"
    broken format                 JSON was asked for but isn't valid; an "Answer:" line was asked for but
                                  is missing; multiple choice was offered but no A-D letter was picked
    countable constraints         (optional) "under 50 words", "exactly 3 bullets", via constraints.py;
                                  off by default because live requests don't come with a constraint list

check_answer() returns ok=True when nothing looks wrong, plus the reasons when something does.
"""
import re
from dataclasses import dataclass, field
from typing import Optional

from app.verifier.constraints import check_constraint
from app.verifier.scorers import extract_choice_letter, extract_final_answer, parse_json_output

REFUSAL = re.compile(
    r"\b(?:i(?:'m| am) sorry,? but i (?:can(?:'|no)t|won't)|i can(?:'|no)t (?:help|assist|provide|comply)|"
    r"i(?:'m| am) (?:not able|unable) to (?:help|assist|provide)|as an ai(?: language model)?,? i)",
    re.IGNORECASE)
JSON_REQUEST = re.compile(r"\bjson\b", re.IGNORECASE)
ANSWER_LINE_REQUEST = re.compile(r"answer:", re.IGNORECASE)
# Multiple-choice options at the start of a line: "A) ...", "A. ...", "(A) ..."
CHOICE_OPTION = re.compile(r"^\s*\(?([A-D])[).:]\s", re.MULTILINE)


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    reasons: tuple = field(default_factory=tuple)


def latin_share(text: str) -> Optional[float]:
    """Share of letters that are Latin-script (English-like). None if there are too few letters to tell."""
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 20:
        return None
    return sum(ord(c) < 0x250 for c in letters) / len(letters)


def conversation_text(messages: list) -> str:
    """All message contents (system included: format instructions often live there)."""
    return "\n".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))


def check_answer(messages: list, answer: Optional[str], *, truncated: bool = False, error=None,
                 constraints: Optional[list] = None) -> CheckResult:
    reasons = []
    text = (answer or "").strip()
    asked = conversation_text(messages)

    # 1. The call itself
    if error:
        reasons.append("call failed")
    if not text:
        reasons.append("empty answer")
    if truncated:
        reasons.append("cut off at the token limit")
    if not text:
        return CheckResult(False, tuple(reasons))

    # 2. Language: an English conversation should get a mostly Latin-script answer
    asked_latin, answer_latin = latin_share(asked), latin_share(text)
    if asked_latin is not None and asked_latin > 0.9 and answer_latin is not None and answer_latin < 0.6:
        reasons.append("answered in a different language/script")

    # 3. Refusal: only short answers, so a long helpful answer that mentions "I can't" isn't flagged
    if len(text) < 400 and REFUSAL.search(text):
        reasons.append("looks like a refusal")

    # 4. Format the conversation asked for
    if JSON_REQUEST.search(asked):
        try:
            if not isinstance(parse_json_output(text), (dict, list)):
                reasons.append("JSON requested but the answer isn't a JSON object or list")
        except ValueError:
            reasons.append("JSON requested but the answer isn't valid JSON")
    if ANSWER_LINE_REQUEST.search(asked):
        final = extract_final_answer(text)
        if not final:
            reasons.append("'Answer:' line requested but missing or empty")
        elif len(set(CHOICE_OPTION.findall(asked))) >= 3 and extract_choice_letter(final) is None:
            reasons.append("multiple choice, but no A-D letter after 'Answer:'")

    # 5. Countable constraints (optional)
    for constraint in constraints or []:
        if check_constraint(constraint, text) is False:
            reasons.append(f"breaks constraint: {constraint}")

    return CheckResult(not reasons, tuple(reasons))