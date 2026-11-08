"""Agreement check: do two answers reach the same final answer?

Two models solving the same problem independently and landing on the same answer is a strong sign the
answer is right; disagreement means at least one is wrong. This judges CONTENT without an answer key,
which is what the format-only checker couldn't do.

It only works when an answer has a short, comparable final result:
    multiple choice   the A-D letter after "Answer:"
    numeric           the number after "Answer:" (18, 18.0, $18 all equal)
    other "Answer:"   the text after "Answer:", normalized
    JSON requested    the parsed JSON, with strings normalized (key order and spacing don't matter)
Anything else (essays, summaries, code) returns None: "not comparable", so callers fall back to another
check. An answer that is cut off, or doesn't contain the requested format, also gives None.
"""
import json
from typing import Optional

from app.verifier.checker import ANSWER_LINE_REQUEST, CHOICE_OPTION, JSON_REQUEST, conversation_text
from app.verifier.scorers import (extract_choice_letter, extract_final_answer, normalize_text, parse_json_output,
                                  parse_number)


def comparable(messages: list) -> bool:
    """True if this conversation asks for an answer we can compare (an "Answer:" line or JSON)."""
    asked = conversation_text(messages)
    return bool(JSON_REQUEST.search(asked) or ANSWER_LINE_REQUEST.search(asked))


def _canonical(value):
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, list):
        return [_canonical(v) for v in value]
    if isinstance(value, dict):
        return {k: _canonical(v) for k, v in value.items()}
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def final_key(messages: list, answer: Optional[str], truncated: bool = False) -> Optional[str]:
    """A normalized final answer that can be compared between models, or None if there isn't one."""
    if not answer or truncated:
        return None
    asked = conversation_text(messages)
    if JSON_REQUEST.search(asked):
        try:
            return "J:" + json.dumps(_canonical(parse_json_output(answer)), sort_keys=True)
        except ValueError:
            return None
    if ANSWER_LINE_REQUEST.search(asked):
        value = extract_final_answer(answer)
        if not value:
            return None
        if len(set(CHOICE_OPTION.findall(asked))) >= 3:
            letter = extract_choice_letter(value)
            return f"L:{letter}" if letter else None
        number = parse_number(value)
        if number is not None:
            return f"N:{number.normalize()}"
        return "T:" + normalize_text(value)
    return None


def agree(a: Optional[str], b: Optional[str]) -> bool:
    """Both answers have a final answer, and it's the same one."""
    return a is not None and a == b