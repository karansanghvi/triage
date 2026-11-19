"""Small talk: a pure greeting or pleasantry goes to the cheapest tier without asking the classifier.

Only when the WHOLE latest user message, normalized, is one of the phrases in routing.yaml (`small_talk.phrases`).
Normalized = trimmed, lowercased, inner whitespace collapsed, and trailing punctuation, symbols and emoji removed
("Thanks!!", "Good   Morning 👋" and "ok." match; "hi, can you help with X" doesn't).
"""
from typing import Iterable, Optional


def normalize(text: str) -> str:
    text = " ".join(text.strip().lower().split())
    end = len(text)
    while end > 0 and not text[end - 1].isalnum():   # punctuation, emoji, variation selectors, spaces, ...
        end -= 1
    return text[:end]


def latest_user_text(messages: list) -> Optional[str]:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            return content if isinstance(content, str) else None
    return None


def is_small_talk(messages: list, phrases: Iterable[str]) -> bool:
    """phrases: already normalized (routing.yaml is normalized when it is loaded). Empty = never."""
    text = latest_user_text(messages)
    return text is not None and normalize(text) in phrases
