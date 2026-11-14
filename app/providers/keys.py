"""Per-request provider API keys for BYOK (bring your own key).

A KeyContext lives for exactly one request. It is never stored, cached,
or logged, and its repr hides every key value.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Mapping

# Whose keys a request uses: "byok" = the user's own (only those providers); "server" = Triage's own env keys and
# routing.yaml's server_providers. (Not the request mode: RequestRow.mode is "api" | "byok" | "demo".)
Mode = Literal["byok", "server"]

REDACTED = "[redacted]"
# Common provider key shapes, removed even when we don't know the exact value (e.g. a key echoed back by a provider
# in a different form). Minimum lengths keep ordinary words like "sk-learn" untouched.
_KEY_PATTERNS = re.compile(
    r"sk-ant-[A-Za-z0-9_\-]{8,}"        # Anthropic
    r"|sk-[A-Za-z0-9_\-]{8,}"           # OpenAI (sk-..., sk-proj-...)
    r"|gsk_[A-Za-z0-9]{8,}"             # Groq
    r"|AIza[0-9A-Za-z_\-]{20,}"         # Google / Gemini
)


def redact(text, secrets=()) -> str:
    """`text` with every exact secret value (e.g. a KeyContext's keys) and every common key pattern replaced by
    [redacted]. Use it on any provider error text before it becomes an Attempt message, an API error or a log."""
    if not text:
        return text
    text = str(text)
    for secret in sorted((s for s in secrets if s), key=len, reverse=True):
        text = text.replace(secret, REDACTED)
    return _KEY_PATTERNS.sub(REDACTED, text)

@dataclass(frozen=True, repr=False, eq=False)
class KeyContext:
    mode: Mode
    keys: Mapping[str, str] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        # Copy and make read-only so nothing downstream can add or swap keys 
        object.__setattr__(self, "keys", MappingProxyType(dict(self.keys)))
    
    @property
    def providers(self) -> frozenset[str]:
        """Provider names this request is allowed to use"""
        return frozenset(self.keys)
    
    def has(self, provider: str) -> bool:
        return provider in self.keys
    
    def key_for(self, provider: str) -> str | None:
        return self.keys.get(provider)

    @property
    def secrets(self) -> tuple:
        """Every key value, for redact()."""
        return tuple(self.keys.values())
    
    def __repr__(self) -> str:
        hidden = ", ".join(f"{p}=***" for p in sorted(self.keys))
        return f"KeyContext(mode={self.mode!r}, keys={{{hidden}}})"