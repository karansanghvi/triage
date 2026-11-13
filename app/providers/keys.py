"""Per-request provider API keys for BYOK (bring your own key).

A KeyContext lives for exactly one request. It is never stored, cached,
or logged, and its repr hides every key value.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Mapping

Mode = Literal["byok", "demo"]

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
    
    def __repr__(self) -> str:
        hidden = ", ".join(f"{p}=***" for p in sorted(self.keys))
        return f"KeyContext(mode={self.mode!r}, keys={{{hidden}}})"