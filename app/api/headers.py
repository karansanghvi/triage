"""Triage request headers: X-Triage-Quality, X-Triage-Task, X-Triage-Verify. Bad values -> 400."""
from typing import Optional
from collections.abc import Mapping

from app.api.errors import APIError, bad_request
from app.router.features import TASK_TYPES, normalize_task_type
from app.router.quality import QualityError, QualitySettings, resolve_tau
from app.providers.keys import KeyContext

QUALITY, TASK, VERIFY = "X-Triage-Quality", "X-Triage-Task", "X-Triage-Verify"
VERIFY_MODES = ("off", "sync")
# Lowercase header name -> provider name used by the registry and adapters.
KEY_HEADERS = {
    "x-openai-key": "openai",
    "x-anthropic-key": "anthropic",
    "x-groq-key": "groq",
    "x-gemini-key": "gemini",
}
# Provider -> the header as shown in error messages (the same headers as KEY_HEADERS, which matches any case)
PROVIDER_KEY_HEADERS = {"openai": "X-OpenAI-Key", "anthropic": "X-Anthropic-Key", "groq": "X-Groq-Key",
                        "gemini": "X-Gemini-Key"}
assert {h.lower(): p for p, h in PROVIDER_KEY_HEADERS.items()} == KEY_HEADERS


def parse_quality(value: Optional[str], settings: QualitySettings) -> float:
    try:
        return resolve_tau(value, settings)
    except QualityError:
        options = " | ".join(settings.presets)
        raise bad_request(f"{QUALITY} must be {options} or a number from 0 to 1, got {value!r}", QUALITY) from None


def parse_task(value: Optional[str]) -> Optional[str]:
    if value is None or not value.strip():
        return None
    task = normalize_task_type(value)
    if task is None:
        raise bad_request(f"{TASK} must be one of {', '.join(TASK_TYPES)} (or a known alias), got {value!r}", TASK)
    return task


def parse_verify(value: Optional[str], stream: bool) -> str:
    mode = (value or "off").strip().lower()
    if mode not in VERIFY_MODES:
        raise bad_request(f"{VERIFY} must be off or sync, got {value!r}", VERIFY)
    if mode == "sync" and stream:
        raise bad_request(f"{VERIFY}: sync can't be combined with stream=true: the answer has to be checked "
                          "before any of it is sent", VERIFY, "unsupported_combination")
    return mode


def check_verify_model(mode: str, model: str) -> None:
    """Sync verification may escalate to a stronger model, so it only works when Triage picks the model."""
    if mode == "sync" and model != "auto":
        raise bad_request(f"{VERIFY}: sync only works with model \"auto\". With a named model ({model!r}) Triage "
                          "won't switch to a model you didn't choose; drop the header or use model \"auto\".",
                          VERIFY, "unsupported_combination")

def resolve_keys(headers: Mapping[str, str]) -> KeyContext:
    """Build this request's KeyContext from its headers.
    
    Any key sent -> "byok" with only those providers. 
    No keys sent -> "demo" with no keys (server keys are added in Step 2)
    Makes no network calls and never checks whether a key is valid 
    """
    keys: dict[str, str] = {}
    for header, provider in KEY_HEADERS.items():
        value = (headers.get(header) or "").strip()
        if value:
            keys[provider] = value
    return KeyContext("byok", keys) if keys else KeyContext("demo")
