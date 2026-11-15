"""Checking a provider key with one free call: each provider's list-models endpoint (Phase 5, accounts).

A 200 means the provider ACCEPTED the key ("key accepted"). It says nothing about credit, quota or which models
the key can use: list-models works for keys that can't afford a single answer.

    openai     GET https://api.openai.com/v1/models                        Authorization: Bearer <key>
    anthropic  GET https://api.anthropic.com/v1/models                     x-api-key: <key>, anthropic-version
    groq       GET https://api.groq.com/openai/v1/models                   Authorization: Bearer <key>
    gemini     GET https://generativelanguage.googleapis.com/v1beta/models x-goog-api-key: <key> (never in the URL)

Outcomes (status, and whether the provider gave a definite answer):
    200                                 valid
    401 / 403                           invalid (the provider rejected the key)
    400 saying API_KEY_INVALID          invalid (Gemini's older way of saying the same)
    timeout, network error, 429, 5xx    unverified (couldn't tell; the key may be fine)
    anything else                       unverified, with the status in the detail

The key goes only into the request header. Every provider message is passed through redact() with the key as a
secret, shortened, and no exception that could carry the request (and so the key) is ever re-raised.
"""
from dataclasses import dataclass
from typing import Optional

import httpx

from app.providers.anthropic_provider import API_VERSION as ANTHROPIC_VERSION
from app.providers.keys import redact

TIMEOUT_SECONDS = 10.0
MAX_DETAIL = 160
NAMES = {"openai": "OpenAI", "anthropic": "Anthropic", "groq": "Groq", "gemini": "Gemini"}
ENDPOINTS = {
    "openai": "https://api.openai.com/v1/models",
    "anthropic": "https://api.anthropic.com/v1/models",
    "groq": "https://api.groq.com/openai/v1/models",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models",
}


def _headers(provider: str, key: str) -> dict:
    if provider == "anthropic":
        return {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION}
    if provider == "gemini":
        return {"x-goog-api-key": key}
    return {"Authorization": f"Bearer {key}"}


@dataclass(frozen=True)
class Verdict:
    status: str                    # valid | invalid | unverified
    detail: Optional[str]          # short, redacted; None when valid
    answered: bool                 # the provider gave a definite answer (valid or invalid)
    http_status: Optional[int] = None   # the provider's status code, when it answered at all


def _provider_message(response: httpx.Response, key: str) -> str:
    """The provider's own error message, if any, redacted and shortened."""
    try:
        body = response.json()
        error = body.get("error") if isinstance(body, dict) else None
        message = error.get("message") if isinstance(error, dict) else (error if isinstance(error, str) else None)
    except Exception:
        message = None
    if not message:
        return ""
    return redact(" ".join(str(message).split()), (key,))[:MAX_DETAIL]


def _short(text: str, key: str) -> str:
    return redact(text, (key,))[:MAX_DETAIL]


def verify_key(provider: str, key: str, client: Optional[httpx.Client] = None) -> Verdict:
    """One GET to the provider's list-models endpoint with `key`. Blocking: call from a worker thread.
    client: an httpx.Client to use (tests pass one with a MockTransport); None = a new one, closed after."""
    name = NAMES[provider]
    own = client is None
    client = client or httpx.Client(timeout=TIMEOUT_SECONDS)
    try:
        response = client.get(ENDPOINTS[provider], headers=_headers(provider, key), timeout=TIMEOUT_SECONDS)
    except httpx.TimeoutException:
        return Verdict("unverified", f"{name} didn't answer within {int(TIMEOUT_SECONDS)} s", False)
    except Exception as error:               # network errors: only the type, the exception could carry the request
        return Verdict("unverified", _short(f"couldn't reach {name} ({type(error).__name__})", key), False)
    finally:
        if own:
            client.close()
    status = response.status_code
    if status == 200:
        return Verdict("valid", None, True, 200)
    message = _provider_message(response, key)
    detail = _short(f"{name} answered {status}" + (f": {message}" if message else ""), key)
    if status in (401, 403):
        return Verdict("invalid", detail, True, status)
    if status == 400 and "API_KEY_INVALID" in response.text:
        return Verdict("invalid", detail, True, status)
    return Verdict("unverified", detail, False, status)


def fake_verify_key(provider: str, key: str) -> Verdict:
    """Fake-model mode (TRIAGE_FAKE_MODELS=1) makes no API calls, so keys aren't checked: always unverified."""
    return Verdict("unverified", "not checked in fake mode", False)
