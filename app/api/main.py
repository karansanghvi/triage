"""Triage's OpenAI-compatible API (Phase 4 Step 5).

Run it:
    TRIAGE_API_KEYS=dev-key TRIAGE_ADMIN_KEYS=admin-key TRIAGE_HASH_SECRET=<random> TRIAGE_FAKE_MODELS=1 \
        uvicorn app.api.main:app --no-access-log
(--no-access-log always: uvicorn's access log prints client IPs, and the privacy policy says Triage logs none.
Drop TRIAGE_FAKE_MODELS for real models; TRIAGE_CACHE=off disables the response cache. Fake mode uses its own
cache and log, data/cache/fake_responses.sqlite3 and data/logs/fake_requests.sqlite3, never the real ones)

Endpoints
    POST /v1/chat/completions     OpenAI chat completions, stream or not; model "auto" or a models.yaml name
    GET  /v1/models               "auto" plus every model, with prices
    GET  /v1/stats                savings summary (your key's requests; an admin key sees all)
    GET  /v1/stats/public         no auth: live aggregates of auto-routed requests (app/api/public_stats.py)
    GET  /v1/requests/{id}        one logged request, only for the key that made it (public access: a keyless
                                  request's outcome and cost, by its id, without a key)
    GET  /v1/routing-config       the config in use (admin)
    PUT  /v1/routing-config       update quality presets/default and backup models live (admin; not persisted)
    GET  /v1/admin/forwarded-for  the caller's own proxy headers, to set TRIAGE_TRUSTED_PROXY_HOPS (admin; DEPLOY.md)

Response headers: X-Triage-Request-Id, -Model, -Tier, -Reason, -Fallback, -Cache, -Verified, -Escalated,
-Check, -P-Low, -P-Medium and -Tau (model "auto" only, 6 decimals), and on non-streaming responses -Cost,
-Strong-Cost and -Saved (including any checks and escalation; -Saved for model "auto" only). A stream's headers are sent once its first token
exists, so model, tier and fallback are real; its cost isn't known yet, so -Cost, -Strong-Cost and -Saved are left
out (never faked). The stream itself carries the same "triage" object as a non-streaming response: routing only on
its first chunk, complete (with cost and savings) on its finish_reason chunk. The figures are also in
GET /v1/requests/{id} once the stream ends, and in /v1/stats.
"""
import asyncio
import contextlib
import dataclasses
import hashlib
import json
import logging
import os
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

import anyio
from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import true as sql_true
from starlette.background import BackgroundTask

from app.api import core
from app.api.auth import (ADMIN_KEYS_ENV, API_KEYS_ENV, Auth, Caller, accounts_from_env, keys_from_env,
                          public_from_env)
from app.accounts.credentials import API_KEY_PREFIX, SESSION_PREFIX
from app.api.config_state import NOT_PERSISTED_NOTE, ConfigUpdateError
from app.api.demo import DemoLimitError, client_ip, visitor_id
from app.api.errors import APIError, bad_request, stream_error_event
from app.api.headers import (PROVIDER_KEY_HEADERS, check_verify_model, parse_prefer_provider, parse_quality,
                             parse_task, parse_verify, resolve_keys)
from app.accounts.limits import AccountLimitError
from app.api.request_log import RequestRow, embedding_to_bytes, is_training_eligible, prompt_hmac
from app.router.small_talk import is_small_talk
from app.api import health, maintenance
from app.providers import lazy as lazy_providers
from app.api import public_stats, usage_sql
from app.api.stats import summarize
from app.api.verification import Served, verify_sync
from app.cache import is_cacheable
from app.providers.base import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, _build_response
from app.providers.common import ProviderError
from app.providers.keys import KeyContext, redact
from app.reliability import AllModelsFailed, ModelNotAllowed

logger = logging.getLogger("triage.api")
# One JSON line per request (Step 10): ids, routing, status, latency and cost only. Never keys or text.
request_logger = logging.getLogger("triage.requests")
JSON_LOG_FIELDS = ("request_id", "timestamp", "key_id", "requested_model", "stream", "tier", "served_model",
                   "fell_back", "cache_hit", "status", "http_status", "error_kind", "latency_ms", "ttft_ms",
                   "price_as_cost", "saved", "verified", "escalated", "decided_by", "routed_by",
                   "training_eligible")


def json_log_line(row: RequestRow) -> str:
    return json.dumps({k: getattr(row, k) for k in JSON_LOG_FIELDS}, separators=(",", ":"))


def _ensure_json_handler() -> None:
    """Print the JSON lines to stderr unless logging is already configured for them (added once)."""
    if not request_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        request_logger.addHandler(handler)
        request_logger.setLevel(logging.INFO)
        request_logger.propagate = False
    auth_logger = logging.getLogger("triage.auth")      # sign-in events: event, outcome, visitor hash only
    if not auth_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        auth_logger.addHandler(handler)
        auth_logger.setLevel(logging.INFO)


class _RecordingPredictor:
    """Wraps the shared predictor for ONE request and keeps the Prediction it returned, so the log can store
    the features and embedding the classifier already computed (nothing is computed twice). route_and_stream
    gets a per-request copy of deps holding this wrapper; it is otherwise the same predictor."""

    def __init__(self, predictor):
        self._predictor = predictor
        self.prediction = None

    def predict_messages(self, messages, task_type=None):
        self.prediction = self._predictor.predict_messages(messages, task_type)
        return self.prediction


def _apply_inputs(row: RequestRow, prediction) -> None:
    """Step 8 storage: only for model="auto" requests that answered, and only "api" requests (an allow-list: never
    BYOK, demo, account or an unknown mode)."""
    if prediction is None or row.requested_model != "auto" or row.status != "ok" or row.mode != "api":
        return
    from app.router.embeddings import MODEL_NAME
    from app.router.features import FEATURE_VERSION
    if prediction.features is not None:
        row.features, row.feature_version = list(prediction.features), FEATURE_VERSION
    if prediction.embedding is not None:
        row.embedding, row.embedding_model = embedding_to_bytes(prediction.embedding), MODEL_NAME

# Request fields Triage understands. Anything else in the body is a 400 naming it, rather than silently ignored.
ACCEPTED_FIELDS = {"model", "messages", "stream", "stream_options", "temperature", "max_tokens",
                   "max_completion_tokens", "n", "user"}


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: Literal["system", "user", "assistant"]
    content: str


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    messages: list[Message] = Field(min_length=1)
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    max_completion_tokens: Optional[int] = Field(default=None, gt=0)
    n: Optional[int] = None
    user: Optional[str] = None


# ---------- helpers ----------

def _money(x: float) -> str:
    return f"{x:.8f}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _header_safe(text: str) -> str:
    return text.encode("latin-1", "replace").decode("latin-1")


def _apply_costs(row: RequestRow, response, deps, providers=None, prefer=None) -> None:
    """providers: a BYOK request's own; the savings baseline is then the strong model among them, after its
    preferred provider (prefer) is moved first (deps.strong_model(providers, prefer)). None = the server's
    baseline, as before."""
    if response is None:
        return
    row.cache_hit = bool(response.cached)
    row.input_tokens, row.output_tokens = response.input_tokens, response.output_tokens
    row.reasoning_tokens, row.tokens_estimated = response.reasoning_tokens, response.tokens_estimated
    row.cost, row.price_as_cost = response.cost, response.reported_cost
    try:
        strong = deps.registry[deps.strong_model(providers, prefer)]
    except LookupError:                # no high-tier model on these providers: savings unknown, not faked
        row.strong_cost = row.saved = None
        return
    row.strong_cost = strong.reported_cost(response.input_tokens, response.output_tokens)
    _credit_saving(row)


def _estimate_stopped_cost(row: RequestRow, decision, routed, text: str, messages: list, deps, providers, prefer,
                           started: float) -> None:
    """A stream the client stopped part-way: the provider bills what it generated, but sends no token counts for
    an unfinished stream. Estimate them from the text (about 4 characters a token, tokens_estimated=True), price
    them like any answer, and add any billed truncated attempts. A cache hit cost nothing; a model that sent no
    text yet is charged for the prompt only."""
    if routed.cached or decision.model not in deps.registry:
        return
    latency_ms = (time.perf_counter() - started) * 1000
    estimate = _build_response(deps.registry[decision.model], text, None, messages, latency_ms, finish_reason="stopped")
    _apply_costs(row, estimate, deps, providers, prefer)
    _add_truncated_costs(row, decision.attempts)


def _credit_saving(row: RequestRow) -> None:
    """saved = strong_cost - price_as_cost, for model="auto" only: with a named model the caller chose it, so
    Triage claims no saving (None; the X-Triage-Saved header is then left out). Stats apply the same rule in
    SQL (app/api/usage_sql.py), so rows logged before it still count right."""
    row.saved = (row.strong_cost - row.price_as_cost
                 if row.strong_cost is not None and row.requested_model == "auto" else None)


MAX_ATTEMPT_MESSAGE_CHARS = 300


def _attempt_dict(a) -> dict:
    """One attempt for the request log. A failed or skipped attempt keeps its message, redacted again here (the
    reliability layer already removed the request's keys) and shortened."""
    out = {"model": a.model_name, "outcome": a.outcome, "error_kind": a.error_kind}
    if a.message:
        out["message"] = redact(a.message)[:MAX_ATTEMPT_MESSAGE_CHARS]
    billed = getattr(a, "response", None)          # only a truncated attempt carries one
    if billed is not None:
        out.update(input_tokens=billed.input_tokens, output_tokens=billed.output_tokens, cost=billed.cost,
                   price_as_cost=billed.reported_cost)
    return out


def _apply_decision(row: RequestRow, d: core.RouteDecision) -> None:
    row.tau, row.p_low, row.p_medium = d.tau, d.p_low, d.p_medium
    row.task_type, row.task_type_guessed = d.task_type, d.task_type_guessed
    row.tier, row.primary_model, row.served_model, row.fell_back = d.tier, d.primary_model, d.model, d.fell_back
    row.attempts = [_attempt_dict(a) for a in d.attempts]
    row.routed_by = d.routed_by


def _add_truncated_costs(row: RequestRow, attempts) -> None:
    """An attempt that used every output token thinking and wrote no answer (then fell back) was still billed:
    add its cost to the request's total. Its tokens are in row.attempts; input/output_tokens stay the served
    answer's, and strong_cost stays the strong model at those tokens."""
    billed = [a.response for a in attempts if getattr(a, "response", None) is not None]
    if not billed:
        return
    row.cost = (row.cost or 0.0) + sum(r.cost for r in billed)
    row.price_as_cost = (row.price_as_cost or 0.0) + sum(r.reported_cost for r in billed)
    _credit_saving(row)


def _triage_headers(d: core.RouteDecision, cache: str, *, model: Optional[str] = None, tier: Optional[str] = None,
                    reason: Optional[str] = None, verified: str = "off", escalated: bool = False,
                    check: str = "none") -> dict:
    return {"X-Triage-Request-Id": d.request_id, "X-Triage-Model": model or d.model, "X-Triage-Tier": tier or d.tier,
            "X-Triage-Reason": _header_safe(reason or d.reason),
            "X-Triage-Fallback": f"{d.primary_model}->{d.model}" if d.fell_back else "none",
            "X-Triage-Cache": cache, "X-Triage-Verified": verified,
            "X-Triage-Escalated": "true" if escalated else "false", "X-Triage-Check": check,
            "X-Triage-Routed-By": d.routed_by,
            "X-Triage-Preferred-Provider": d.preferred_provider or "none", **_probability_headers(d)}


def _six(x: float) -> str:
    return f"{x:.6f}"


def _probability_headers(d: core.RouteDecision) -> dict:
    """The routing numbers as plain numbers (6 decimals). model="auto" only: a named model has no prediction and
    no tau, so these are left out, not faked. With sync verification they stay the first decision's."""
    values = {"X-Triage-P-Low": d.p_low, "X-Triage-P-Medium": d.p_medium, "X-Triage-Tau": d.tau}
    return {name: _six(v) for name, v in values.items() if v is not None}


def _apply_verification(row: RequestRow, first, outcome, deps, providers=None, prefer=None) -> None:
    """Costs and log fields after sync verification: the served answer's tokens, every call's cost."""
    final = outcome.final.response
    _apply_costs(row, final, deps, providers, prefer)  # tokens, cache and strong_cost from the served answer
    row.price_as_cost = first.reported_cost + outcome.check_cost + outcome.escalation_cost
    row.cost = first.cost + outcome.check_actual_cost + outcome.escalation_actual_cost
    _credit_saving(row)
    row.escalated, row.decided_by = outcome.escalated, outcome.decided_by
    row.checker_verdict, row.checker_version = outcome.checker_verdict, outcome.checker_version
    row.escalated_from, row.escalated_to = outcome.escalated_from, outcome.escalated_to
    row.escalation_extra_cost = outcome.escalation_cost
    row.escalation_extra_latency_ms = round(outcome.escalation_latency_ms, 1)
    row.check_cost, row.check_latency_ms = outcome.check_cost, round(outcome.check_latency_ms, 1)
    if outcome.escalation_error:
        row.error_kind = outcome.escalation_error      # served the last answer; the escalation found no model
    row.tier, row.served_model = outcome.final.tier, outcome.final.model


def _verified_reason(d: core.RouteDecision, outcome) -> str:
    if not outcome.steps:
        return d.reason
    if outcome.escalated:
        return f"{d.reason}; verified: {outcome.escalated_from}->{outcome.escalated_to} ({outcome.decided_by})"
    if outcome.decided_by.startswith("skipped_"):
        return f"{d.reason}; verified: kept {d.tier} (AI check {outcome.decided_by})"
    return f"{d.reason}; verified: kept {d.tier} ({outcome.decided_by}: {outcome.checker_verdict})"


def _cost_headers(row: RequestRow) -> dict:
    """X-Triage-Cost, -Strong-Cost and -Saved for a finished request; each is left out when unknown, never faked
    (strong cost: no baseline model reachable; saved: a named model, see _credit_saving)."""
    headers = {}
    if row.price_as_cost is not None:
        headers["X-Triage-Cost"] = _money(row.price_as_cost)
    if row.strong_cost is not None:
        headers["X-Triage-Strong-Cost"] = _money(row.strong_cost)
    if row.saved is not None:
        headers["X-Triage-Saved"] = _money(row.saved)
    return headers


def triage_body(headers: dict) -> dict:
    """The top-level "triage" object of a non-streaming response: the same values as its X-Triage-* headers (built
    from them, so the two can't disagree), numbers as JSON numbers. A header that was left out is null.
    OpenAI clients ignore the unknown field."""
    def number(name):
        value = headers.get(name)
        return None if value is None else float(value)

    preferred = headers.get("X-Triage-Preferred-Provider")

    return {"request_id": headers["X-Triage-Request-Id"], "model": headers["X-Triage-Model"],
            "tier": headers["X-Triage-Tier"], "p_low": number("X-Triage-P-Low"),
            "p_medium": number("X-Triage-P-Medium"), "tau": number("X-Triage-Tau"),
            "reason": headers["X-Triage-Reason"], "cost": number("X-Triage-Cost"),
            "strong_cost": number("X-Triage-Strong-Cost"), "saved": number("X-Triage-Saved"),
            "cache": headers["X-Triage-Cache"], "fallback": headers["X-Triage-Fallback"],
            "mode": headers.get("X-Triage-Mode"), "routed_by": headers.get("X-Triage-Routed-By"),
            "preferred_provider": None if preferred in (None, "none") else preferred}


def _usage(response) -> dict:
    return {"prompt_tokens": response.input_tokens, "completion_tokens": response.output_tokens,
            "total_tokens": response.input_tokens + response.output_tokens}


def _chunk(request_id: str, created: int, model: str, delta: dict, finish=None, usage=None, choices=True,
           triage=None) -> str:
    """One SSE chunk. triage: the same object as a non-streaming response's "triage", on chunks that keep their
    choices[0], so clients that read chunk.choices[0] never meet an empty list (the SDKs keep unknown fields)."""
    body = {"id": request_id, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}] if choices else []}
    if usage is not None:
        body["usage"] = usage
    if triage is not None:
        body["triage"] = triage
    return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"


def keyless_lookup_minutes_from_env(environ=os.environ) -> float:
    """TRIAGE_KEYLESS_LOOKUP_MINUTES: how long a keyless request's cost can be read back without a key (default 10).
    A number above 0; anything else refuses to start."""
    raw = (environ.get(KEYLESS_LOOKUP_ENV) or "").strip()
    if not raw:
        return DEFAULT_KEYLESS_LOOKUP_MINUTES
    try:
        value = float(raw)
    except ValueError:
        value = 0
    if not value > 0:
        raise RuntimeError(f"refusing to start: {KEYLESS_LOOKUP_ENV} must be a number of minutes above 0")
    return value


def _not_allowed_message(error: ModelNotAllowed) -> str:
    header = PROVIDER_KEY_HEADERS.get(error.provider)
    if header is None:                 # e.g. the local Ollama with a BYOK request: only the user's providers
        return (f"The model `{error.model_name}` runs on {error.provider}, which a request with your own keys "
                "doesn't use. Pick a model on a provider you sent a key for, or use \"auto\".")
    return (f"The model `{error.model_name}` runs on {error.provider} and needs your own {error.provider} key: "
            f"send it in the {header} header.")


NO_USABLE_MODEL_MESSAGE = ("None of the provider keys sent can reach a model. Send a key in one of: "
                           + ", ".join(PROVIDER_KEY_HEADERS.values()) + ".")

PROVIDER_NAMES = {"openai": "OpenAI", "anthropic": "Anthropic", "groq": "Groq", "gemini": "Gemini"}


def rejected_key_providers(attempts) -> list:
    """BYOK: the providers that rejected the user's key, if EVERY failed attempt was an auth error (401/403);
    otherwise [] (mixed failures keep the usual 503). In the order they were tried."""
    failed = [a for a in attempts if a.outcome == "error"]
    if not failed or any(a.error_kind != "auth" for a in failed):
        return []
    return list(dict.fromkeys(a.provider for a in failed))


def providers_if_every_failure_was(attempts, *kinds: str) -> list:
    """The providers tried, if EVERY failed attempt had one of these error kinds (e.g. "timeout"); otherwise []
    (other failures keep the usual 503). In the order they were tried."""
    failed = [a for a in attempts if a.outcome == "error"]
    if not failed or any(a.error_kind not in kinds for a in failed):
        return []
    return list(dict.fromkeys(a.provider for a in failed))


def unavailable_message(providers: list, own_keys: bool) -> str:
    names = _join([PROVIDER_NAMES.get(p, p) for p in providers])
    if own_keys:
        return (f"{names} couldn't answer right now (busy or too slow). Try again in a moment, or add another "
                "provider key so Triage can fall back.")
    return f"{names} couldn't answer right now (busy or too slow). Try again in a moment."


def timed_out_providers(attempts) -> list:
    return providers_if_every_failure_was(attempts, "timeout")


def overloaded_message(providers: list, own_keys: bool) -> str:
    names = _join([PROVIDER_NAMES.get(p, p) for p in providers])
    verb = "are" if len(providers) > 1 else "is"
    if own_keys:
        return (f"{names} {verb} very busy right now. Try again in a moment, or add another provider key so Triage "
                "can fall back.")
    return f"{names} {verb} very busy right now. Try again in a moment."


def timeout_message(providers: list, own_keys: bool) -> str:
    names = _join([PROVIDER_NAMES.get(p, p) for p in providers])
    if own_keys:       # BYOK or an account: another key of theirs gives Triage something to fall back to
        return f"{names} took too long to answer. Try again, or add another provider key so Triage can fall back."
    return f"{names} took too long to answer. Try again."


def _join(items: list) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def auth_rejections(attempts) -> list:
    """(provider, redacted message) of every attempt a provider refused as unauthorized (401/403)."""
    return [(a.provider, a.message) for a in attempts if a.outcome == "error" and a.error_kind == "auth"]


def invalid_key_message(providers: list, saved=frozenset()) -> str:
    """Names the provider(s) and header(s), or says a saved account key, never a key or a tier."""
    from_saved = [p for p in providers if p in saved]
    sent = [p for p in providers if p not in saved]
    parts = []
    if from_saved:
        names = _join([PROVIDER_NAMES.get(p, p) for p in from_saved])
        plural = len(from_saved) > 1
        parts.append(f"{names} rejected your saved key{'s' if plural else ''}. Update "
                     f"{'them' if plural else 'it'} in your account settings (PUT /v1/account/provider-keys/"
                     f"{from_saved[0] if not plural else '{provider}'}).")
    if sent:
        parts.append(_sent_key_message(sent))
    return " ".join(parts)


def _sent_key_message(providers: list) -> str:
    names = _join([PROVIDER_NAMES.get(p, p) for p in providers])
    headers = _join([PROVIDER_KEY_HEADERS.get(p, p) for p in providers])
    plural = len(providers) > 1
    return (f"{names} rejected the API key{'s' if plural else ''} you sent in {headers}. "
            f"Check the key{'s' if plural else ''} and try again.")


def _error_response(error: APIError, headers: Optional[dict] = None) -> JSONResponse:
    return JSONResponse(error.body(), status_code=error.status, headers={**error.headers, **(headers or {})})


INTERNAL_ERROR = APIError(500, "Triage hit an internal error. Try again; if it keeps happening, please report it.",
                          "server_error", code="internal_error")


class InternalErrorMiddleware:
    """Unhandled exceptions as an OpenAI-style JSON 500. Added inside CORSMiddleware, so a browser can read the
    error: Starlette's own 500 is sent from outside every middleware, without CORS headers, and a browser reports
    it as a network error. Carries the headers this request already decided on (X-Triage-Mode, -Request-Id).
    Never puts the exception's text in the response; the log gets its type and a redacted, shortened message.
    A response that has already started (a stream) can't be replaced: the error is re-raised as before."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        started = False

        async def tracking_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, receive, tracking_send)
        except Exception as error:
            if started:
                raise
            logger.error("unhandled %s on %s %s: %s\n%s", type(error).__name__, scope.get("method"),
                         scope.get("path"), redact(str(error))[:MAX_ATTEMPT_MESSAGE_CHARS],
                         "".join(traceback.format_tb(error.__traceback__)))
            headers = (scope.get("state") or {}).get("triage_headers")
            await _error_response(INTERNAL_ERROR, headers)(scope, receive, send)


CORS_ORIGINS_ENV = "TRIAGE_CORS_ORIGINS"      # comma-separated, e.g. https://triage.example.com
# Request headers a browser page may send, and response headers its JavaScript may read
# Every method an endpoint uses (tests/test_cors.py checks each route's preflight against this list)
CORS_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
CORS_REQUEST_HEADERS = ["Authorization", "Content-Type", *PROVIDER_KEY_HEADERS.values(), "X-Triage-Quality",
                        "X-Triage-Task", "X-Triage-Verify", "X-Triage-Prefer-Provider"]
CORS_RESPONSE_HEADERS = ["X-Triage-Request-Id", "X-Triage-Model", "X-Triage-Tier", "X-Triage-Reason",
                         "X-Triage-Fallback", "X-Triage-Cache", "X-Triage-Verified", "X-Triage-Escalated",
                         "X-Triage-Check", "X-Triage-Routed-By", "X-Triage-Preferred-Provider", "X-Triage-P-Low", "X-Triage-P-Medium", "X-Triage-Tau",
                         "X-Triage-Cost", "X-Triage-Strong-Cost", "X-Triage-Saved", "X-Triage-Mode",
                         "X-Triage-Demo-Remaining", "X-Triage-Demo-Limit", "X-Triage-Demo-Resets",
                         "X-Triage-Max-Tokens", "Retry-After"]


def set_up_keyring(deps) -> None:
    """Accounts need TRIAGE_ENCRYPTION_KEYS to encrypt saved provider keys: load and check it (raising
    EncryptionConfigError, which never prints a key) unless deps already has a keyring."""
    if deps.keyring is None:
        from app.accounts.encryption import KeyRing
        deps.keyring = KeyRing.from_env()
    deps.provider_keys.keyring = deps.keyring


def set_up_email(deps) -> None:
    """Password reset emails: choose the sender (app/accounts/email.py) unless deps already has one. Refuses to
    start if TRIAGE_DEV_EMAIL_LOG=on outside local development; without Resend settings, warns (names only) and
    the reset endpoints answer 503. A TRIAGE_EMAIL_LOGO_URL that doesn't serve an image, or a missing TRIAGE_APP_URL: one warning each."""
    if deps.email_sender is None:
        from app.accounts.email import sender_from_env
        from app.api.deps import FAKE_MODELS_ENV
        deps.email_sender = sender_from_env(fake=os.environ.get(FAKE_MODELS_ENV) == "1")
    if hasattr(deps.email_sender, "check_logo"):
        deps.email_sender.check_logo()            # one HEAD request (5 s); a failure only drops the logo
    if not deps.email_sender.available:
        logger.warning("password reset is unavailable: %s", deps.email_sender.reason)
        return
    for problem in getattr(deps.email_sender, "problems", ()):
        logger.warning("%s", problem)
    if deps.email_sender.name == "console":
        logger.warning("TRIAGE_DEV_EMAIL_LOG is on: password reset codes are printed to this console")


def cors_origins_from_env() -> list:
    return [o.strip().rstrip("/") for o in os.environ.get(CORS_ORIGINS_ENV, "").split(",") if o.strip()]


def request_mode(who: Optional[Caller], keys) -> str:
    """account: a user's trk_ key or session (their saved keys, plus any X-*-Key headers). api: a valid Triage
    key and no provider keys. byok: provider keys (with or without a Triage key). demo: neither (only reachable
    when public access is on)."""
    if who is not None and who.kind == "account":
        return "account"
    if keys.mode == "byok":
        return "byok"
    return "api" if who is not None else "demo"


# key_id in the request log for requests without a Triage key (no visitor identity goes into the log)
PUBLIC_KEY_IDS = {"demo": "demo", "byok": "public"}
# What GET /v1/requests/{id} returns without a key, for a request made without one: its outcome and cost, nothing
# about the prompt, the routing probabilities or the attempts. Only while the request is recent: the web chat asks
# within seconds of a Stop; after that window, the id is treated like an unknown one.
KEYLESS_LOOKUP_ENV = "TRIAGE_KEYLESS_LOOKUP_MINUTES"
DEFAULT_KEYLESS_LOOKUP_MINUTES = 10
REQUEST_COST_FIELDS = ("request_id", "status", "error_kind", "stream", "served_model", "tier", "cache_hit",
                       "tokens_estimated", "input_tokens", "output_tokens", "price_as_cost", "strong_cost", "saved")
_KEY_HEADER_LIST = ", ".join(PROVIDER_KEY_HEADERS.values())


def demo_limit_error(error: DemoLimitError, settings) -> APIError:
    """A refused demo / public BYOK request as an OpenAI-style 429 with Retry-After."""
    messages = {
        "demo_limit_reached": (f"You've used today's {settings.visitor_daily_requests} free demo requests. Add your "
                               f"own provider key ({_KEY_HEADER_LIST}) to keep going, or try again after 00:00 UTC."),
        "demo_paused": ("The free demo has reached today's limit and is paused until 00:00 UTC. You can still use "
                        f"Triage with your own provider key ({_KEY_HEADER_LIST})."),
        "demo_busy": ("The free demo is busy right now. Try again in a few seconds, or add your own provider key "
                      f"({_KEY_HEADER_LIST})."),
        "public_limit_reached": (f"This address has used today's {settings.public_byok_visitor_daily_requests} "
                                 "requests without a Triage API key. Send a Triage API key (Authorization: Bearer "
                                 "<key>) or try again after 00:00 UTC."),
    }
    return APIError(429, messages[error.code], "rate_limit_error", code=error.code,
                    headers={"Retry-After": str(error.retry_after)})


def demo_headers(remaining: int, limit: int, resets_at) -> dict:
    return {"X-Triage-Demo-Remaining": str(remaining), "X-Triage-Demo-Limit": str(limit),
            "X-Triage-Demo-Resets": resets_at.strftime("%Y-%m-%dT%H:%M:%SZ")}


def demo_usage(row: RequestRow, reserved: int) -> tuple:
    """(tokens, model calls) a finished demo request used, for DemoLimiter.settle.
    Calls: every attempt that reached a provider (a cache hit or a circuit-breaker skip didn't).
    Tokens: the served answer (none for a cache hit) plus billed truncated attempts. If an answer was started but
    its token count is unknown (a stream cut off part-way), the reservation is kept as the best estimate."""
    attempts = row.attempts or []
    calls = max(0, sum(a.get("outcome") in ("success", "error") for a in attempts) - (1 if row.cache_hit else 0))
    if row.served_model is not None and row.input_tokens is None:
        return reserved, calls
    truncated = sum((a.get("input_tokens") or 0) + (a.get("output_tokens") or 0) for a in attempts)
    served = 0 if row.cache_hit else (row.input_tokens or 0) + (row.output_tokens or 0)
    return served + truncated, calls


# ---------- the app ----------

SDK_IMPORT_DELAY = 0.5      # seconds after startup: uvicorn binds the port first


def create_app(deps=None, auth: Optional[Auth] = None, public: Optional[bool] = None,
               cors_origins: Optional[list] = None, accounts: Optional[bool] = None) -> FastAPI:
    """deps / auth given: use them (tests). Otherwise load both at startup from config and the environment,
    refusing to start if there are no keys or the predictor fails its safety checks.
    public: accept chat requests without a Triage key (demo / BYOK); None = TRIAGE_PUBLIC (default off).
    cors_origins: browser origins allowed to call the API; None = TRIAGE_CORS_ORIGINS. Empty (the default): no
    cross-origin access at all.
    accounts: the /v1/auth and /v1/account endpoints (app/api/account_routes.py); None = TRIAGE_ACCOUNTS (default
    off: those paths don't exist)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        retention_days = maintenance.retention_days_from_env()      # refuses a bad value before loading anything
        if app.state.auth is None:
            app.state.auth = Auth(keys_from_env(API_KEYS_ENV), keys_from_env(ADMIN_KEYS_ENV))
        if app.state.public:
            logger.warning("public access is ON (TRIAGE_PUBLIC): chat requests without a Triage key are accepted")
        if app.state.accounts:
            logger.warning("accounts are ON (TRIAGE_ACCOUNTS): anyone who can reach the server can sign up")
        if app.state.deps is None:
            from app.api.deps import load_deps
            app.state.deps = load_deps()          # predictor + embedding model, once
        if app.state.accounts:
            set_up_keyring(app.state.deps)        # refuses to start without valid TRIAGE_ENCRYPTION_KEYS
            set_up_email(app.state.deps)          # refuses TRIAGE_DEV_EMAIL_LOG=on outside local development
        # Housekeeping (privacy policy retention): now, then cache hourly, database daily; stopped at shutdown
        housekeeping = asyncio.create_task(maintenance.run_forever(app.state.deps, retention_days))
        # The OpenAI and Gemini SDKs (the server's own providers) load in a background thread once uvicorn is
        # listening, so /healthz answers sooner; a request that needs one first waits for it (app/providers/lazy.py)
        asyncio.get_running_loop().call_later(SDK_IMPORT_DELAY, lazy_providers.warm_in_background)
        try:
            yield
        finally:
            housekeeping.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await housekeeping

    _ensure_json_handler()
    app = FastAPI(title="Triage", version="0.5", lifespan=lifespan)
    app.state.deps, app.state.auth = deps, auth
    app.state.db_health = None                       # /readyz's database check (built on first use)
    app.state.public = public_from_env() if public is None else public
    app.state.accounts = accounts_from_env() if accounts is None else accounts
    app.state.public_stats = public_stats.PublicStatsCache()
    app.state.public_stats_threshold = public_stats.threshold_from_env()   # refuses a bad value
    app.state.keyless_lookup_minutes = keyless_lookup_minutes_from_env()   # refuses a bad value
    origins = cors_origins_from_env() if cors_origins is None else list(cors_origins)
    app.add_middleware(InternalErrorMiddleware)       # first, so it sits inside CORSMiddleware
    if origins:
        # No cookies or credentials: the browser sends keys only as the explicit headers listed
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=CORS_METHODS,
                           allow_headers=CORS_REQUEST_HEADERS, expose_headers=CORS_RESPONSE_HEADERS,
                           allow_credentials=False, max_age=600)

    @app.exception_handler(APIError)
    async def api_error(request: Request, error: APIError):
        # headers this request already decided on (e.g. X-Triage-Mode), so errors carry them too
        return _error_response(error, getattr(request.state, "triage_headers", None))

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError):
        first = error.errors()[0] if error.errors() else {}
        loc = [str(x) for x in first.get("loc", ()) if x != "body"]
        param = ".".join(loc) or None
        return _error_response(bad_request(f"Invalid request: {param or 'body'}: {first.get('msg', 'invalid')}",
                                           param, "invalid_request"))

    async def account_caller(authorization: Optional[str]) -> Optional[Caller]:
        """Accounts on and a "trk_" or "trs_" bearer token: the account behind it (a worker thread: database).
        trk_: a user's API key (key_id = its public id; its last use is recorded, at most once a minute).
        trs_: a session token, i.e. website chat (key_id = "session").
        Revoked, unknown, expired or malformed: the same 401 as any bad key. Anything else: None, and the env
        keys (TRIAGE_API_KEYS / TRIAGE_ADMIN_KEYS) decide, unchanged."""
        if not app.state.accounts or not authorization or not authorization.lower().startswith("bearer "):
            return None
        token = authorization[7:].strip()
        deps = app.state.deps
        if token.startswith(API_KEY_PREFIX):
            def check():
                found = deps.api_keys.lookup(token)
                if found is None:
                    return None
                deps.api_keys.touch(found)
                user = deps.accounts.user_by_id(found.user_id)
                return None if user is None else (found.public_id, user)
        elif token.startswith(SESSION_PREFIX):
            def check():
                found = deps.accounts.session_for_token(token)
                return None if found is None else ("session", found[1])
        else:
            return None
        found = await anyio.to_thread.run_sync(check)
        if found is None:
            raise APIError(401, "Incorrect API key provided.", code="invalid_api_key")
        key_id, user = found
        return Caller(key_id, False, "account", user.id, user.preferred_provider)

    async def caller(authorization: Optional[str] = Header(None)) -> Caller:
        return await account_caller(authorization) or app.state.auth.check(authorization)

    async def admin(authorization: Optional[str] = Header(None)) -> Caller:
        if await account_caller(authorization) is not None:        # accounts are never admins
            raise APIError(403, "This endpoint needs an admin key.", "permission_error", code="admin_key_required")
        return app.state.auth.check(authorization, admin=True)

    async def chat_caller(authorization: Optional[str] = Header(None)) -> Optional[Caller]:
        """Public access on: no Authorization header = no caller (demo or BYOK); a header that is sent must be
        valid. Off: exactly `caller`."""
        account = await account_caller(authorization)
        if account is not None:                  # a trk_ key or a session: account mode
            return account
        if not app.state.public:
            return app.state.auth.check(authorization)
        return app.state.auth.optional(authorization)

    # ----- chat -----

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatRequest, request: Request, who: Optional[Caller] = Depends(chat_caller),
                               x_triage_quality: Optional[str] = Header(None),
                               x_triage_task: Optional[str] = Header(None),
                               x_triage_verify: Optional[str] = Header(None),
                               x_triage_prefer_provider: Optional[str] = Header(None)):
        deps = app.state.deps
        started = time.perf_counter()
        # This request's provider keys (X-OpenAI-Key, ...). Passed down the call path only: never stored on deps,
        # app state, the cache or a background task. Closures below use the plain `byok` flag, not `keys`.
        keys = resolve_keys(request.headers)
        account = who is not None and who.kind == "account"
        saved_providers = frozenset()            # account: providers whose SAVED key this request uses
        if account:
            # The account's saved keys (valid or unverified), decrypted for this request only; X-*-Key headers
            # replace the saved key for their provider. The plain keys live only in this request's KeyContext.
            header_keys = dict(keys.keys)
            saved = await anyio.to_thread.run_sync(deps.provider_keys.usable_keys, who.user_id) \
                if deps.provider_keys.keyring is not None else {}
            saved_providers = frozenset(saved) - frozenset(header_keys)
            merged = {**saved, **header_keys}
            keys = KeyContext("byok", merged) if merged else KeyContext("server")    # none: 400 below
            saved = header_keys = merged = None
        byok = keys.mode == "byok"
        mode = request_mode(who, keys)
        key_id = who.key_id if who is not None else PUBLIC_KEY_IDS[mode]
        request.state.triage_headers = {"X-Triage-Mode": mode}
        if account and not byok:
            raise APIError(400, "Your account has no usable provider keys. Add one in your account settings "
                           "(PUT /v1/account/provider-keys/{provider}), or send one with this request "
                           f"({_KEY_HEADER_LIST}).", code="no_provider_keys")
        baseline_providers = keys.providers if byok else None     # provider names only, for the savings baseline
        unknown = sorted(set(body.model_extra or {}) - ACCEPTED_FIELDS)
        if unknown:
            raise bad_request(f"Triage doesn't support these parameters yet: {', '.join(unknown)}", unknown[0],
                              "unsupported_parameter")
        if body.n not in (None, 1):
            raise bad_request("Triage returns one choice: n must be 1", "n", "unsupported_parameter")
        snapshot = deps.config.snapshot()
        tau = parse_quality(x_triage_quality, snapshot.quality)
        task = parse_task(x_triage_task)
        verify = parse_verify(x_triage_verify, body.stream)
        check_verify_model(verify, body.model)
        auto = body.model == "auto"
        # Checked in every mode, applied only to BYOK "auto" requests with a key for that provider (else ignored,
        # so api and demo requests route exactly as before)
        prefer = parse_prefer_provider(x_triage_prefer_provider)
        if prefer is None and account:           # the account's saved preference is the default
            prefer = who.preferred_provider
        if not (byok and auto and prefer in keys.providers):
            prefer = None
        requested_max_tokens = body.max_tokens or body.max_completion_tokens

        # Requests without a Triage key: a visitor (client IP, only ever as a keyed hash) with daily limits
        public = who is None
        visitor = visitor_id(client_ip(request, deps.trusted_proxy_hops), deps.hash_secret) if public else None
        demo = deps.demo_settings
        if mode == "demo":
            # The limiter's calls are database round trips: worker threads, like the request log
            status = await anyio.to_thread.run_sync(deps.demo_limiter.status, visitor)
            request.state.triage_headers.update(demo_headers(status.remaining, status.limit, status.resets_at))
            if verify == "sync":         # checks and escalations would multiply the free calls
                raise APIError(400, "X-Triage-Verify: sync isn't available in the free demo. Add your own provider "
                               f"key ({_KEY_HEADER_LIST}) to use it.", param="X-Triage-Verify",
                               code="demo_unsupported")
            prompt_chars = sum(len(m.content) for m in body.messages)
            if prompt_chars > demo.max_prompt_chars:
                raise APIError(413, f"Demo prompts are limited to {demo.max_prompt_chars:,} characters across all "
                               f"messages (yours: {prompt_chars:,}). Add your own provider key ({_KEY_HEADER_LIST}) "
                               "to send longer prompts.", param="messages", code="demo_prompt_too_long")
            default_max = snapshot.routing.generation.get("max_tokens", DEFAULT_MAX_TOKENS)
            requested_max_tokens = min(requested_max_tokens or default_max, demo.max_tokens)   # quietly lowered
            request.state.triage_headers["X-Triage-Max-Tokens"] = str(requested_max_tokens)

        messages = [m.model_dump() for m in body.messages]
        settings = core.GenerationSettings(body.temperature, requested_max_tokens)
        temperature = settings.temperature if settings.temperature is not None else \
            snapshot.routing.generation.get("temperature", DEFAULT_TEMPERATURE)
        max_tokens = settings.max_tokens or snapshot.routing.generation.get("max_tokens", DEFAULT_MAX_TOKENS)
        training = snapshot.routing.generation
        request_id = f"chatcmpl-triage-{uuid.uuid4().hex}"
        # From here on every outcome is logged under this id, so errors carry it too (GET /v1/requests/{id})
        request.state.triage_headers["X-Triage-Request-Id"] = request_id
        row = RequestRow(request_id=request_id, timestamp=_now(), key_id=key_id, requested_model=body.model,
                         stream=body.stream, quality=x_triage_quality, tau=tau if auto else None,
                         task_type=task, classifier=deps.classifier if auto else None,
                         config_version=snapshot.version, temperature=temperature, max_tokens=max_tokens,
                         settings_match_training=(temperature == training.get("temperature")
                                                  and max_tokens == training.get("max_tokens")) if auto else None,
                         prompt_hmac=prompt_hmac(messages, deps.hash_secret), verified=verify,
                         mode=mode, user_id=who.user_id if account else None)
        if not auto and body.model in deps.registry:       # a named model is its own primary, even if it fails
            row.primary_model, row.tier = body.model, deps.registry[body.model].tier
        # What decides the tier, known before routing (so a request whose every model fails says it too)
        small_talk = auto and is_small_talk(messages, snapshot.routing.small_talk_phrases)
        row.routed_by = "small_talk" if small_talk else ("classifier" if auto else "requested_model")
        if small_talk:                                      # no classifier, no quality dial
            row.tau = row.classifier = None

        logged, admitted = [], []          # admitted: this request's DemoLimiter Admission, once counted
        invalidate = []                    # account: (provider, redacted detail) of saved keys a provider rejected
        user_id = who.user_id if account else None

        def note_rejected_saved_keys(attempts):
            """Any 401/403 on a SAVED key marks it invalid (even if a fallback then answered), so later requests
            skip it. Header keys are the caller's own business and are never touched."""
            if not account:
                return
            for provider, message in auth_rejections(attempts):
                if provider in saved_providers and provider not in {p for p, _ in invalidate}:
                    detail = f"{PROVIDER_NAMES.get(provider, provider)} rejected it during a chat request"
                    if message:
                        detail += f" ({redact(message, keys.secrets)})"
                    invalidate.append((provider, detail[:160]))
        recorder = _RecordingPredictor(deps.predictor)
        request_deps = dataclasses.replace(deps, predictor=recorder)    # same client, registry and log

        def persist():
            deps.log.write(row)
            request_logger.info(json_log_line(row))
            for provider, detail in invalidate:
                deps.provider_keys.set_status(user_id, provider, "invalid", detail, verified=True)
            if admitted:                   # replace the token reservation with what it really used
                tokens, calls = demo_usage(row, admitted[0].reserved_tokens)
                deps.demo_limiter.settle(admitted[0], tokens=tokens, calls=calls)

        async def finish(status="ok", http_status=200, error_kind=None):
            if logged:
                return
            logged.append(True)
            row.status, row.http_status = status, http_status
            row.error_kind = error_kind if error_kind is not None else row.error_kind   # keep a verification note
            row.latency_ms = round((time.perf_counter() - started) * 1000, 1)
            _apply_inputs(row, recorder.prediction)
            row.training_eligible = is_training_eligible(row)
            # A worker thread: a database round trip (Postgres over the network) must not stall the event loop
            await anyio.to_thread.run_sync(persist)

        if not auto and body.model not in deps.registry:              # logged, then an OpenAI-style 404
            await finish("error", 404, "model_not_found")
            raise APIError(404, str(core.UnknownModel(body.model)), param="model", code="model_not_found")

        if mode == "demo":                 # counted only now, after every check that costs nothing
            estimate = -(-sum(len(m.content) for m in body.messages) // 4) + requested_max_tokens
            try:
                admitted.append(await anyio.to_thread.run_sync(deps.demo_limiter.admit_demo, visitor, estimate))
            except DemoLimitError as error:
                status = await anyio.to_thread.run_sync(deps.demo_limiter.status, visitor)
                request.state.triage_headers.update(demo_headers(status.remaining, status.limit, status.resets_at))
                await finish("error", 429, error.code)
                raise demo_limit_error(error, demo) from None
            request.state.triage_headers["X-Triage-Demo-Remaining"] = str(admitted[0].remaining)
        elif mode == "account":            # no demo or public-BYOK limits; per-account ones instead
            try:
                await anyio.to_thread.run_sync(deps.account_limiter.admit, user_id)
            except AccountLimitError as error:
                await finish("error", 429, "account_rate_limited")
                per = "minute" if error.which == "minute" else "day"
                raise APIError(429, f"Your account has reached its requests-per-{per} limit. Try again in "
                               f"{error.retry_after} seconds.", "rate_limit_error", code="account_rate_limited",
                               headers={"Retry-After": str(error.retry_after)}) from None
        elif mode == "byok" and public:   # BYOK without a Triage key: a generous per-visitor cap (the user pays)
            try:
                admitted.append(await anyio.to_thread.run_sync(deps.demo_limiter.admit_public_byok, visitor))
            except DemoLimitError as error:
                await finish("error", 429, error.code)
                raise demo_limit_error(error, demo) from None

        row.preferred_provider = prefer    # applied to the chains from here on, whatever the outcome
        try:
            routed = await core.route_and_stream(messages, tau=tau if auto else None, task_type=task,
                                                 model=body.model, stream=body.stream, settings=settings,
                                                 deps=request_deps, request_id=request_id, keys=keys,
                                                 prefer=prefer)
        except core.UnknownModel as error:
            await finish("error", 404, "model_not_found")
            raise APIError(404, str(error), param="model", code="model_not_found") from None
        except ModelNotAllowed as error:                   # e.g. a claude-* model, and this request has no key
            await finish("error", 403, "model_not_allowed")
            raise APIError(403, _not_allowed_message(error), "permission_error", param="model",
                           code="model_not_allowed") from None
        except core.NoUsableModel:
            await finish("error", 400, "no_usable_model")
            raise APIError(400, NO_USABLE_MODEL_MESSAGE, code="no_usable_model") from None
        except AllModelsFailed as error:
            # Every attempt is logged (model, outcome, error_kind, redacted message), even though none answered
            row.attempts = [_attempt_dict(a) for a in error.attempts]
            _add_truncated_costs(row, error.attempts)
            if auto:
                row.tier = error.tier
                row.primary_model = error.attempts[0].model_name if error.attempts else None
            last = error.attempts[-1] if error.attempts else None
            if not auto and last is not None and last.error_kind == "truncated":     # a named model: no chain
                await finish("error", 400, "truncated")
                used = last.response.output_tokens if last.response is not None else "all its"
                raise APIError(400, f"`{body.model}` used {used} output tokens thinking and wrote no answer. "
                               "Send a higher max_tokens: thinking counts toward it.", param="max_tokens",
                               code="max_tokens_too_low") from None
            note_rejected_saved_keys(error.attempts)
            rejected = rejected_key_providers(error.attempts) if byok else []
            if rejected:                                    # the user's own key(s) were refused: say so, as a 401
                await finish("error", 401, "invalid_provider_key")
                raise APIError(401, invalid_key_message(rejected, saved_providers), code="invalid_provider_key") \
                    from None
            timed_out = timed_out_providers(error.attempts)
            if timed_out:                                   # every model ran out of time: say who, as a 504
                await finish("error", 504, "provider_timeout")
                raise APIError(504, timeout_message(timed_out, byok), "server_error",
                               code="provider_timeout") from None
            overloaded = providers_if_every_failure_was(error.attempts, "overloaded")
            if overloaded:                                  # every model was too busy: say who
                await finish("error", 503, "provider_overloaded")
                raise APIError(503, overloaded_message(overloaded, byok), "server_error",
                               code="provider_overloaded") from None
            busy_or_slow = providers_if_every_failure_was(error.attempts, "overloaded", "timeout")
            if busy_or_slow:                                # a mix of the two: one message for both
                await finish("error", 503, "provider_unavailable")
                raise APIError(503, unavailable_message(busy_or_slow, byok), "server_error",
                               code="provider_unavailable") from None
            await finish("error", 503, "all_models_failed")
            raise APIError(503, f"No model could answer: {error}", "server_error", code="all_models_failed") from None
        decision = routed.decision
        _apply_decision(row, decision)
        note_rejected_saved_keys(decision.attempts)
        row.ttft_ms = round((time.perf_counter() - started) * 1000, 1)
        created = int(time.time())

        # Background labels: only for auto requests with verification off. The labeler decides (eligibility,
        # sampling, daily cap) after the response has been sent; it gets the text from memory only.
        def background(answer_text):
            # Only for "api" requests: BYOK (a user key must never reach a background job) and demo (strangers'
            # prompts, and labels would spend the free quota) get none
            if mode != "api" or not auto or verify != "off" or deps.labeler is None:
                return None
            return BackgroundTask(deps.labeler.run, deps, row, messages, answer_text)

        if not body.stream:
            try:
                text = "".join([piece async for piece in routed.pieces])
            except ProviderError as error:
                await finish("error", 502, error.kind)
                raise APIError(502, f"The model failed while answering: {error.kind}", "server_error",
                               code="model_error") from None
            response = routed.response
            served_model, served_tier, reason, check, escalated = decision.model, decision.tier, None, "none", False
            if verify == "sync":
                outcome = await verify_sync(messages, Served(decision.tier, decision.model, response, text),
                                            snapshot.verification.sync, deps, settings,
                                            probabilities={"low": decision.p_low, "medium": decision.p_medium},
                                            keys=keys, prefer=prefer)
                _apply_verification(row, response, outcome, deps, baseline_providers, prefer)
                response, text = outcome.final.response, outcome.final.text
                served_model, served_tier, check, escalated = (outcome.final.model, outcome.final.tier,
                                                               outcome.decided_by, outcome.escalated)
                reason = _verified_reason(decision, outcome)
            else:
                _apply_costs(row, response, deps, baseline_providers, prefer)
            _add_truncated_costs(row, decision.attempts)
            await finish()
            cache = "hit" if response.cached else ("miss" if is_cacheable(temperature) else "bypass")
            headers = _triage_headers(decision, cache, model=served_model, tier=served_tier, reason=reason,
                                      verified=verify, escalated=escalated, check=check) | _cost_headers(row)
            headers |= request.state.triage_headers
            completion = {"id": request_id, "object": "chat.completion", "created": created, "model": served_model,
                          "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                       "finish_reason": response.finish_reason or "stop"}],
                          "usage": _usage(response), "triage": triage_body(headers)}
            return JSONResponse(completion, headers=headers, background=background(text))

        include_usage = bool(body.stream_options and body.stream_options.include_usage)
        streamed = []                                         # the answer, in memory only, for a background label
        # The routing is known now (the first token exists); cost only at the end. The first chunk carries the
        # "triage" object without cost, the finish_reason chunk the complete one, as a non-streaming response has
        cache_status = "hit" if routed.cached else ("miss" if is_cacheable(temperature) else "bypass")
        start_headers = _triage_headers(decision, cache_status) | request.state.triage_headers

        async def events():
            try:
                async for event in stream_events():
                    yield event
            finally:
                with anyio.CancelScope(shield=True):           # a disconnect cancels the stream; still log it
                    if not logged:
                        # The client went away (the web chat's Stop): end the provider's stream so the model
                        # stops generating, and record what the answer cost so far as an estimate
                        await routed.pieces.aclose()
                        _estimate_stopped_cost(row, decision, routed, "".join(streamed), messages, deps,
                                               baseline_providers, prefer, started)
                    # Not a failure: its own outcome, counted apart from errors in every stat
                    await finish("stopped", 499, "client_disconnected")    # no-op if the row was already written

        async def stream_events():
            yield _chunk(request_id, created, decision.model, {"role": "assistant", "content": ""},
                         triage=triage_body(start_headers))
            try:
                async for piece in routed.pieces:
                    streamed.append(piece)
                    yield _chunk(request_id, created, decision.model, {"content": piece})
            except Exception as error:                           # after the first token: no fallback
                kind = getattr(error, "kind", type(error).__name__)
                await finish("error", 200, f"stream_interrupted:{kind}")
                logger.warning("stream %s interrupted after the first token: %s", request_id, kind)
                yield stream_error_event(f"The model failed part-way through the answer ({kind}). "
                                         "Triage can't switch models once text has been sent.")
                return
            response = routed.response
            _apply_costs(row, response, deps, baseline_providers, prefer)
            _add_truncated_costs(row, decision.attempts)
            await finish()
            yield _chunk(request_id, created, decision.model, {}, finish=(response and response.finish_reason) or "stop",
                         triage=triage_body(start_headers | _cost_headers(row)))
            if include_usage and response is not None:
                yield _chunk(request_id, created, decision.model, None, usage=_usage(response), choices=False)
            yield "data: [DONE]\n\n"

        async def stream_background():                       # runs after the last byte has been sent
            if background(None) is not None:
                await deps.labeler.run(deps, row, messages, "".join(streamed) if row.status == "ok" else None)

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers=start_headers | {"Cache-Control": "no-cache"},
                                 background=BackgroundTask(stream_background))

    if app.state.accounts:
        from app.api.account_routes import account_router
        from app.api.usage_routes import usage_router
        app.include_router(account_router())
        app.include_router(usage_router(account_caller))

    # ----- health -----

    @app.get("/healthz", include_in_schema=False)
    async def healthz():
        """Liveness, for Docker's healthcheck and uptime monitors: 200 once the classifier and the embedding model
        are loaded, else 503. Never touches the database (so it can't keep Neon awake). No key needed."""
        code, body = health.liveness_report(app.state.deps)
        return JSONResponse(body, status_code=code, headers={"Cache-Control": "no-store"})

    @app.get("/readyz", include_in_schema=False)
    async def readyz():
        """Readiness, for the deploy smoke test and manual checks: /healthz plus `SELECT 1` on the database within
        health.DB_TIMEOUT seconds. Can wake a sleeping database: don't point monitors here. No key needed."""
        deps = app.state.deps
        if getattr(app.state, "db_health", None) is None and deps is not None and deps.engine is not None:
            app.state.db_health = health.DbHealthCache(health.select_one_probe(deps.engine))
        code, body = await anyio.to_thread.run_sync(health.readiness_report, deps,
                                                    getattr(app.state, "db_health", None))
        return JSONResponse(body, status_code=code, headers={"Cache-Control": "no-store"})

    # ----- the free demo -----

    @app.get("/v1/demo/status")
    async def demo_status(request: Request):
        """Public (no key): this visitor's demo allowance and the demo-wide state, so a website can show its banner
        and counters before the first request. Reading it counts nothing."""
        if not app.state.public:
            return {"enabled": False}
        deps = app.state.deps
        visitor = visitor_id(client_ip(request, deps.trusted_proxy_hops), deps.hash_secret)
        status = await anyio.to_thread.run_sync(deps.demo_limiter.status, visitor)
        demo = deps.demo_settings
        return {"enabled": True, "paused": status.paused, "paused_reason": status.paused_reason,
                "visitor": {"limit": status.limit, "used": status.used, "remaining": status.remaining},
                "limits": {"max_prompt_chars": demo.max_prompt_chars, "max_tokens": demo.max_tokens},
                "resets_at": status.resets_at.strftime("%Y-%m-%dT%H:%M:%SZ")}

    # ----- models, stats, requests -----

    @app.get("/v1/models")
    async def list_models(who: Caller = Depends(caller)):
        deps = app.state.deps
        server_providers = deps.config.snapshot().routing.server_providers
        data = [{"id": "auto", "object": "model", "created": 0, "owned_by": "triage",
                 "triage": {"description": "Triage picks the cheapest model likely to pass (classifier "
                            f"{deps.classifier or 'configured in routing.yaml'})"}}]
        for name, m in deps.registry.items():          # retired models were already left out (serving_registry)
            # Only models a caller can reach: a provider that takes the caller's own key, or one this server runs
            # itself (the local Ollama only where server_providers has it, so never on a host without it)
            if m.provider not in PROVIDER_KEY_HEADERS and server_providers is not None \
                    and m.provider not in server_providers:
                continue
            price = m.price_as
            data.append({"id": name, "object": "model", "created": 0, "owned_by": m.provider,
                         "triage": {"tier": m.tier, "provider_model": m.model_id,
                                    "price_per_1m_input_usd": price.input_price if price else m.input_price,
                                    "price_per_1m_output_usd": price.output_price if price else m.output_price,
                                    "price_label": price.label if price else "actual price",
                                    # true: only usable with your own key for this provider (BYOK)
                                    "requires_key": server_providers is not None
                                                    and m.provider not in server_providers}})
        return {"object": "list", "data": data}

    @app.get("/v1/stats")
    async def stats(who: Caller = Depends(caller)):
        # read from the log (survives restarts), without the feature and embedding columns
        log, c = app.state.deps.log, usage_sql.c
        if who.kind == "account":              # every request of this account, whichever of its keys made it
            scope, where = "your account", c.user_id == who.user_id
        elif who.admin:
            scope, where = "all keys", sql_true()
        else:
            scope, where = "your key", c.key_id == who.key_id
        summary = await anyio.to_thread.run_sync(lambda: log.run(lambda connection: summarize(connection, where)))
        return {"scope": scope, **summary}

    @app.get("/v1/stats/public")
    async def stats_public(range: Optional[str] = None):
        """No auth: the Overview tab's live aggregates (app/api/public_stats.py), cached for 5 minutes."""
        deps = app.state.deps
        if deps.fake_mode:
            raise APIError(404, "Public stats aren't served in fake-model mode.", code="not_found")
        try:
            range_name = usage_sql.parse_range(range)
        except usage_sql.RangeError as error:
            raise bad_request(str(error), "range", "invalid_value") from None
        now = datetime.now(timezone.utc)
        result = await anyio.to_thread.run_sync(lambda: app.state.public_stats.get(
            range_name, lambda: public_stats.compute(deps.log, deps.registry, range_name, now,
                                                     app.state.public_stats_threshold)))
        return JSONResponse(result, headers={"Cache-Control": f"public, max-age={public_stats.TTL_SECONDS}"})

    @app.get("/v1/requests/{request_id}")
    async def get_request(request_id: str, authorization: Optional[str] = Header(None)):
        """A logged request, for the key (or account) that made it. Public access on, and no key sent: only a
        request made without a key (the demo or public BYOK), only while it's recent (TRIAGE_KEYLESS_LOOKUP_MINUTES,
        default 10), and only its outcome and cost (REQUEST_COST_FIELDS), so the web chat can show what a stopped
        answer cost. Request ids are random UUIDs: unguessable."""
        if authorization is None and app.state.public:
            row = await anyio.to_thread.run_sync(lambda: app.state.deps.log.get(request_id, include_vectors=False))
            oldest = (datetime.now(timezone.utc) - timedelta(minutes=app.state.keyless_lookup_minutes)).isoformat()
            if (row is None or row.user_id is not None or row.key_id not in PUBLIC_KEY_IDS.values()
                    or row.timestamp < oldest):                # too old: the same 404 as an unknown id
                raise APIError(404, f"No request {request_id!r} without a key.", param="request_id", code="not_found")
            return {name: getattr(row, name) for name in REQUEST_COST_FIELDS}
        who = await caller(authorization)
        row = await anyio.to_thread.run_sync(lambda: app.state.deps.log.get(request_id, include_vectors=False))
        owner = (row.user_id == who.user_id) if (row is not None and who.kind == "account") else \
            (row is not None and row.user_id is None and row.key_id == who.key_id)
        if not owner:                                       # someone else's request looks like a missing one
            raise APIError(404, f"No request {request_id!r} for this key.", param="request_id", code="not_found")
        return row.public_dict()

    # ----- proxy check (admin) -----

    @app.get("/v1/admin/forwarded-for", include_in_schema=False)
    async def forwarded_for(request: Request, who: Caller = Depends(admin)):
        """The X-Forwarded-For chain (and other client-address headers) of THIS request, as the API received it, to
        set TRIAGE_TRUSTED_PROXY_HOPS behind a host whose proxies aren't documented (Render): call it from your own
        machine, find your public IP in the chain, and count its position from the right. Admin key only, never
        logged; it shows the caller only their own request's addresses."""
        deps = app.state.deps
        chain = [part.strip() for part in ",".join(request.headers.getlist("x-forwarded-for")).split(",")
                 if part.strip()]
        others = {name: request.headers[name] for name in ("forwarded", "x-real-ip", "true-client-ip",
                                                             "cf-connecting-ip", "x-forwarded-proto")
                  if name in request.headers}
        hops = deps.trusted_proxy_hops
        return JSONResponse({
            "x_forwarded_for": chain,
            "direct_connection": request.client.host if request.client else None,
            "other_headers": others,
            "trusted_proxy_hops": hops,
            "client_ip_used": client_ip(request, hops),
            "how_to_read": "Find your own public IP in x_forwarded_for. Its position counted from the right "
                           "(1 = last) is the value for TRIAGE_TRUSTED_PROXY_HOPS.",
        }, headers={"Cache-Control": "no-store"})

    # ----- live routing config -----

    @app.get("/v1/routing-config")
    async def get_routing_config(who: Caller = Depends(admin)):
        snap = app.state.deps.config.snapshot()
        return {"version": snap.version, "config": snap.public(), "persisted": False, "note": NOT_PERSISTED_NOTE}

    @app.put("/v1/routing-config")
    async def put_routing_config(request: Request, who: Caller = Depends(admin)):
        try:
            patch = await request.json()
        except ValueError:
            raise bad_request("The body must be JSON.", None, "invalid_json") from None
        try:
            snap = app.state.deps.update_config(patch)
        except ConfigUpdateError as error:
            raise bad_request(f"Config not changed: {error}", None, "invalid_config") from None
        logger.info("routing config updated to version %d by key %s", snap.version, who.key_id)
        return {"version": snap.version, "config": snap.public(), "persisted": False, "note": NOT_PERSISTED_NOTE}

    return app


app = create_app()
