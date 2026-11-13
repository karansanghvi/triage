"""Triage's OpenAI-compatible API (Phase 4 Step 5).

Run it:
    TRIAGE_API_KEYS=dev-key TRIAGE_ADMIN_KEYS=admin-key TRIAGE_HASH_SECRET=<random> TRIAGE_FAKE_MODELS=1 \
        uvicorn app.api.main:app
(drop TRIAGE_FAKE_MODELS for real models; TRIAGE_CACHE=off disables the response cache. Fake mode uses its own
cache and log, data/cache/fake_responses.sqlite3 and data/logs/fake_requests.sqlite3, never the real ones)

Endpoints
    POST /v1/chat/completions     OpenAI chat completions, stream or not; model "auto" or a models.yaml name
    GET  /v1/models               "auto" plus every model, with prices
    GET  /v1/stats                savings summary (your key's requests; an admin key sees all)
    GET  /v1/requests/{id}        one logged request, only for the key that made it
    GET  /v1/routing-config       the config in use (admin)
    PUT  /v1/routing-config       update quality presets/default and backup models live (admin; not persisted)

Response headers: X-Triage-Request-Id, -Model, -Tier, -Reason, -Fallback, -Cache, -Verified, -Escalated,
-Check, and on non-streaming responses -Cost and -Saved (including any checks and escalation). A stream's headers are sent once its first token exists, so
model, tier and fallback are real; its cost isn't known yet, so -Cost and -Saved are left out (never faked).
The real figures are in GET /v1/requests/{id} once the stream ends, and in /v1/stats.
"""
import dataclasses
import hashlib
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.background import BackgroundTask

from app.api import core
from app.api.auth import ADMIN_KEYS_ENV, API_KEYS_ENV, Auth, Caller, keys_from_env, public_from_env
from app.api.config_state import NOT_PERSISTED_NOTE, ConfigUpdateError
from app.api.errors import APIError, bad_request, stream_error_event
from app.api.headers import (PROVIDER_KEY_HEADERS, check_verify_model, parse_quality, parse_task, parse_verify,
                             resolve_keys)
from app.api.request_log import RequestRow, embedding_to_bytes, is_training_eligible, prompt_hmac
from app.api.stats import summarize
from app.api.verification import Served, verify_sync
from app.cache import is_cacheable
from app.providers.base import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE
from app.providers.common import ProviderError
from app.providers.keys import redact
from app.reliability import AllModelsFailed, ModelNotAllowed

logger = logging.getLogger("triage.api")
# One JSON line per request (Step 10): ids, routing, status, latency and cost only. Never keys or text.
request_logger = logging.getLogger("triage.requests")
JSON_LOG_FIELDS = ("request_id", "timestamp", "key_id", "requested_model", "stream", "tier", "served_model",
                   "fell_back", "cache_hit", "status", "http_status", "error_kind", "latency_ms", "ttft_ms",
                   "price_as_cost", "saved", "verified", "escalated", "decided_by", "training_eligible")


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
    """Step 8 storage: only for model="auto" requests that answered, and only "api" requests (never BYOK or demo)."""
    if prediction is None or row.requested_model != "auto" or row.status != "ok" or row.mode in ("byok", "demo"):
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


def _apply_costs(row: RequestRow, response, deps, providers=None) -> None:
    """providers: a BYOK request's own; the savings baseline is then the strong model among them
    (deps.strong_model(providers)). None = the server's baseline, as before."""
    if response is None:
        return
    row.cache_hit = bool(response.cached)
    row.input_tokens, row.output_tokens = response.input_tokens, response.output_tokens
    row.reasoning_tokens, row.tokens_estimated = response.reasoning_tokens, response.tokens_estimated
    row.cost, row.price_as_cost = response.cost, response.reported_cost
    try:
        strong = deps.registry[deps.strong_model(providers)]
    except LookupError:                # no high-tier model on these providers: savings unknown, not faked
        row.strong_cost = row.saved = None
        return
    row.strong_cost = strong.reported_cost(response.input_tokens, response.output_tokens)
    row.saved = row.strong_cost - row.price_as_cost


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


def _add_truncated_costs(row: RequestRow, attempts) -> None:
    """An attempt that used every output token thinking and wrote no answer (then fell back) was still billed:
    add its cost to the request's total. Its tokens are in row.attempts; input/output_tokens stay the served
    answer's, and strong_cost stays the strong model at those tokens."""
    billed = [a.response for a in attempts if getattr(a, "response", None) is not None]
    if not billed:
        return
    row.cost = (row.cost or 0.0) + sum(r.cost for r in billed)
    row.price_as_cost = (row.price_as_cost or 0.0) + sum(r.reported_cost for r in billed)
    if row.strong_cost is not None:
        row.saved = row.strong_cost - row.price_as_cost


def _triage_headers(d: core.RouteDecision, cache: str, *, model: Optional[str] = None, tier: Optional[str] = None,
                    reason: Optional[str] = None, verified: str = "off", escalated: bool = False,
                    check: str = "none") -> dict:
    return {"X-Triage-Request-Id": d.request_id, "X-Triage-Model": model or d.model, "X-Triage-Tier": tier or d.tier,
            "X-Triage-Reason": _header_safe(reason or d.reason),
            "X-Triage-Fallback": f"{d.primary_model}->{d.model}" if d.fell_back else "none",
            "X-Triage-Cache": cache, "X-Triage-Verified": verified,
            "X-Triage-Escalated": "true" if escalated else "false", "X-Triage-Check": check}


def _apply_verification(row: RequestRow, first, outcome, deps, providers=None) -> None:
    """Costs and log fields after sync verification: the served answer's tokens, every call's cost."""
    final = outcome.final.response
    _apply_costs(row, final, deps, providers)          # tokens, cache and strong_cost from the served answer
    row.price_as_cost = first.reported_cost + outcome.check_cost + outcome.escalation_cost
    row.cost = first.cost + outcome.check_actual_cost + outcome.escalation_actual_cost
    row.saved = None if row.strong_cost is None else row.strong_cost - row.price_as_cost
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


def _usage(response) -> dict:
    return {"prompt_tokens": response.input_tokens, "completion_tokens": response.output_tokens,
            "total_tokens": response.input_tokens + response.output_tokens}


def _chunk(request_id: str, created: int, model: str, delta: dict, finish=None, usage=None, choices=True) -> str:
    body = {"id": request_id, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}] if choices else []}
    if usage is not None:
        body["usage"] = usage
    return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"


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


def _join(items: list) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def invalid_key_message(providers: list) -> str:
    """Names the provider(s) and header(s), never a key or a tier."""
    names = _join([PROVIDER_NAMES.get(p, p) for p in providers])
    headers = _join([PROVIDER_KEY_HEADERS.get(p, p) for p in providers])
    plural = len(providers) > 1
    return (f"{names} rejected the API key{'s' if plural else ''} you sent in {headers}. "
            f"Check the key{'s' if plural else ''} and try again.")


def _error_response(error: APIError, headers: Optional[dict] = None) -> JSONResponse:
    return JSONResponse(error.body(), status_code=error.status, headers={**error.headers, **(headers or {})})


def request_mode(who: Optional[Caller], keys) -> str:
    """api: a valid Triage key and no provider keys. byok: provider keys (with or without a Triage key).
    demo: neither (only reachable when public access is on)."""
    if keys.mode == "byok":
        return "byok"
    return "api" if who is not None else "demo"


# key_id in the request log for requests without a Triage key (no visitor identity goes into the log)
PUBLIC_KEY_IDS = {"demo": "demo", "byok": "public"}


# ---------- the app ----------

def create_app(deps=None, auth: Optional[Auth] = None, public: Optional[bool] = None) -> FastAPI:
    """deps / auth given: use them (tests). Otherwise load both at startup from config and the environment,
    refusing to start if there are no keys or the predictor fails its safety checks.
    public: accept chat requests without a Triage key (demo / BYOK); None = TRIAGE_PUBLIC (default off)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.auth is None:
            app.state.auth = Auth(keys_from_env(API_KEYS_ENV), keys_from_env(ADMIN_KEYS_ENV))
        if app.state.public:
            logger.warning("public access is ON (TRIAGE_PUBLIC): chat requests without a Triage key are accepted")
        if app.state.deps is None:
            from app.api.deps import load_deps
            app.state.deps = load_deps()          # predictor + embedding model, once
        yield

    _ensure_json_handler()
    app = FastAPI(title="Triage", version="0.5", lifespan=lifespan)
    app.state.deps, app.state.auth = deps, auth
    app.state.public = public_from_env() if public is None else public

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

    def caller(authorization: Optional[str] = Header(None)) -> Caller:
        return app.state.auth.check(authorization)

    def admin(authorization: Optional[str] = Header(None)) -> Caller:
        return app.state.auth.check(authorization, admin=True)

    def chat_caller(authorization: Optional[str] = Header(None)) -> Optional[Caller]:
        """Public access on: no Authorization header = no caller (demo or BYOK); a header that is sent must be
        valid. Off: exactly `caller`."""
        if not app.state.public:
            return app.state.auth.check(authorization)
        return app.state.auth.optional(authorization)

    # ----- chat -----

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatRequest, request: Request, who: Optional[Caller] = Depends(chat_caller),
                               x_triage_quality: Optional[str] = Header(None),
                               x_triage_task: Optional[str] = Header(None),
                               x_triage_verify: Optional[str] = Header(None)):
        deps = app.state.deps
        started = time.perf_counter()
        # This request's provider keys (X-OpenAI-Key, ...). Passed down the call path only: never stored on deps,
        # app state, the cache or a background task. Closures below use the plain `byok` flag, not `keys`.
        keys = resolve_keys(request.headers)
        byok = keys.mode == "byok"
        mode = request_mode(who, keys)
        key_id = who.key_id if who is not None else PUBLIC_KEY_IDS[mode]
        request.state.triage_headers = {"X-Triage-Mode": mode}
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

        messages = [m.model_dump() for m in body.messages]
        settings = core.GenerationSettings(body.temperature, body.max_tokens or body.max_completion_tokens)
        temperature = settings.temperature if settings.temperature is not None else \
            snapshot.routing.generation.get("temperature", DEFAULT_TEMPERATURE)
        max_tokens = settings.max_tokens or snapshot.routing.generation.get("max_tokens", DEFAULT_MAX_TOKENS)
        training = snapshot.routing.generation
        request_id = f"chatcmpl-triage-{uuid.uuid4().hex}"
        row = RequestRow(request_id=request_id, timestamp=_now(), key_id=key_id, requested_model=body.model,
                         stream=body.stream, quality=x_triage_quality, tau=tau if auto else None,
                         task_type=task, classifier=deps.classifier if auto else None,
                         config_version=snapshot.version, temperature=temperature, max_tokens=max_tokens,
                         settings_match_training=(temperature == training.get("temperature")
                                                  and max_tokens == training.get("max_tokens")) if auto else None,
                         prompt_hmac=prompt_hmac(messages, deps.hash_secret), verified=verify,
                         mode=mode)
        if not auto and body.model in deps.registry:       # a named model is its own primary, even if it fails
            row.primary_model, row.tier = body.model, deps.registry[body.model].tier

        logged = []
        recorder = _RecordingPredictor(deps.predictor)
        request_deps = dataclasses.replace(deps, predictor=recorder)    # same client, registry and log

        def finish(status="ok", http_status=200, error_kind=None):
            if logged:
                return
            logged.append(True)
            row.status, row.http_status = status, http_status
            row.error_kind = error_kind if error_kind is not None else row.error_kind   # keep a verification note
            row.latency_ms = round((time.perf_counter() - started) * 1000, 1)
            _apply_inputs(row, recorder.prediction)
            row.training_eligible = is_training_eligible(row)
            deps.log.write(row)
            request_logger.info(json_log_line(row))

        if not auto and body.model not in deps.registry:              # logged, then an OpenAI-style 404
            finish("error", 404, "model_not_found")
            raise APIError(404, str(core.UnknownModel(body.model)), param="model", code="model_not_found")

        try:
            routed = await core.route_and_stream(messages, tau=tau if auto else None, task_type=task,
                                                 model=body.model, stream=body.stream, settings=settings,
                                                 deps=request_deps, request_id=request_id, keys=keys)
        except core.UnknownModel as error:
            finish("error", 404, "model_not_found")
            raise APIError(404, str(error), param="model", code="model_not_found") from None
        except ModelNotAllowed as error:                   # e.g. a claude-* model, and this request has no key
            finish("error", 403, "model_not_allowed")
            raise APIError(403, _not_allowed_message(error), "permission_error", param="model",
                           code="model_not_allowed") from None
        except core.NoUsableModel:
            finish("error", 400, "no_usable_model")
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
                finish("error", 400, "truncated")
                used = last.response.output_tokens if last.response is not None else "all its"
                raise APIError(400, f"`{body.model}` used {used} output tokens thinking and wrote no answer. "
                               "Send a higher max_tokens: thinking counts toward it.", param="max_tokens",
                               code="max_tokens_too_low") from None
            rejected = rejected_key_providers(error.attempts) if byok else []
            if rejected:                                    # the user's own key(s) were refused: say so, as a 401
                finish("error", 401, "invalid_provider_key")
                raise APIError(401, invalid_key_message(rejected), code="invalid_provider_key") from None
            finish("error", 503, "all_models_failed")
            raise APIError(503, f"No model could answer: {error}", "server_error", code="all_models_failed") from None
        decision = routed.decision
        _apply_decision(row, decision)
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
                finish("error", 502, error.kind)
                raise APIError(502, f"The model failed while answering: {error.kind}", "server_error",
                               code="model_error") from None
            response = routed.response
            served_model, served_tier, reason, check, escalated = decision.model, decision.tier, None, "none", False
            if verify == "sync":
                outcome = await verify_sync(messages, Served(decision.tier, decision.model, response, text),
                                            snapshot.verification.sync, deps, settings,
                                            probabilities={"low": decision.p_low, "medium": decision.p_medium},
                                            keys=keys)
                _apply_verification(row, response, outcome, deps, baseline_providers)
                response, text = outcome.final.response, outcome.final.text
                served_model, served_tier, check, escalated = (outcome.final.model, outcome.final.tier,
                                                               outcome.decided_by, outcome.escalated)
                reason = _verified_reason(decision, outcome)
            else:
                _apply_costs(row, response, deps, baseline_providers)
            _add_truncated_costs(row, decision.attempts)
            finish()
            cache = "hit" if response.cached else ("miss" if is_cacheable(temperature, False) else "bypass")
            headers = _triage_headers(decision, cache, model=served_model, tier=served_tier, reason=reason,
                                      verified=verify, escalated=escalated, check=check) | {
                "X-Triage-Cost": _money(row.price_as_cost)}
            if row.saved is not None:          # unknown when no baseline model is reachable: left out, not faked
                headers["X-Triage-Saved"] = _money(row.saved)
            completion = {"id": request_id, "object": "chat.completion", "created": created, "model": served_model,
                          "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                       "finish_reason": response.finish_reason or "stop"}],
                          "usage": _usage(response)}
            return JSONResponse(completion, headers=headers | request.state.triage_headers,
                                background=background(text))

        include_usage = bool(body.stream_options and body.stream_options.include_usage)
        streamed = []                                         # the answer, in memory only, for a background label

        async def events():
            try:
                async for event in stream_events():
                    yield event
            finally:
                finish("error", 499, "client_disconnected")    # no-op if the row was already written

        async def stream_events():
            yield _chunk(request_id, created, decision.model, {"role": "assistant", "content": ""})
            try:
                async for piece in routed.pieces:
                    streamed.append(piece)
                    yield _chunk(request_id, created, decision.model, {"content": piece})
            except Exception as error:                           # after the first token: no fallback
                kind = getattr(error, "kind", type(error).__name__)
                finish("error", 200, f"stream_interrupted:{kind}")
                logger.warning("stream %s interrupted after the first token: %s", request_id, kind)
                yield stream_error_event(f"The model failed part-way through the answer ({kind}). "
                                         "Triage can't switch models once text has been sent.")
                return
            response = routed.response
            _apply_costs(row, response, deps, baseline_providers)
            _add_truncated_costs(row, decision.attempts)
            finish()
            yield _chunk(request_id, created, decision.model, {}, finish=(response and response.finish_reason) or "stop")
            if include_usage and response is not None:
                yield _chunk(request_id, created, decision.model, None, usage=_usage(response), choices=False)
            yield "data: [DONE]\n\n"

        async def stream_background():                       # runs after the last byte has been sent
            if background(None) is not None:
                await deps.labeler.run(deps, row, messages, "".join(streamed) if row.status == "ok" else None)

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers=_triage_headers(decision, "bypass") | {"Cache-Control": "no-cache"}
                                 | request.state.triage_headers,
                                 background=BackgroundTask(stream_background))

    # ----- models, stats, requests -----

    @app.get("/v1/models")
    async def list_models(who: Caller = Depends(caller)):
        deps = app.state.deps
        server_providers = deps.config.snapshot().routing.server_providers
        data = [{"id": "auto", "object": "model", "created": 0, "owned_by": "triage",
                 "triage": {"description": "Triage picks the cheapest model likely to pass (classifier "
                            f"{deps.classifier or 'configured in routing.yaml'})"}}]
        for name, m in deps.registry.items():
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
        mine = app.state.deps.log.rows(None if who.admin else who.key_id, include_vectors=False)
        return {"scope": "all keys" if who.admin else "your key", **summarize(mine)}

    @app.get("/v1/requests/{request_id}")
    async def get_request(request_id: str, who: Caller = Depends(caller)):
        row = app.state.deps.log.get(request_id, include_vectors=False)
        if row is None or row.key_id != who.key_id:          # someone else's request looks like a missing one
            raise APIError(404, f"No request {request_id!r} for this key.", param="request_id", code="not_found")
        return row.public_dict()

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
