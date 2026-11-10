"""Triage's OpenAI-compatible API (Phase 4 Step 5).

Run it:
    TRIAGE_API_KEYS=dev-key TRIAGE_ADMIN_KEYS=admin-key TRIAGE_FAKE_MODELS=1 uvicorn app.api.main:app
(drop TRIAGE_FAKE_MODELS for real models; TRIAGE_CACHE=off disables the response cache)

Endpoints
    POST /v1/chat/completions     OpenAI chat completions, stream or not; model "auto" or a models.yaml name
    GET  /v1/models               "auto" plus every model, with prices
    GET  /v1/stats                savings summary (your key's requests; an admin key sees all)
    GET  /v1/requests/{id}        one logged request, only for the key that made it
    GET  /v1/routing-config       the config in use (admin)
    PUT  /v1/routing-config       update quality presets/default and backup models live (admin; not persisted)

Response headers: X-Triage-Request-Id, -Model, -Tier, -Reason, -Fallback, -Cache, -Verified, -Escalated,
and on non-streaming responses -Cost and -Saved. A stream's headers are sent once its first token exists, so
model, tier and fallback are real; its cost isn't known yet, so -Cost and -Saved are left out (never faked).
The real figures are in GET /v1/requests/{id} once the stream ends, and in /v1/stats.
"""
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

from app.api import core
from app.api.auth import ADMIN_KEYS_ENV, API_KEYS_ENV, Auth, Caller, keys_from_env
from app.api.config_state import NOT_PERSISTED_NOTE, ConfigUpdateError
from app.api.errors import APIError, bad_request, stream_error_event
from app.api.headers import parse_quality, parse_task, parse_verify
from app.api.request_log import RequestRow
from app.api.stats import summarize
from app.cache import is_cacheable
from app.providers.base import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE
from app.providers.common import ProviderError
from app.reliability import AllModelsFailed

logger = logging.getLogger("triage.api")

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


def _prompt_sha(messages: list) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _header_safe(text: str) -> str:
    return text.encode("latin-1", "replace").decode("latin-1")


def _apply_costs(row: RequestRow, response, deps) -> None:
    if response is None:
        return
    strong = deps.registry[deps.strong_model()]
    row.cache_hit = bool(response.cached)
    row.input_tokens, row.output_tokens = response.input_tokens, response.output_tokens
    row.reasoning_tokens, row.tokens_estimated = response.reasoning_tokens, response.tokens_estimated
    row.cost, row.price_as_cost = response.cost, response.reported_cost
    row.strong_cost = strong.reported_cost(response.input_tokens, response.output_tokens)
    row.saved = row.strong_cost - row.price_as_cost


def _apply_decision(row: RequestRow, d: core.RouteDecision) -> None:
    row.tau, row.p_low, row.p_medium = d.tau, d.p_low, d.p_medium
    row.task_type, row.task_type_guessed = d.task_type, d.task_type_guessed
    row.tier, row.primary_model, row.served_model, row.fell_back = d.tier, d.primary_model, d.model, d.fell_back
    row.attempts = [{"model": a.model_name, "outcome": a.outcome, "error_kind": a.error_kind} for a in d.attempts]


def _triage_headers(d: core.RouteDecision, cache: str) -> dict:
    return {"X-Triage-Request-Id": d.request_id, "X-Triage-Model": d.model, "X-Triage-Tier": d.tier,
            "X-Triage-Reason": _header_safe(d.reason),
            "X-Triage-Fallback": f"{d.primary_model}->{d.model}" if d.fell_back else "none",
            "X-Triage-Cache": cache, "X-Triage-Verified": "off", "X-Triage-Escalated": "false"}


def _usage(response) -> dict:
    return {"prompt_tokens": response.input_tokens, "completion_tokens": response.output_tokens,
            "total_tokens": response.input_tokens + response.output_tokens}


def _chunk(request_id: str, created: int, model: str, delta: dict, finish=None, usage=None, choices=True) -> str:
    body = {"id": request_id, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}] if choices else []}
    if usage is not None:
        body["usage"] = usage
    return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"


def _error_response(error: APIError, headers: Optional[dict] = None) -> JSONResponse:
    return JSONResponse(error.body(), status_code=error.status, headers=headers)


# ---------- the app ----------

def create_app(deps=None, auth: Optional[Auth] = None) -> FastAPI:
    """deps / auth given: use them (tests). Otherwise load both at startup from config and the environment,
    refusing to start if there are no keys or the predictor fails its safety checks."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.auth is None:
            app.state.auth = Auth(keys_from_env(API_KEYS_ENV), keys_from_env(ADMIN_KEYS_ENV))
        if app.state.deps is None:
            from app.api.deps import load_deps
            app.state.deps = load_deps()          # predictor + embedding model, once
        yield

    app = FastAPI(title="Triage", version="0.5", lifespan=lifespan)
    app.state.deps, app.state.auth = deps, auth

    @app.exception_handler(APIError)
    async def api_error(request: Request, error: APIError):
        return _error_response(error)

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

    # ----- chat -----

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatRequest, request: Request, who: Caller = Depends(caller),
                               x_triage_quality: Optional[str] = Header(None),
                               x_triage_task: Optional[str] = Header(None),
                               x_triage_verify: Optional[str] = Header(None)):
        deps = app.state.deps
        started = time.perf_counter()
        unknown = sorted(set(body.model_extra or {}) - ACCEPTED_FIELDS)
        if unknown:
            raise bad_request(f"Triage doesn't support these parameters yet: {', '.join(unknown)}", unknown[0],
                              "unsupported_parameter")
        if body.n not in (None, 1):
            raise bad_request("Triage returns one choice: n must be 1", "n", "unsupported_parameter")
        snapshot = deps.config.snapshot()
        tau = parse_quality(x_triage_quality, snapshot.quality)
        task = parse_task(x_triage_task)
        parse_verify(x_triage_verify, body.stream)
        auto = body.model == "auto"
        if not auto and body.model not in deps.registry:
            raise APIError(404, str(core.UnknownModel(body.model)), param="model", code="model_not_found")

        messages = [m.model_dump() for m in body.messages]
        settings = core.GenerationSettings(body.temperature, body.max_tokens or body.max_completion_tokens)
        temperature = settings.temperature if settings.temperature is not None else \
            snapshot.routing.generation.get("temperature", DEFAULT_TEMPERATURE)
        max_tokens = settings.max_tokens or snapshot.routing.generation.get("max_tokens", DEFAULT_MAX_TOKENS)
        training = snapshot.routing.generation
        request_id = f"chatcmpl-triage-{uuid.uuid4().hex}"
        row = RequestRow(request_id=request_id, timestamp=_now(), key_id=who.key_id, requested_model=body.model,
                         stream=body.stream, quality=x_triage_quality, tau=tau if auto else None,
                         task_type=task, classifier=deps.classifier if auto else None,
                         config_version=snapshot.version, temperature=temperature, max_tokens=max_tokens,
                         settings_match_training=(temperature == training.get("temperature")
                                                  and max_tokens == training.get("max_tokens")) if auto else None,
                         prompt_sha256=_prompt_sha(messages))

        logged = []

        def finish(status="ok", http_status=200, error_kind=None):
            if logged:
                return
            logged.append(True)
            row.status, row.http_status, row.error_kind = status, http_status, error_kind
            row.latency_ms = round((time.perf_counter() - started) * 1000, 1)
            deps.log.write(row)

        try:
            routed = await core.route_and_stream(messages, tau=tau if auto else None, task_type=task,
                                                 model=body.model, stream=body.stream, settings=settings,
                                                 deps=deps, request_id=request_id)
        except core.UnknownModel as error:
            finish("error", 404, "model_not_found")
            raise APIError(404, str(error), param="model", code="model_not_found") from None
        except AllModelsFailed as error:
            finish("error", 503, "all_models_failed")
            raise APIError(503, f"No model could answer: {error}", "server_error", code="all_models_failed") from None
        decision = routed.decision
        _apply_decision(row, decision)
        row.ttft_ms = round((time.perf_counter() - started) * 1000, 1)
        created = int(time.time())

        if not body.stream:
            try:
                text = "".join([piece async for piece in routed.pieces])
            except ProviderError as error:
                finish("error", 502, error.kind)
                raise APIError(502, f"The model failed while answering: {error.kind}", "server_error",
                               code="model_error") from None
            response = routed.response
            _apply_costs(row, response, deps)
            finish()
            cache = "hit" if response.cached else ("miss" if is_cacheable(temperature, False) else "bypass")
            headers = _triage_headers(decision, cache) | {"X-Triage-Cost": _money(row.price_as_cost),
                                                          "X-Triage-Saved": _money(row.saved)}
            completion = {"id": request_id, "object": "chat.completion", "created": created, "model": decision.model,
                          "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                       "finish_reason": response.finish_reason or "stop"}],
                          "usage": _usage(response)}
            return JSONResponse(completion, headers=headers)

        include_usage = bool(body.stream_options and body.stream_options.include_usage)

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
                    yield _chunk(request_id, created, decision.model, {"content": piece})
            except Exception as error:                           # after the first token: no fallback
                kind = getattr(error, "kind", type(error).__name__)
                finish("error", 200, f"stream_interrupted:{kind}")
                logger.warning("stream %s interrupted after the first token: %s", request_id, kind)
                yield stream_error_event(f"The model failed part-way through the answer ({kind}). "
                                         "Triage can't switch models once text has been sent.")
                return
            response = routed.response
            _apply_costs(row, response, deps)
            finish()
            yield _chunk(request_id, created, decision.model, {}, finish=(response and response.finish_reason) or "stop")
            if include_usage and response is not None:
                yield _chunk(request_id, created, decision.model, None, usage=_usage(response), choices=False)
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers=_triage_headers(decision, "bypass") | {"Cache-Control": "no-cache"})

    # ----- models, stats, requests -----

    @app.get("/v1/models")
    async def list_models(who: Caller = Depends(caller)):
        deps = app.state.deps
        data = [{"id": "auto", "object": "model", "created": 0, "owned_by": "triage",
                 "triage": {"description": "Triage picks the cheapest model likely to pass (classifier "
                            f"{deps.classifier or 'configured in routing.yaml'})"}}]
        for name, m in deps.registry.items():
            price = m.price_as
            data.append({"id": name, "object": "model", "created": 0, "owned_by": m.provider,
                         "triage": {"tier": m.tier, "provider_model": m.model_id,
                                    "price_per_1m_input_usd": price.input_price if price else m.input_price,
                                    "price_per_1m_output_usd": price.output_price if price else m.output_price,
                                    "price_label": price.label if price else "actual price"}})
        return {"object": "list", "data": data}

    @app.get("/v1/stats")
    async def stats(who: Caller = Depends(caller)):
        rows = app.state.deps.log.rows()
        mine = rows if who.admin else [r for r in rows if r.key_id == who.key_id]
        return {"scope": "all keys" if who.admin else "your key", **summarize(mine)}

    @app.get("/v1/requests/{request_id}")
    async def get_request(request_id: str, who: Caller = Depends(caller)):
        row = app.state.deps.log.get(request_id)
        if row is None or row.key_id != who.key_id:          # someone else's request looks like a missing one
            raise APIError(404, f"No request {request_id!r} for this key.", param="request_id", code="not_found")
        return row.to_dict()

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
