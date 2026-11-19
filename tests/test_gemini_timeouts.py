"""Gemini timeouts and thinking: the exact time limit and ThinkingConfig sent to the SDK, Google's deadline errors
classified as "timeout", no retries of a model that timed out, and the provider_timeout API error.
No network: httpx MockTransport."""
import dataclasses
import json
import random

import httpx
import pytest
from fastapi.testclient import TestClient
from google import genai
from google.genai import types

from app.api.auth import Auth
from app.api.fake_models import FakeBehavior
from app.api.main import create_app
from app.providers import gemini_provider, openai_provider
from app.providers.base import send_request
from app.providers.common import ProviderError
from app.registry import load_registry
from app.reliability import AllModelsFailed, ReliableClient, load_routing, routing_from_dict
from tests.api_helpers import API_KEY, ASK, FakePredictor, make_deps, routing_raw

REGISTRY = load_registry()
MESSAGES = [{"role": "user", "content": "Hi"}]
SECRET = "gm-test-SECRET123"

GEMINI_REPLY = {"candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2}}
GROQ_REPLY = {"id": "c", "object": "chat.completion", "created": 0, "model": "m",
              "choices": [{"index": 0, "message": {"role": "assistant", "content": "from groq"},
                           "finish_reason": "stop"}],
              "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
# What Google sends when its server-side deadline (our X-Server-Timeout) runs out
DEADLINE = (504, {"error": {"code": 504, "message": "Deadline expired before operation could complete.",
                            "status": "DEADLINE_EXCEEDED"}})


class Wire:
    """Answers every request with the same status and JSON body, and keeps the requests."""

    def __init__(self, status, body):
        self.status, self.body, self.requests = status, body, []

    def __call__(self, request):
        self.requests.append(request)
        return httpx.Response(self.status, content=json.dumps(self.body), headers={"content-type": "application/json"})

    def client(self):
        return httpx.Client(transport=httpx.MockTransport(self))


@pytest.fixture(autouse=True)
def env_keys(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "server-gemini-key")
    monkeypatch.setenv("GROQ_API_KEY", "server-groq-key")


def gemini_wire(monkeypatch, status=200, body=GEMINI_REPLY):
    wire = Wire(status, body)
    monkeypatch.setattr(gemini_provider, "HTTP_CLIENT", wire.client())
    return wire


# ---------- 1. the exact time limit ----------

def test_the_sdk_gets_the_timeout_in_milliseconds(monkeypatch):
    """HttpOptions.timeout is in milliseconds: 30 s must reach the SDK as 30000, never 30 (= 30 ms)."""
    seen = []
    real_client = genai.Client

    def spy(**kwargs):
        seen.append(kwargs["http_options"].timeout)
        return real_client(**kwargs)

    monkeypatch.setattr(gemini_provider.genai, "Client", spy)
    gemini_wire(monkeypatch)
    send_request(MESSAGES, REGISTRY["gemini-flash"], max_tokens=100, timeout=30)
    assert seen == [30000]


@pytest.mark.parametrize("seconds, header", [(30, "30"), (8, "8"), (2.5, "3")])
def test_the_request_on_the_wire_has_the_timeout_in_seconds(monkeypatch, seconds, header):
    """What actually goes out: httpx waits `seconds`, and Google is told the same limit (rounded up)."""
    wire = gemini_wire(monkeypatch)
    send_request(MESSAGES, REGISTRY["gemini-flash"], max_tokens=100, timeout=seconds)
    (request,) = wire.requests
    assert request.extensions["timeout"]["read"] == pytest.approx(seconds)
    assert request.headers["x-server-timeout"] == header


def test_the_routing_timeout_for_gemini_flash_reaches_the_wire(monkeypatch):
    routing = load_routing(REGISTRY)
    wire = gemini_wire(monkeypatch)
    ReliableClient(REGISTRY, routing, sleep=lambda s: None).send_model(MESSAGES, "gemini-flash")
    (request,) = wire.requests
    expected = routing.timeout_for("gemini-flash")
    assert request.extensions["timeout"]["read"] == pytest.approx(expected)
    assert request.headers["x-server-timeout"] == str(int(expected))


# ---------- 2. Google's deadline errors are timeouts ----------

@pytest.mark.parametrize("status, body", [
    DEADLINE,
    (504, {"error": {"code": 504, "message": "Gateway timeout"}}),
    (500, {"error": {"code": 500, "message": "Deadline expired before operation could complete.",
                     "status": "INTERNAL"}}),
    (499, {"error": {"code": 499, "message": "cancelled", "status": "DEADLINE_EXCEEDED"}}),
])
def test_deadline_errors_are_timeouts(monkeypatch, status, body):
    gemini_wire(monkeypatch, status, body)
    with pytest.raises(ProviderError) as info:
        send_request(MESSAGES, REGISTRY["gemini-flash"], max_tokens=100, timeout=30)
    assert info.value.kind == "timeout"
    assert not info.value.retryable


def test_a_deadline_while_streaming_is_a_timeout(monkeypatch):
    gemini_wire(monkeypatch, *DEADLINE)
    stream = send_request(MESSAGES, REGISTRY["gemini-flash"], stream=True, max_tokens=100, timeout=30)
    with pytest.raises(ProviderError) as info:
        list(stream)
    assert info.value.kind == "timeout"


@pytest.mark.parametrize("status, google_status, kind", [(500, "INTERNAL", "server"),
                                                         (429, "RESOURCE_EXHAUSTED", "rate_limit"),
                                                         (403, "PERMISSION_DENIED", "auth")])
def test_other_errors_keep_their_kind(monkeypatch, status, google_status, kind):
    gemini_wire(monkeypatch, status, {"error": {"code": status, "message": "nope", "status": google_status}})
    with pytest.raises(ProviderError) as info:
        send_request(MESSAGES, REGISTRY["gemini-flash"], max_tokens=100, timeout=30)
    assert info.value.kind == kind


# ---------- 3. a timeout is never retried on the same model ----------

def test_a_gemini_timeout_moves_straight_to_the_next_model(monkeypatch):
    gemini = gemini_wire(monkeypatch, *DEADLINE)
    groq = Wire(200, GROQ_REPLY)
    monkeypatch.setattr(openai_provider, "HTTP_CLIENT", groq.client())
    routing = dataclasses.replace(load_routing(REGISTRY), chains={"medium": ["gemini-flash", "groq-small"]})
    sleeps = []
    client = ReliableClient(REGISTRY, routing, sleep=sleeps.append, rng=random.Random(0))
    result = client.send(MESSAGES, "medium")
    assert result.model.name == "groq-small" and result.response.text == "from groq"
    assert len(gemini.requests) == 1 and sleeps == []            # one try, no backoff
    assert [(a.model_name, a.error_kind) for a in result.attempts] == [("gemini-flash", "timeout"),
                                                                       ("groq-small", None)]


def test_a_gemini_only_chain_fails_after_one_try(monkeypatch):
    gemini = gemini_wire(monkeypatch, *DEADLINE)
    sleeps = []
    client = ReliableClient(REGISTRY, load_routing(REGISTRY), sleep=sleeps.append)
    with pytest.raises(AllModelsFailed) as info:
        client.send_model(MESSAGES, "gemini-flash")
    assert len(gemini.requests) == 1 and sleeps == []
    assert [a.error_kind for a in info.value.attempts] == ["timeout"]


# ---------- the API error ----------

def app(script):
    deps = make_deps(script, predictor=FakePredictor(0.4, 0.95))
    return TestClient(create_app(deps=deps, auth=Auth([API_KEY]))), deps


def chat(http, headers=None):
    return http.post("/v1/chat/completions", json={"model": "auto", "messages": ASK},
                     headers={"Authorization": f"Bearer {API_KEY}"} | (headers or {}))


def test_every_model_timing_out_is_a_504_that_names_the_provider():
    http, deps = app({name: FakeBehavior(fail_before="timeout") for name in REGISTRY})
    r = chat(http, {"x-gemini-key": SECRET})
    assert r.status_code == 504
    err = r.json()["error"]
    assert err["code"] == "provider_timeout" and err["type"] == "server_error"
    assert err["message"] == ("Gemini took too long to answer. Try again, or add another provider key so Triage "
                              "can fall back.")
    assert SECRET not in r.text
    (row,) = deps.log.rows()
    assert row.http_status == 504 and row.error_kind == "provider_timeout"


def test_server_keys_timing_out_dont_suggest_adding_a_key():
    http, _ = app({name: FakeBehavior(fail_before="timeout") for name in REGISTRY})
    r = chat(http)
    assert r.status_code == 504 and r.json()["error"]["code"] == "provider_timeout"
    message = r.json()["error"]["message"]
    assert message.endswith("took too long to answer. Try again.") and "provider key" not in message


def test_mixed_failures_keep_the_503():
    script = {name: FakeBehavior(fail_before="server") for name in REGISTRY}
    script["gemini-flash"] = FakeBehavior(fail_before="timeout")
    script["gemini-flash-lite"] = FakeBehavior(fail_before="timeout")
    http, _ = app(script)
    r = chat(http)
    assert r.status_code == 503 and r.json()["error"]["code"] == "all_models_failed"


# ---------- thinking_level: LOW for gemini-flash only ----------

def body_of(request) -> dict:
    return json.loads(request.content)


# What google-genai 2.29 puts on the wire: it copies ThinkingConfig into generationConfig.thinkingConfig as is,
# so the inner field keeps its snake_case name (Google's JSON API accepts proto field names either way)
LOW_ON_THE_WIRE = {"thinking_level": "LOW"}


def test_thinking_level_reaches_the_sdk_as_a_thinking_config(monkeypatch):
    seen = []
    real = gemini_provider.types.GenerateContentConfig

    def spy(**kwargs):
        seen.append(kwargs.get("thinking_config"))
        return real(**kwargs)

    monkeypatch.setattr(gemini_provider.types, "GenerateContentConfig", spy)
    gemini_wire(monkeypatch)
    send_request(MESSAGES, REGISTRY["gemini-flash"], max_tokens=100, timeout=30, thinking_level="low")
    assert seen == [types.ThinkingConfig(thinking_level="LOW")]
    assert seen[0].thinking_level == types.ThinkingLevel.LOW


@pytest.mark.parametrize("stream", [False, True])
def test_thinking_level_on_the_wire(monkeypatch, stream):
    wire = gemini_wire(monkeypatch)
    result = send_request(MESSAGES, REGISTRY["gemini-flash"], stream=stream, max_tokens=100, timeout=30,
                          thinking_level="low")
    if stream:
        list(result)
    (request,) = wire.requests
    assert body_of(request)["generationConfig"]["thinkingConfig"] == LOW_ON_THE_WIRE


def test_without_thinking_level_no_thinking_config_is_sent(monkeypatch):
    wire = gemini_wire(monkeypatch)
    send_request(MESSAGES, REGISTRY["gemini-flash-lite"], max_tokens=100, timeout=30)
    (request,) = wire.requests
    assert "thinkingConfig" not in body_of(request).get("generationConfig", {})


def test_thinking_level_is_refused_for_other_providers():
    with pytest.raises(ProviderError) as info:
        send_request(MESSAGES, REGISTRY["groq-small"], max_tokens=100, thinking_level="low")
    assert info.value.kind == "config"


def test_routing_yaml_gives_low_thinking_to_gemini_flash_only():
    params = load_routing(REGISTRY).model_params
    assert params["gemini-flash"] == {"thinking_level": "low"}
    assert "gemini-flash-lite" not in params           # already "minimal" by default: "low" would be more
    assert [name for name, p in params.items() if "thinking_level" in p] == ["gemini-flash"]


def test_live_calls_send_low_for_gemini_flash_and_nothing_new_elsewhere(monkeypatch):
    """Through the real ReliableClient and routing.yaml: gemini-flash gets LOW, gemini-flash-lite no config,
    and a Groq call's body is as before (reasoning_effort only, no thinking fields)."""
    gemini = gemini_wire(monkeypatch)
    groq = Wire(200, GROQ_REPLY)
    monkeypatch.setattr(openai_provider, "HTTP_CLIENT", groq.client())
    client = ReliableClient(REGISTRY, load_routing(REGISTRY), sleep=lambda s: None)
    client.send_model(MESSAGES, "gemini-flash")
    client.send_model(MESSAGES, "gemini-flash-lite")
    client.send_model(MESSAGES, "groq-small")
    flash, lite = (body_of(r).get("generationConfig", {}) for r in gemini.requests)
    assert flash["thinkingConfig"] == LOW_ON_THE_WIRE
    assert "thinkingConfig" not in lite
    (groq_request,) = groq.requests
    groq_body = body_of(groq_request)
    assert groq_body["reasoning_effort"] == "low"
    assert not {k for k in groq_body if "thinking" in k}


@pytest.mark.parametrize("headers", [{}, {"x-gemini-key": SECRET}])
def test_demo_and_own_key_requests_both_send_low_to_gemini_flash(headers):
    """Server keys (demo-style, no user key) and a user's own key: the same per-model setting."""
    script = {name: FakeBehavior(fail_before="server") for name in REGISTRY}
    script["gemini-flash"] = FakeBehavior(text="ok")
    http, deps = app(script)
    r = chat(http, headers)
    assert r.status_code == 200 and r.headers["x-triage-model"] == "gemini-flash"
    by_model = {c["model"]: c["thinking_level"] for c in deps.fake_models.calls}
    assert by_model.pop("gemini-flash") == "low"
    assert set(by_model.values()) <= {None}


@pytest.mark.parametrize("model, params, message", [
    ("groq-small", {"thinking_level": "low"}, "does not accept thinking_level"),
    ("gemini-flash", {"thinking_level": "minimal"}, "rejects thinking_level minimal"),
    ("gemini-flash", {"thinking_level": "max"}, "thinking_level must be one of"),
])
def test_routing_config_rejects_bad_thinking_levels(model, params, message):
    raw = routing_raw()
    raw["model_params"] = {**raw.get("model_params", {}), model: params}
    with pytest.raises(ValueError, match=message):
        routing_from_dict(raw, REGISTRY)


def test_thinking_level_is_part_of_the_cache_key():
    from app.cache import make_cache_key
    model = REGISTRY["gemini-flash"]
    assert make_cache_key(MESSAGES, model, 0.0, 100) != make_cache_key(MESSAGES, model, 0.0, 100,
                                                                        thinking_level="low")
    assert make_cache_key(MESSAGES, model, 0.0, 100) == make_cache_key(MESSAGES, model, 0.0, 100,
                                                                        thinking_level=None)
