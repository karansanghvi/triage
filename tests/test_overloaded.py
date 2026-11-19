"""Overloaded providers: no hidden SDK retries in any adapter, "overloaded" as its own error kind (never retried on
the same model), gemini-flash-lite as the last resort for Gemini-only users, and the provider_overloaded API error.
No network: httpx MockTransport."""
import dataclasses
import json
import random

import httpx
import httpx2
import pytest
from fastapi.testclient import TestClient
from google import genai

from app.api.auth import Auth
from app.api.fake_models import FakeBehavior
from app.api.main import create_app
from app.providers import anthropic_provider, gemini_provider, openai_provider
from app.providers.base import send_request
from app.providers.common import ProviderError
from app.registry import load_registry
from app.reliability import ReliableClient, load_routing
from tests.api_helpers import API_KEY, ASK, FakePredictor, make_deps

REGISTRY = load_registry()
MESSAGES = [{"role": "user", "content": "Hi"}]
SECRET = "sk-test-SECRET123"

GEMINI_REPLY = {"candidates": [{"content": {"role": "model", "parts": [{"text": "from lite"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2}}
GEMINI_BUSY = (503, {"error": {"code": 503, "message": "This model is currently experiencing high demand. Spikes in "
                               "demand are usually temporary. Please try again later.", "status": "UNAVAILABLE"}})
ANTHROPIC_BUSY = (529, {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})
OPENAI_BUSY = (503, {"error": {"message": "The engine is currently overloaded, please try again later.",
                               "type": "server_error", "param": None, "code": None}})


class Wire:
    """Answers with (status, body), or per Gemini model when `by_model` is given; keeps every request."""

    def __init__(self, status=200, body=None, lib=httpx, by_model=None):
        self.status, self.body, self.lib, self.by_model, self.requests = status, body, lib, by_model, []

    def __call__(self, request):
        self.requests.append(request)
        status, body = self.status, self.body
        if self.by_model:
            status, body = next(v for k, v in self.by_model.items() if f"/models/{k}:" in request.url.path)
        return self.lib.Response(status, content=json.dumps(body), headers={"content-type": "application/json"})

    def client(self):
        return self.lib.Client(transport=self.lib.MockTransport(self))


@pytest.fixture(autouse=True)
def env_keys(monkeypatch):
    for env in ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(env, "server-env-key")


# ---------- 1. no SDK retries, in any adapter ----------

ADAPTERS = [
    ("gemini-flash", gemini_provider, httpx, GEMINI_BUSY),
    ("openai-luna", openai_provider, httpx, OPENAI_BUSY),
    ("groq-small", openai_provider, httpx, (503, {"error": {"message": "over capacity"}})),
    ("groq-small", openai_provider, httpx, (429, {"error": {"message": "rate limited"}})),
    ("claude-haiku", anthropic_provider, httpx2, ANTHROPIC_BUSY),
    ("claude-haiku", anthropic_provider, httpx2, (500, {"type": "error", "error": {"type": "api_error",
                                                                                   "message": "boom"}})),
]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("model, module, lib, reply", ADAPTERS)
def test_every_adapter_sends_exactly_one_request_on_a_retryable_status(monkeypatch, model, module, lib, reply,
                                                                       stream):
    """The SDKs retry 429/5xx on their own by default (openai and anthropic: max_retries=2); Triage's reliability
    layer must be the only one deciding, so each failure is exactly one request."""
    wire = Wire(*reply, lib=lib)
    monkeypatch.setattr(module, "HTTP_CLIENT", wire.client())
    with pytest.raises(ProviderError):
        result = send_request(MESSAGES, REGISTRY[model], stream=stream, max_tokens=100, timeout=30, api_key=SECRET)
        if stream:
            list(result)
    assert len(wire.requests) == 1


def test_the_gemini_sdk_is_told_to_make_one_attempt(monkeypatch):
    seen = []
    real = genai.Client

    def spy(**kwargs):
        seen.append(kwargs["http_options"].retry_options)
        return real(**kwargs)

    monkeypatch.setattr(gemini_provider.genai, "Client", spy)
    monkeypatch.setattr(gemini_provider, "HTTP_CLIENT", Wire(200, GEMINI_REPLY).client())
    send_request(MESSAGES, REGISTRY["gemini-flash"], max_tokens=100, timeout=30)
    (options,) = seen
    assert options is not None and options.attempts == 1


@pytest.mark.parametrize("module, kwargs", [
    (openai_provider, {"provider": "groq", "base_url": "https://api.groq.com/openai/v1"}),
    (anthropic_provider, {}),
])
def test_the_openai_and_anthropic_clients_are_built_with_max_retries_0(monkeypatch, module, kwargs):
    seen = []
    cls_name = "OpenAI" if module is openai_provider else "Anthropic"
    owner = module if module is openai_provider else module.anthropic
    real = getattr(owner, cls_name)

    def spy(*args, **kw):
        seen.append(kw.get("max_retries"))
        return real(*args, **kw)

    monkeypatch.setattr(owner, cls_name, spy)
    model = "groq-small" if module is openai_provider else "claude-haiku"
    lib = httpx if module is openai_provider else httpx2
    monkeypatch.setattr(module, "HTTP_CLIENT", Wire(500, {"error": {"message": "x"}}, lib=lib).client())
    with pytest.raises(ProviderError):
        send_request(MESSAGES, REGISTRY[model], max_tokens=100, timeout=30, api_key=SECRET)
    assert seen == [0]


# ---------- 2. "overloaded" ----------

@pytest.mark.parametrize("model, module, lib, reply, kind", [
    ("gemini-flash", gemini_provider, httpx, GEMINI_BUSY, "overloaded"),
    ("claude-haiku", anthropic_provider, httpx2, ANTHROPIC_BUSY, "overloaded"),
    # overloaded_error with another status still counts (the error type is what Anthropic documents)
    ("claude-haiku", anthropic_provider, httpx2,
     (503, {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}), "overloaded"),
    ("claude-haiku", anthropic_provider, httpx2,
     (500, {"type": "error", "error": {"type": "api_error", "message": "boom"}}), "server"),
    ("openai-luna", openai_provider, httpx, OPENAI_BUSY, "overloaded"),
    ("openai-luna", openai_provider, httpx, (500, {"error": {"message": "boom"}}), "server"),
    ("groq-small", openai_provider, httpx, (503, {"error": {"message": "over capacity"}}), "server"),   # unchanged
])
def test_overload_answers_are_their_own_kind(monkeypatch, model, module, lib, reply, kind):
    monkeypatch.setattr(module, "HTTP_CLIENT", Wire(*reply, lib=lib).client())
    with pytest.raises(ProviderError) as info:
        send_request(MESSAGES, REGISTRY[model], max_tokens=100, timeout=30, api_key=SECRET)
    assert info.value.kind == kind
    assert info.value.retryable is (kind == "server")


def test_an_overloaded_gemini_flash_falls_straight_back_to_flash_lite(monkeypatch):
    """A Gemini-only request: Flash once (no retry, no backoff), then the last resort answers."""
    wire = Wire(by_model={"gemini-3.8-flash": GEMINI_BUSY, "gemini-3.5-flash-lite": (200, GEMINI_REPLY)})
    monkeypatch.setattr(gemini_provider, "HTTP_CLIENT", wire.client())
    sleeps = []
    client = ReliableClient(REGISTRY, load_routing(REGISTRY), sleep=sleeps.append, rng=random.Random(0))
    for tier in ("medium", "high"):
        wire.requests.clear()
        result = client.send(MESSAGES, tier, providers={"gemini"})
        assert result.model.name == "gemini-flash-lite" and result.response.text == "from lite"
        assert [(a.model_name, a.error_kind) for a in result.attempts] == [("gemini-flash", "overloaded"),
                                                                           ("gemini-flash-lite", None)]
        assert len(wire.requests) == 2
    assert sleeps == []


def test_overloaded_trips_only_that_models_breaker(monkeypatch):
    monkeypatch.setattr(gemini_provider, "HTTP_CLIENT", Wire(*GEMINI_BUSY).client())
    routing = load_routing(REGISTRY)
    client = ReliableClient(REGISTRY, dataclasses.replace(routing, chains={"medium": ["gemini-flash"]}),
                            sleep=lambda s: None)
    for _ in range(routing.breaker.failure_threshold):
        with pytest.raises(Exception):
            client.send(MESSAGES, "medium")
    assert client.breaker.is_open("model:gemini-flash") and not client.breaker.is_open("gemini")


# ---------- 3. gemini-flash-lite as the last resort ----------

def test_gemini_flash_lite_is_last_in_medium_and_high():
    routing = load_routing(REGISTRY)
    for tier in ("medium", "high"):
        assert routing.chains[tier][-1] == "gemini-flash-lite"
        assert routing.chain_for(tier, REGISTRY, {"gemini"}) == ["gemini-flash", "gemini-flash-lite"]
        # preferring Gemini doesn't move Flash-Lite ahead of the other providers
        everyone = routing.chain_for(tier, REGISTRY, {"groq", "gemini", "openai", "anthropic"}, "gemini")
        assert everyone[0] == "gemini-flash" and everyone[-1] == "gemini-flash-lite"


# ---------- 4. the API error ----------

def app(script):
    deps = make_deps(script, predictor=FakePredictor(0.4, 0.95))
    return TestClient(create_app(deps=deps, auth=Auth([API_KEY]))), deps


def chat(http, headers=None):
    return http.post("/v1/chat/completions", json={"model": "auto", "messages": ASK},
                     headers={"Authorization": f"Bearer {API_KEY}"} | (headers or {}))


BUSY = FakeBehavior(fail_before="overloaded")


def test_a_gemini_only_user_gets_an_answer_from_flash_lite_when_flash_is_busy():
    http, deps = app({"gemini-flash": BUSY})
    r = chat(http, {"x-gemini-key": SECRET})
    assert r.status_code == 200 and r.headers["x-triage-model"] == "gemini-flash-lite"
    assert [c["model"] for c in deps.fake_models.calls] == ["gemini-flash", "gemini-flash-lite"]   # no retry


def test_every_model_overloaded_is_a_503_that_names_the_provider():
    http, deps = app({name: BUSY for name in REGISTRY})
    r = chat(http, {"x-gemini-key": SECRET})
    assert r.status_code == 503
    err = r.json()["error"]
    assert err["code"] == "provider_overloaded" and err["type"] == "server_error"
    assert err["message"] == ("Gemini is very busy right now. Try again in a moment, or add another provider key "
                              "so Triage can fall back.")
    assert SECRET not in r.text
    (row,) = deps.log.rows()
    assert row.http_status == 503 and row.error_kind == "provider_overloaded"


def test_server_keys_overloaded_say_just_try_again_in_a_moment():
    http, _ = app({name: BUSY for name in REGISTRY})
    r = chat(http)
    assert r.status_code == 503 and r.json()["error"]["code"] == "provider_overloaded"
    assert r.json()["error"]["message"] == "Groq and Gemini are very busy right now. Try again in a moment."


def test_overloaded_mixed_with_other_failures_keeps_all_models_failed():
    script = {name: BUSY for name in REGISTRY}
    script["groq-small"] = FakeBehavior(fail_before="server")
    http, _ = app(script)
    r = chat(http)
    assert r.status_code == 503 and r.json()["error"]["code"] == "all_models_failed"


# ---------- busy and slow mixed: provider_unavailable ----------

SLOW = FakeBehavior(fail_before="timeout")


def test_overloaded_and_timeouts_mixed_are_provider_unavailable():
    script = {name: BUSY for name in REGISTRY}
    script["gemini-flash-lite"] = SLOW
    http, deps = app(script)
    r = chat(http, {"x-gemini-key": SECRET})
    assert r.status_code == 503
    err = r.json()["error"]
    assert err["code"] == "provider_unavailable" and err["type"] == "server_error"
    assert err["message"] == ("Gemini couldn't answer right now (busy or too slow). Try again in a moment, or add "
                              "another provider key so Triage can fall back.")
    (row,) = deps.log.rows()
    assert row.http_status == 503 and row.error_kind == "provider_unavailable"


def test_server_keys_busy_and_slow_say_just_try_again_in_a_moment():
    script = {name: SLOW for name in REGISTRY}
    script["groq-small"] = BUSY
    http, _ = app(script)
    r = chat(http)
    assert r.status_code == 503 and r.json()["error"]["code"] == "provider_unavailable"
    assert r.json()["error"]["message"] == ("Groq and Gemini couldn't answer right now (busy or too slow). "
                                            "Try again in a moment.")


@pytest.mark.parametrize("behavior, code, status", [(BUSY, "provider_overloaded", 503),
                                                    (SLOW, "provider_timeout", 504)])
def test_the_pure_cases_keep_their_own_codes(behavior, code, status):
    http, _ = app({name: behavior for name in REGISTRY})
    r = chat(http, {"x-gemini-key": SECRET})
    assert r.status_code == status and r.json()["error"]["code"] == code


def test_busy_slow_and_a_server_error_is_still_all_models_failed():
    script = {name: BUSY for name in REGISTRY}
    script["gemini-flash"] = SLOW
    script["groq-small"] = FakeBehavior(fail_before="server")
    http, _ = app(script)
    assert chat(http).json()["error"]["code"] == "all_models_failed"
