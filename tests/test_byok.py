"""BYOK (bring your own key), Phase 5: per-request keys through the whole request path. No network.

SECRET is a user key; it must reach only the provider it belongs to, and never a log, a cache key, an error
body, a background task or the shared circuit breaker.
"""
import json
import random

import httpx
import httpx2
import pytest
from fastapi.testclient import TestClient

from app.api.auth import Auth
from app.api.fake_models import FakeBehavior
from app.api.main import create_app
from app.providers import anthropic_provider, gemini_provider, openai_provider
from app.providers.base import send_request
from app.providers.common import ProviderError
from app.providers.keys import KeyContext
from app.registry import load_registry
from app.reliability import BYOKKeyMissing, ReliableClient, load_routing
from tests.api_helpers import API_KEY, ASK, FakePredictor, make_deps

SECRET = "sk-test-SECRET123"
REGISTRY = load_registry()
MESSAGES = [{"role": "user", "content": "Hi"}]


class Wire:
    def __init__(self, body, lib=httpx, ctype="application/json"):
        self.body, self.lib, self.ctype, self.requests = body, lib, ctype, []

    def __call__(self, request):
        self.requests.append(request)
        return self.lib.Response(200, content=json.dumps(self.body), headers={"content-type": self.ctype})

    def client(self):
        return self.lib.Client(transport=self.lib.MockTransport(self))


OPENAI_REPLY = {"id": "c", "object": "chat.completion", "created": 0, "model": "m",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
ANTHROPIC_REPLY = {"id": "m", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
                   "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "stop_sequence": None,
                   "usage": {"input_tokens": 1, "output_tokens": 1}}
GEMINI_REPLY = {"candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2}}


@pytest.fixture
def server_keys(monkeypatch):
    """Server env keys that must NOT be used when a user key is given."""
    for env in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(env, "server-env-key")


# ---------- a: each adapter sends the user's key ----------

@pytest.mark.parametrize("model, module, reply, lib, header, prefix", [
    ("openai-luna", openai_provider, OPENAI_REPLY, httpx, "authorization", "Bearer "),
    ("groq-small", openai_provider, OPENAI_REPLY, httpx, "authorization", "Bearer "),
    ("claude-haiku", anthropic_provider, ANTHROPIC_REPLY, httpx2, "x-api-key", ""),
    ("gemini-flash", gemini_provider, GEMINI_REPLY, httpx, "x-goog-api-key", ""),
])
def test_adapter_sends_the_given_key_not_the_env_key(server_keys, monkeypatch, model, module, reply, lib, header,
                                                     prefix):
    wire = Wire(reply, lib)
    monkeypatch.setattr(module, "HTTP_CLIENT", wire.client())
    r = send_request(MESSAGES, REGISTRY[model], max_tokens=100, api_key=SECRET)
    assert r.text == "ok"
    (request,) = wire.requests
    assert request.headers[header] == prefix + SECRET
    assert "server-env-key" not in str(request.headers)


@pytest.mark.parametrize("model, module, reply, lib, header, prefix", [
    ("groq-small", openai_provider, OPENAI_REPLY, httpx, "authorization", "Bearer "),
    ("claude-haiku", anthropic_provider, ANTHROPIC_REPLY, httpx2, "x-api-key", ""),
    ("gemini-flash", gemini_provider, GEMINI_REPLY, httpx, "x-goog-api-key", ""),
])
def test_without_a_key_the_adapter_uses_the_env_key_as_before(server_keys, monkeypatch, model, module, reply, lib,
                                                              header, prefix):
    wire = Wire(reply, lib)
    monkeypatch.setattr(module, "HTTP_CLIENT", wire.client())
    send_request(MESSAGES, REGISTRY[model], max_tokens=100)
    assert wire.requests[0].headers[header] == prefix + "server-env-key"


def test_an_empty_given_key_is_an_error_not_the_env_key(server_keys):
    with pytest.raises(ProviderError) as error:
        send_request(MESSAGES, REGISTRY["groq-small"], max_tokens=100, api_key="")
    assert error.value.kind == "config"


def test_ollama_ignores_a_key(monkeypatch):
    from app.providers import ollama_provider
    seen = {}
    monkeypatch.setattr(ollama_provider, "send_openai_format", lambda *a, **kw: seen.update(kw) or "ok")
    ollama_provider.send(MESSAGES, "llama3.2:3b", api_key=SECRET, stream=False, max_tokens=1, timeout=1)
    assert seen["api_key"] == "ollama"


# ---------- b: the KeyContext travels with the request ----------


GROQ_ONLY = {"x-groq-key": SECRET}


def app(script=None, predictor=None, **kw):
    deps = make_deps(script, predictor=predictor or FakePredictor(0.4, 0.95), **kw)
    return TestClient(create_app(deps=deps, auth=Auth([API_KEY]))), deps


def chat(http, body=None, headers=None):
    return http.post("/v1/chat/completions", json=body or {"model": "auto", "messages": ASK},
                     headers={"Authorization": f"Bearer {API_KEY}"} | (headers or {}))


@pytest.mark.parametrize("stream", [False, True])
def test_the_users_key_reaches_the_provider(stream):
    http, deps = app()
    r = chat(http, {"model": "auto", "messages": ASK, "stream": stream}, GROQ_ONLY)
    assert r.status_code == 200
    assert deps.fake_models.last_api_key == SECRET
    assert all(c["byok_key"] for c in deps.fake_models.calls)


def test_no_key_request_sends_no_api_key():
    http, deps = app()
    assert chat(http).status_code == 200
    assert deps.fake_models.last_api_key is None and not any(c["byok_key"] for c in deps.fake_models.calls)


def test_the_guard_refuses_a_byok_call_without_its_providers_key():
    deps = make_deps()
    with pytest.raises(BYOKKeyMissing):
        deps.client._call(REGISTRY["claude-opus"], ASK, False, {}, KeyContext("byok", {"groq": SECRET}))


def test_api_key_cant_be_passed_around_the_keys_context():
    deps = make_deps()
    with pytest.raises(ValueError, match="KeyContext"):
        deps.client.send(ASK, "low", api_key=SECRET)
    with pytest.raises(ValueError, match="not both"):
        deps.client.send(ASK, "low", keys=KeyContext("byok", {"groq": SECRET}), providers={"groq"})


def test_keys_are_not_stored_on_deps_or_the_app():
    http, deps = app()
    chat(http, headers=GROQ_ONLY)
    for obj in (deps, deps.client, deps.config, http.app.state):
        assert SECRET not in repr(vars(obj))


# ---------- c: BYOK routing, tier fallback, errors, baseline ----------

DOWN = FakeBehavior(fail_before="server")
QUALITY = {"low": "0.1", "medium": "0.92", "high": "0.99"}        # FakePredictor(0.4, 0.95) -> these tiers


def providers_called(deps):
    return {REGISTRY[c["model"]].provider for c in deps.fake_models.calls}


@pytest.mark.parametrize("tier", ["low", "medium", "high"])
def test_groq_only_byok_routes_only_to_groq_models(tier):
    http, deps = app()
    r = chat(http, headers=GROQ_ONLY | {"X-Triage-Quality": QUALITY[tier]})
    assert r.status_code == 200 and r.headers["x-triage-tier"] == tier
    assert providers_called(deps) == {"groq"}


def test_groq_only_byok_never_falls_back_to_the_local_ollama():
    http, deps = app({"groq-allam": DOWN})
    r = chat(http, headers=GROQ_ONLY | {"X-Triage-Quality": "0.1"})
    assert r.status_code == 503 and providers_called(deps) == {"groq"}


def test_gemini_only_byok_on_a_low_request_escalates_to_a_tier_with_a_gemini_model():
    http, deps = app()
    r = chat(http, headers={"x-gemini-key": SECRET, "X-Triage-Quality": "0.1"})
    assert r.status_code == 200
    assert (r.headers["x-triage-tier"], r.headers["x-triage-model"]) == ("medium", "gemini-flash")
    assert "no model for your keys in low: used medium" in r.headers["x-triage-reason"]
    (row,) = deps.log.rows()
    assert (row.tier, row.served_model, row.primary_model, row.fell_back) == ("medium", "gemini-flash",
                                                                               "gemini-flash", False)


def test_tier_fallback_goes_up_first_then_down():
    from app.api.core import usable_tier
    routing = load_routing(REGISTRY)
    assert usable_tier("low", routing, REGISTRY, {"gemini"}) == "medium"
    assert usable_tier("high", routing, REGISTRY, {"groq"}) == "high"
    assert usable_tier("low", routing, REGISTRY, {"openrouter"}) is None


def test_byok_with_no_reachable_model_is_a_400_naming_the_key_headers():
    from tests.api_helpers import routing_raw
    raw = routing_raw()
    raw["fallback_chains"] = {t: [c[0]] for t, c in raw["fallback_chains"].items()}   # groq models only
    http, deps = app(raw=raw)
    r = chat(http, headers={"x-gemini-key": SECRET})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "no_usable_model"
    assert all(h in err["message"] for h in ("X-OpenAI-Key", "X-Anthropic-Key", "X-Groq-Key", "X-Gemini-Key"))
    assert SECRET not in r.text and deps.fake_models.calls == []


@pytest.mark.parametrize("headers", [{}, GROQ_ONLY])
def test_named_model_without_its_key_is_403_naming_the_header(headers):
    http, deps = app()
    r = chat(http, {"model": "claude-opus", "messages": ASK}, headers)
    assert r.status_code == 403
    err = r.json()["error"]
    assert err["code"] == "model_not_allowed" and "X-Anthropic-Key" in err["message"]
    assert deps.fake_models.calls == []


def test_named_model_with_its_key_is_served():
    http, deps = app()
    r = chat(http, {"model": "claude-opus", "messages": ASK}, {"x-anthropic-key": SECRET})
    assert r.status_code == 200 and r.json()["model"] == "claude-opus"


def test_byok_named_local_model_is_403():
    http, _ = app()
    r = chat(http, {"model": "ollama-llama", "messages": ASK}, GROQ_ONLY)
    assert r.status_code == 403 and "doesn't use" in r.json()["error"]["message"]


@pytest.mark.parametrize("headers, strong", [({"x-anthropic-key": SECRET}, "claude-opus"),
                                             ({"x-gemini-key": SECRET}, "gemini-flash"),
                                             (GROQ_ONLY, "groq-large"),
                                             ({}, "groq-large")])
def test_savings_baseline_is_the_strong_model_among_the_users_providers(headers, strong):
    http, deps = app()
    assert chat(http, headers=headers).status_code == 200
    (row,) = deps.log.rows()
    assert row.strong_cost == pytest.approx(REGISTRY[strong].reported_cost(row.input_tokens, row.output_tokens))
    assert row.saved == pytest.approx(row.strong_cost - row.price_as_cost)
