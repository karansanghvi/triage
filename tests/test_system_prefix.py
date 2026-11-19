"""Per-model system_prefix (routing.yaml model_params): groq-allam gets a reply-language instruction at the start of
its system message; every other model's request is unchanged. Mock transports and fake models; no network."""
import copy
import json

import httpx
import httpx2
import pytest
from fastapi.testclient import TestClient

from app.api.auth import Auth
from app.api.main import create_app
from app.cache import CachedSender, ResponseCache, make_cache_key
from app.providers import anthropic_provider, gemini_provider, openai_provider
from app.providers.base import send_request, with_system_prefix
from app.registry import load_registry
from app.reliability import ReliableClient, load_routing, routing_from_dict
from tests.api_helpers import API_KEY, FakePredictor, make_deps, routing_raw

REGISTRY = load_registry()
ROUTING = load_routing(REGISTRY)
PREFIX = "Always reply in the same language as the user's latest message."
HELLO = [{"role": "user", "content": "hello there, how do I sort a list in Python?"}]
WITH_SYSTEM = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Explain recursion."}]

OPENAI_REPLY = {"id": "c", "object": "chat.completion", "created": 0, "model": "m",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
OPENAI_STREAM = (b'data: {"id":"c","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,'
                 b'"delta":{"content":"ok"},"finish_reason":null}]}\n\ndata: [DONE]\n\n')
ANTHROPIC_REPLY = {"id": "m", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
                   "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "stop_sequence": None,
                   "usage": {"input_tokens": 1, "output_tokens": 1}}
GEMINI_REPLY = {"candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2}}


@pytest.fixture(autouse=True)
def env_keys(monkeypatch):
    for env in ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(env, "server-env-key")


class Wire:
    def __init__(self, body, lib=httpx, sse=None):
        self.body, self.lib, self.sse, self.requests = body, lib, sse, []

    def __call__(self, request):
        self.requests.append(request)
        if self.sse is not None and json.loads(request.content).get("stream"):
            return self.lib.Response(200, content=self.sse, headers={"content-type": "text/event-stream"})
        return self.lib.Response(200, content=json.dumps(self.body), headers={"content-type": "application/json"})

    def client(self):
        return self.lib.Client(transport=self.lib.MockTransport(self))

    def body_of(self, i=0):
        return json.loads(self.requests[i].content)


def openai_wire(monkeypatch):
    wire = Wire(OPENAI_REPLY, sse=OPENAI_STREAM)
    monkeypatch.setattr(openai_provider, "HTTP_CLIENT", wire.client())
    return wire


def client():
    return ReliableClient(REGISTRY, ROUTING, sleep=lambda s: None)


def call(model, messages, stream=False):
    result = client().send_model(messages, model, stream=stream)
    if stream:
        list(result.response)
    return result


# ---------- the merge ----------

def test_without_a_system_message_the_prefix_becomes_the_first_message():
    assert with_system_prefix(HELLO, PREFIX) == [{"role": "system", "content": PREFIX}] + HELLO


def test_an_existing_system_message_is_kept_after_the_prefix():
    merged = with_system_prefix(WITH_SYSTEM, PREFIX)
    assert merged == [{"role": "system", "content": f"{PREFIX}\n\nYou are terse."}, WITH_SYSTEM[1]]


def test_only_the_first_system_message_is_merged_and_the_input_is_not_changed():
    messages = [{"role": "user", "content": "a"}, {"role": "system", "content": "one"},
                {"role": "system", "content": "two"}]
    before = copy.deepcopy(messages)
    merged = with_system_prefix(messages, PREFIX)
    assert [m["content"] for m in merged] == ["a", f"{PREFIX}\n\none", "two"]
    assert messages == before


def test_an_empty_system_message_becomes_just_the_prefix():
    assert with_system_prefix([{"role": "system", "content": "  "}, HELLO[0]], PREFIX)[0]["content"] == PREFIX


# ---------- groq-allam's requests ----------

def test_routing_yaml_sets_the_prefix_for_groq_allam_only():
    with_prefix = {name for name, params in ROUTING.model_params.items() if "system_prefix" in params}
    assert with_prefix == {"groq-allam"}
    assert ROUTING.model_params["groq-allam"]["system_prefix"] == PREFIX


@pytest.mark.parametrize("stream", [False, True])
def test_groq_allam_requests_start_with_the_prefix(monkeypatch, stream):
    wire = openai_wire(monkeypatch)
    call("groq-allam", HELLO, stream)
    assert wire.body_of()["messages"] == [{"role": "system", "content": PREFIX}] + HELLO


@pytest.mark.parametrize("stream", [False, True])
def test_groq_allam_merges_the_prefix_into_an_existing_system_message(monkeypatch, stream):
    wire = openai_wire(monkeypatch)
    call("groq-allam", WITH_SYSTEM, stream)
    sent = wire.body_of()["messages"]
    assert sent == [{"role": "system", "content": f"{PREFIX}\n\nYou are terse."}, WITH_SYSTEM[1]]
    assert sum(m["role"] == "system" for m in sent) == 1


def test_send_request_alone_adds_nothing_without_the_param(monkeypatch):
    """eval/ and scripts/ call send_request directly: without system_prefix, the messages go out as given."""
    wire = openai_wire(monkeypatch)
    send_request(HELLO, REGISTRY["groq-allam"], max_tokens=100)
    assert wire.body_of()["messages"] == HELLO


# ---------- other models: unchanged ----------

@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("model", ["groq-small", "groq-large", "openai-luna"])
def test_openai_format_models_send_the_messages_exactly_as_given(monkeypatch, model, stream):
    from app.providers.keys import KeyContext
    keys = KeyContext("byok", {"openai": "sk-test"}) if model.startswith("openai") else None
    wire = openai_wire(monkeypatch)
    for messages in (HELLO, WITH_SYSTEM):
        wire.requests.clear()
        result = client().send_model(messages, model, stream=stream, keys=keys)
        if stream:
            list(result.response)
        assert wire.body_of()["messages"] == messages


def test_anthropic_and_gemini_requests_are_unchanged(monkeypatch):
    from app.providers.keys import KeyContext
    anthropic_wire = Wire(ANTHROPIC_REPLY, lib=httpx2)
    gemini_wire = Wire(GEMINI_REPLY)
    monkeypatch.setattr(anthropic_provider, "HTTP_CLIENT", anthropic_wire.client())
    monkeypatch.setattr(gemini_provider, "HTTP_CLIENT", gemini_wire.client())
    client().send_model(HELLO, "claude-haiku", keys=KeyContext("byok", {"anthropic": "sk-test"}))
    client().send_model(HELLO, "gemini-flash-lite")
    assert "system" not in anthropic_wire.body_of()
    assert "systemInstruction" not in gemini_wire.body_of()


# ---------- the cache key ----------

def test_the_prefix_is_part_of_the_cache_key():
    model = REGISTRY["groq-allam"]
    plain = make_cache_key(HELLO, model, 0.0, 100)
    assert make_cache_key(HELLO, model, 0.0, 100, system_prefix=PREFIX) != plain
    assert make_cache_key(HELLO, model, 0.0, 100, system_prefix=None) == plain          # other keys unchanged
    assert make_cache_key(HELLO, model, 0.0, 100, system_prefix="Reply in French.") != \
        make_cache_key(HELLO, model, 0.0, 100, system_prefix=PREFIX)


def test_a_cached_answer_without_the_prefix_isnt_reused_with_it(tmp_path):
    sent = []

    def send(messages, model, stream=False, **settings):
        sent.append(settings.get("system_prefix"))
        from app.providers.base import _build_response
        from app.providers.common import Usage
        return _build_response(model, "ok", Usage(1, 1), messages, 1.0, "stop")

    cached = CachedSender(ResponseCache(tmp_path / "c.sqlite3"), send)
    model = REGISTRY["groq-allam"]
    cached(HELLO, model, temperature=0, max_tokens=100)
    cached(HELLO, model, temperature=0, max_tokens=100, system_prefix=PREFIX)
    cached(HELLO, model, temperature=0, max_tokens=100, system_prefix=PREFIX)       # this one is a hit
    assert sent == [None, PREFIX]


# ---------- startup validation ----------

@pytest.mark.parametrize("params, message", [
    ({"groq-allam": {"system_prefix": ""}}, "non-empty string"),
    ({"groq-allam": {"system_prefix": "   "}}, "non-empty string"),
    ({"groq-allam": {"system_prefix": 42}}, "non-empty string"),
    ({"no-such-model": {"system_prefix": "x"}}, "unknown model 'no-such-model'"),
])
def test_bad_system_prefixes_are_refused(params, message):
    raw = copy.deepcopy(routing_raw())
    raw["model_params"] = {**raw.get("model_params", {}), **params}
    with pytest.raises(ValueError, match=message):
        routing_from_dict(raw, REGISTRY)


def test_a_caller_cant_set_it_per_request():
    with pytest.raises(ValueError, match="set per model"):
        client().send(HELLO, "low", system_prefix="Reply in French.")


# ---------- through the API ----------

def test_api_requests_routed_to_groq_allam_get_the_prefix_and_others_dont():
    deps = make_deps(predictor=FakePredictor(0.95, 0.97))      # low
    http = TestClient(create_app(deps=deps, auth=Auth([API_KEY])))
    for model in ("auto", "groq-small"):
        http.post("/v1/chat/completions", json={"model": model, "messages": HELLO},
                  headers={"Authorization": f"Bearer {API_KEY}"})
    answers = [(c["model"], c["system_prefix"]) for c in deps.fake_models.calls if c["kind"] == "answer"]
    assert answers == [("groq-allam", PREFIX), ("groq-small", None)]
