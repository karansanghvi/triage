"""Reasoning-model adapters (Phase 5): OpenAI gpt-6.x and Claude 5.x. No network, no real keys.

Wire-level tests use the REAL openai / anthropic SDKs with an httpx.MockTransport, so they check the exact request
that would be sent (an SDK argument change, like Anthropic dropping `temperature`, fails here, not in production).
"""
import json

import httpx
import httpx2                    # the anthropic SDK is built on httpx2, the openai SDK on httpx
import pytest
from fastapi.testclient import TestClient

from app.api.auth import Auth
from app.api.fake_models import FakeBehavior
from app.api.main import create_app
from app.cache import make_cache_key
from app.providers import anthropic_provider, openai_provider
from app.providers.base import TruncatedAnswer, send_request
from app.providers.common import ProviderError
from app.registry import load_registry
from app.reliability import load_routing
from tests.api_helpers import API_KEY, ASK, FakePredictor, make_deps, routing_raw

REGISTRY = load_registry()
MESSAGES = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]


class Wire:
    """A mock transport that records requests and answers from a list of (status, body, content_type)."""

    def __init__(self, *replies):
        self.replies, self.requests, self.lib = list(replies), [], httpx

    def __call__(self, request):
        self.requests.append(request)
        status, body, ctype = self.replies.pop(0)
        content = body if isinstance(body, (bytes, str)) else json.dumps(body)
        return self.lib.Response(status, content=content, headers={"content-type": ctype})

    def client(self):
        return self.lib.Client(transport=self.lib.MockTransport(self))

    def body(self, i=0):
        return json.loads(self.requests[i].content)


def sse(*events, named=False):
    out = []
    for e in events:
        if named:
            out.append(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n")
        else:
            out.append(f"data: {json.dumps(e) if not isinstance(e, str) else e}\n\n")
    return "".join(out)


@pytest.fixture
def keys(monkeypatch):
    for env in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY"):
        monkeypatch.setenv(env, "test-key-not-real")


def use(monkeypatch, module, wire):
    wire.lib = httpx2 if module is anthropic_provider else httpx
    monkeypatch.setattr(module, "HTTP_CLIENT", wire.client())


# ---------- OpenAI ----------

def completion(content="Hello", finish="stop", prompt=12, completion_tokens=40, model="gpt-6.1-sol"):
    return {"id": "c1", "object": "chat.completion", "created": 0, "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion_tokens,
                      "total_tokens": prompt + completion_tokens}}


def chunk(content=None, finish=None, usage=None):
    choices = [] if usage else [{"index": 0, "delta": {"content": content} if content is not None else {},
                                 "finish_reason": finish}]
    out = {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "gpt-6.1-sol", "choices": choices}
    if usage:
        out["usage"] = usage
    return out


def test_openai_reasoning_model_gets_max_completion_tokens_and_no_temperature(keys, monkeypatch):
    wire = Wire((200, completion(), "application/json"))
    use(monkeypatch, openai_provider, wire)
    r = send_request(MESSAGES, REGISTRY["openai-sol"], temperature=0, max_tokens=16000, reasoning_effort="low")
    body = wire.body()
    assert body["max_completion_tokens"] == 16000 and body["reasoning_effort"] == "low"
    assert "max_tokens" not in body and "temperature" not in body and "top_p" not in body
    assert body["model"] == "gpt-6.1-sol" and wire.requests[0].url.path == "/v1/chat/completions"
    assert (r.text, r.input_tokens, r.output_tokens) == ("Hello", 12, 40)
    assert r.cost == pytest.approx((12 * 2 + 40 * 10) / 1e6)


def test_openai_stream_pieces_usage_and_finish(keys, monkeypatch):
    usage = {"prompt_tokens": 5, "completion_tokens": 30, "total_tokens": 35,
             "completion_tokens_details": {"reasoning_tokens": 20}}
    wire = Wire((200, sse(chunk("Hel"), chunk("lo"), chunk(finish="stop"), chunk(usage=usage), "[DONE]"),
                 "text/event-stream"))
    use(monkeypatch, openai_provider, wire)
    stream = send_request(MESSAGES, REGISTRY["openai-luna"], stream=True, max_tokens=16000)
    assert list(stream) == ["Hel", "lo"]
    r = stream.response
    assert (r.text, r.finish_reason, r.output_tokens, r.reasoning_tokens) == ("Hello", "stop", 30, 20)
    assert "temperature" not in wire.body() and wire.body()["max_completion_tokens"] == 16000


def test_openai_empty_answer_at_the_limit_is_truncated_and_keeps_its_cost(keys, monkeypatch):
    wire = Wire((200, completion(content="", finish="length", completion_tokens=16000), "application/json"))
    use(monkeypatch, openai_provider, wire)
    with pytest.raises(TruncatedAnswer) as error:
        send_request(MESSAGES, REGISTRY["openai-sol"], max_tokens=16000)
    e = error.value
    assert e.kind == "truncated" and not e.retryable and e.max_tokens == 16000
    assert e.response.output_tokens == 16000 and e.response.cost == pytest.approx((12 * 2 + 16000 * 10) / 1e6)


def test_openai_empty_streamed_answer_at_the_limit_is_truncated(keys, monkeypatch):
    usage = {"prompt_tokens": 5, "completion_tokens": 16000, "total_tokens": 16005}
    wire = Wire((200, sse(chunk(finish="length"), chunk(usage=usage), "[DONE]"), "text/event-stream"))
    use(monkeypatch, openai_provider, wire)
    stream = send_request(MESSAGES, REGISTRY["openai-sol"], stream=True, max_tokens=16000)
    with pytest.raises(TruncatedAnswer) as error:
        list(stream)
    assert error.value.response.output_tokens == 16000


def test_openai_cut_off_answer_with_text_is_still_served(keys, monkeypatch):
    wire = Wire((200, completion(content="Partial", finish="length"), "application/json"))
    use(monkeypatch, openai_provider, wire)
    r = send_request(MESSAGES, REGISTRY["openai-sol"], max_tokens=16000)
    assert (r.text, r.finish_reason) == ("Partial", "length")


# ---------- Groq: requests without keys must not change ----------

def test_groq_request_bytes_are_unchanged(keys, monkeypatch):
    """Same bytes as calling the SDK the way the adapter did before Phase 5."""
    wire = Wire((200, completion(model="openai/gpt-oss-20b"), "application/json"),
                (200, completion(model="openai/gpt-oss-20b"), "application/json"))
    use(monkeypatch, openai_provider, wire)
    send_request(MESSAGES, REGISTRY["groq-small"], temperature=0.0, max_tokens=4000, reasoning_effort="low")
    from openai import OpenAI
    OpenAI(base_url="https://api.groq.com/openai/v1", api_key="test-key-not-real", max_retries=0,
           http_client=wire.client()).chat.completions.create(
        model="openai/gpt-oss-20b", messages=MESSAGES, temperature=0.0, max_tokens=4000, reasoning_effort="low")
    assert wire.requests[0].content == wire.requests[1].content
    assert wire.body()["max_tokens"] == 4000 and wire.body()["temperature"] == 0.0


def test_groq_empty_answer_at_the_limit_is_served_as_before(keys, monkeypatch):
    wire = Wire((200, completion(content="", finish="length", model="openai/gpt-oss-20b"), "application/json"))
    use(monkeypatch, openai_provider, wire)
    r = send_request(MESSAGES, REGISTRY["groq-small"], max_tokens=4000)    # not a `reasoning` model: no error
    assert r.text == "" and r.finish_reason == "length"


def test_groq_stream_reports_no_finish_reason_as_before(keys, monkeypatch):
    wire = Wire((200, sse(chunk("ok"), chunk(finish="length"),
                          chunk(usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}), "[DONE]"),
                 "text/event-stream"))
    use(monkeypatch, openai_provider, wire)
    stream = send_request(MESSAGES, REGISTRY["groq-small"], stream=True)
    assert list(stream) == ["ok"] and stream.response.finish_reason is None


# ---------- Anthropic ----------

def message(content, stop="end_turn", input_tokens=10, output_tokens=50, model="claude-opus-5-5"):
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": model, "content": content,
            "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}}


THINKING = {"type": "thinking", "thinking": "", "signature": "c2lnbmF0dXJl"}


def test_anthropic_request_has_no_sampling_params_and_pins_the_version(keys, monkeypatch):
    wire = Wire((200, message([THINKING, {"type": "text", "text": "Hello"}]), "application/json"))
    use(monkeypatch, anthropic_provider, wire)
    r = send_request(MESSAGES, REGISTRY["claude-opus"], temperature=0, max_tokens=16000, reasoning_effort="low")
    req, body = wire.requests[0], wire.body()
    assert req.headers["anthropic-version"] == "2026-10-21" and req.url.path == "/v1/messages"
    assert not {"temperature", "top_p", "top_k"} & set(body)
    assert body["output_config"] == {"effort": "low"} and body["max_tokens"] == 16000
    assert body["system"] == "Be brief." and body["messages"] == [{"role": "user", "content": "Hi"}]
    assert "thinking" not in body
    assert (r.text, r.finish_reason, r.output_tokens) == ("Hello", "end_turn", 50)   # thinking block left out


def test_anthropic_haiku_thinking_disabled_is_sent(keys, monkeypatch):
    wire = Wire((200, message([{"type": "text", "text": "Hi"}], model="claude-haiku-5-5"), "application/json"))
    use(monkeypatch, anthropic_provider, wire)
    send_request(MESSAGES, REGISTRY["claude-haiku"], max_tokens=16000, reasoning_effort="low", thinking="disabled")
    assert wire.body()["thinking"] == {"type": "disabled"} and wire.body()["output_config"] == {"effort": "low"}


def test_anthropic_stream_yields_text_only_then_usage_and_stop_reason(keys, monkeypatch):
    events = [
        {"type": "message_start", "message": {**message([], stop=None, output_tokens=1), "content": []}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "c2ln"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Hel"}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "lo"}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
         "usage": {"output_tokens": 42}},
        {"type": "message_stop"},
    ]
    wire = Wire((200, sse(*events, named=True), "text/event-stream"))
    use(monkeypatch, anthropic_provider, wire)
    stream = send_request(MESSAGES, REGISTRY["claude-sonnet"], stream=True, max_tokens=16000, reasoning_effort="low")
    assert list(stream) == ["Hel", "lo"]
    r = stream.response
    assert (r.text, r.finish_reason, r.input_tokens, r.output_tokens) == ("Hello", "end_turn", 10, 42)
    assert wire.requests[0].headers["anthropic-version"] == "2026-10-21" and wire.body()["stream"] is True


def test_anthropic_thinking_only_answer_at_the_limit_is_truncated(keys, monkeypatch):
    wire = Wire((200, message([THINKING], stop="max_tokens", output_tokens=16000), "application/json"))
    use(monkeypatch, anthropic_provider, wire)
    with pytest.raises(TruncatedAnswer) as error:
        send_request(MESSAGES, REGISTRY["claude-opus"], max_tokens=16000)
    assert error.value.response.cost == pytest.approx((10 * 4 + 16000 * 20) / 1e6)


def test_anthropic_api_errors_keep_their_kind(keys, monkeypatch):
    wire = Wire((400, {"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}},
                 "application/json"))
    use(monkeypatch, anthropic_provider, wire)
    with pytest.raises(ProviderError) as error:
        send_request(MESSAGES, REGISTRY["claude-opus"], max_tokens=100)
    assert (error.value.kind, error.value.status_code) == ("client", 400)


@pytest.mark.parametrize("stream", [False, True])
def test_unexpected_sdk_errors_become_client_errors_named_by_type_only(keys, monkeypatch, stream):
    class Boom:
        def __init__(self, **kw):
            self.messages = self

        def create(self, **kw):
            raise TypeError("unexpected argument; key sk-ant-SECRET")

        def stream(self, **kw):
            raise TypeError("unexpected argument; key sk-ant-SECRET")

    monkeypatch.setattr(anthropic_provider.anthropic, "Anthropic", Boom)
    with pytest.raises(ProviderError) as error:
        result = send_request(MESSAGES, REGISTRY["claude-opus"], stream=stream, max_tokens=100)
        list(result)
    assert error.value.kind == "client" and "TypeError" in str(error.value) and "SECRET" not in str(error.value)
    assert error.value.__cause__ is None and error.value.__suppress_context__


# ---------- routing.yaml: effort, thinking, the max_tokens floor ----------

def test_floor_raises_only_our_default_and_effort_reaches_every_reasoning_model():
    deps = make_deps()
    for tier, model in (("low", "claude-haiku"), ("medium", "claude-sonnet"), ("high", "claude-opus")):
        deps.client.send(ASK, tier, providers={"anthropic"})
    deps.client.send(ASK, "high", providers={"openai"})
    calls = {c["model"]: c for c in deps.fake_models.calls}
    for name in ("claude-haiku", "claude-sonnet", "claude-opus", "openai-sol"):
        assert calls[name]["max_tokens"] == 16000 and calls[name]["reasoning_effort"] == "low", name
    assert calls["claude-haiku"]["thinking"] == "disabled" and calls["claude-opus"]["thinking"] is None


def test_a_callers_own_max_tokens_is_respected():
    deps = make_deps()
    deps.client.send(ASK, "high", providers={"anthropic"}, max_tokens=500)
    assert deps.fake_models.calls[-1]["max_tokens"] == 500


def test_server_models_keep_the_phase2_settings():
    deps = make_deps()
    for tier in ("low", "medium", "high"):
        deps.client.send(ASK, tier)
    for c in deps.fake_models.calls:
        assert c["max_tokens"] == 4000 and c["temperature"] == 0.0 and c["thinking"] is None, c


def write_routing(tmp_path, model_params="", extra=""):
    path = tmp_path / "routing.yaml"
    path.write_text("fallback_chains:\n  low: [groq-allam, claude-haiku]\n  medium: [groq-small]\n"
                    "  high: [groq-large]\nserver_providers: [groq]\n" + model_params + extra)
    return path


@pytest.mark.parametrize("params, message", [
    ("model_params:\n  groq-small: {thinking: disabled}\n", "does not accept thinking"),
    ("model_params:\n  claude-haiku: {thinking: off}\n", "thinking must be one of"),
    ("model_params:\n  claude-haiku: {thinking: disabled, reasoning_effort: xhigh}\n", "400 at effort xhigh"),
    ("min_default_max_tokens: {nope: 100}\n", "unknown model"),
    ("min_default_max_tokens: {claude-haiku: 0}\n", "above 0"),
])
def test_bad_reasoning_settings_are_rejected(tmp_path, params, message):
    with pytest.raises(ValueError, match=message):
        load_routing(REGISTRY, write_routing(tmp_path, params))


def test_real_config_gives_all_six_reasoning_models_low_effort_and_the_floor():
    routing = load_routing(REGISTRY)
    reasoning = {n for n, m in REGISTRY.items() if m.reasoning}
    assert reasoning == {"openai-luna", "openai-terra", "openai-sol", "claude-haiku", "claude-sonnet", "claude-opus"}
    for name in reasoning:
        assert routing.model_params[name]["reasoning_effort"] == "low"
        assert routing.min_default_max_tokens[name] == 16000
    assert routing.model_params["claude-haiku"]["thinking"] == "disabled"
    assert not set(routing.min_default_max_tokens) - reasoning


def test_thinking_is_part_of_the_cache_key():
    m = REGISTRY["claude-haiku"]
    assert make_cache_key(ASK, m, 0, 16000, "low", None, "disabled") != make_cache_key(ASK, m, 0, 16000, "low")
    assert make_cache_key(ASK, m, 0, 16000, "low", None, None) == make_cache_key(ASK, m, 0, 16000, "low")


def test_registry_reasoning_must_be_a_bool(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text("models:\n  - {name: a, provider: openai, model_id: a, tier: low, input_price: 1, "
                    "output_price: 1, reasoning: yes-please}\n")
    with pytest.raises(ValueError, match="true or false"):
        load_registry(path)


# ---------- truncated answers: fall back, and count their cost ----------

TRUNC = FakeBehavior(truncated=True)
DOWN = FakeBehavior(fail_before="server")


def test_truncated_answer_falls_back_without_retrying_or_tripping_the_breaker():
    deps = make_deps({"openai-sol": TRUNC})
    result = deps.client.send(ASK, "high", providers={"openai", "anthropic"})
    assert result.model.name == "claude-opus"
    first = result.attempts[0]
    assert (first.model_name, first.error_kind) == ("openai-sol", "truncated") and first.response is not None
    assert [c["model"] for c in deps.fake_models.calls] == ["openai-sol", "claude-opus"]     # one try, no retry
    assert not deps.client.breaker.is_open("openai") and not deps.client.breaker.is_open("model:openai-sol")


def byok_style_app(script, predictor=None):
    """Until BYOK wiring exists, let the server itself use openai and anthropic (test-only config)."""
    raw = routing_raw()
    raw["server_providers"] = ["groq", "gemini", "ollama", "openai", "anthropic"]
    deps = make_deps(script, raw=raw, predictor=predictor or FakePredictor(0.1, 0.1))
    return TestClient(create_app(deps=deps, auth=Auth([API_KEY]))), deps


def chat(http, body):
    return http.post("/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {API_KEY}"})


@pytest.mark.parametrize("stream", [False, True])
def test_api_counts_the_truncated_attempts_cost_in_the_total(stream):
    http, deps = byok_style_app({"groq-large": DOWN, "gemini-flash": DOWN, "openai-sol": TRUNC})
    r = chat(http, {"model": "auto", "messages": ASK, "stream": stream})
    assert r.status_code == 200
    assert r.headers["x-triage-model"] == "claude-opus"
    (row,) = deps.log.rows()
    truncated = next(a for a in row.attempts if a["error_kind"] == "truncated")
    assert truncated["model"] == "openai-sol" and truncated["output_tokens"] == 16000
    sol = REGISTRY["openai-sol"]
    assert truncated["cost"] == pytest.approx(sol.actual_cost(truncated["input_tokens"], 16000))
    opus = REGISTRY["claude-opus"].actual_cost(row.input_tokens, row.output_tokens)
    assert row.cost == pytest.approx(opus + truncated["cost"])
    assert row.price_as_cost == pytest.approx(opus + truncated["price_as_cost"])
    assert row.saved == pytest.approx(row.strong_cost - row.price_as_cost)
    if not stream:
        assert float(r.headers["x-triage-cost"]) == pytest.approx(row.price_as_cost, abs=1e-8)


def test_api_named_model_truncated_is_a_400_that_suggests_more_max_tokens():
    http, deps = byok_style_app({"openai-sol": TRUNC})
    r = chat(http, {"model": "openai-sol", "messages": ASK})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "max_tokens_too_low" and err["param"] == "max_tokens"
    assert "16000" in err["message"] and "higher max_tokens" in err["message"]
    (row,) = deps.log.rows()
    assert (row.status, row.http_status, row.error_kind) == ("error", 400, "truncated")
    assert row.cost == pytest.approx(REGISTRY["openai-sol"].actual_cost(row.attempts[0]["input_tokens"], 16000))


def test_api_auto_where_every_model_truncates_is_503_with_the_cost_logged():
    http, deps = byok_style_app({"groq-large": DOWN, "gemini-flash": DOWN, "openai-sol": TRUNC, "claude-opus": TRUNC})
    r = chat(http, {"model": "auto", "messages": ASK})
    assert r.status_code == 503
    (row,) = deps.log.rows()
    assert row.cost > 0 and [a["error_kind"] for a in row.attempts][-2:] == ["truncated", "truncated"]


def test_api_503_without_truncation_logs_as_before():
    http, deps = byok_style_app({"groq-large": DOWN, "gemini-flash": DOWN, "openai-sol": DOWN, "claude-opus": DOWN})
    chat(http, {"model": "auto", "messages": ASK})
    (row,) = deps.log.rows()
    assert row.attempts == [] and row.cost is None
