"""Timeouts, deadlines and overloads for every provider adapter (Groq, OpenAI, Anthropic, Gemini).

The Gemini 504 "Deadline expired" bug (b42d925) was a timeout classified as a retryable server error, so the same
model was tried three times. These tests make sure no provider can do that:
- every way an SDK reports a timeout maps to "timeout": the client's own timeout (a real slow local server), a
  read timeout in the middle of a stream, 408 / 504, Anthropic's timeout_error (any status, or an SSE event);
- overloaded and 5xx errors keep the right kind;
- a timeout is never retried on the same model and moves straight to the next one, streaming or not;
- the API says provider_timeout only when every model tried timed out.
No network: httpx MockTransport, and a local server on 127.0.0.1 for the slow responses.
"""
import dataclasses
import json
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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
from app.reliability import AllModelsFailed, ReliableClient, load_routing
from tests.api_helpers import API_KEY, ASK, FakePredictor, make_deps

REGISTRY = load_registry()
MESSAGES = [{"role": "user", "content": "Hi"}]
SECRET = "sk-test-SECRET123"

# One model per provider, its adapter module, and the HTTP library its SDK uses
PROVIDERS = {
    "groq": ("groq-small", openai_provider, httpx),
    "openai": ("openai-terra", openai_provider, httpx),
    "anthropic": ("claude-sonnet", anthropic_provider, httpx2),
    "gemini": ("gemini-flash", gemini_provider, httpx),
}

# A successful answer in each provider's format (non-streaming, and the SSE body for streaming)
REPLY = {
    "openai": {"id": "c", "object": "chat.completion", "created": 0, "model": "m",
               "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
    "anthropic": {"id": "m", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
                  "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "stop_sequence": None,
                  "usage": {"input_tokens": 1, "output_tokens": 1}},
    "gemini": {"candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
               "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2}},
}
FIRST_SSE = {
    "openai": b'data: {"id":"c","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,'
              b'"delta":{"content":"Hi"},"finish_reason":null}]}\n\n',
    "anthropic": b'event: message_start\ndata: {"type":"message_start","message":{"id":"m","type":"message",'
                 b'"role":"assistant","model":"x","content":[],"stop_reason":null,"stop_sequence":null,"usage":'
                 b'{"input_tokens":1,"output_tokens":1}}}\n\nevent: content_block_start\ndata: {"type":'
                 b'"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
                 b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":'
                 b'{"type":"text_delta","text":"Hi"}}\n\n',
    "gemini": b'data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Hi"}]}}]}\n\n',
}


def fmt(provider):
    return "openai" if provider in ("groq", "openai") else provider


@pytest.fixture(autouse=True)
def env_keys(monkeypatch):
    for env in ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(env, "server-env-key")


class Wire:
    """A MockTransport handler: answers with `respond(request)` and keeps every request."""

    def __init__(self, lib, respond):
        self.lib, self.respond, self.requests = lib, respond, []

    def __call__(self, request):
        self.requests.append(request)
        return self.respond(self.lib, request)

    def client(self):
        return self.lib.Client(transport=self.lib.MockTransport(self))


def status(code, body):
    def respond(lib, request):
        return lib.Response(code, content=json.dumps(body).encode(), headers={"content-type": "application/json"})
    return respond


def times_out(lib, request):
    raise lib.ReadTimeout("read timed out", request=request)


def times_out_mid_stream(provider):
    def respond(lib, request):
        class Body(lib.SyncByteStream):
            def __iter__(self):
                yield FIRST_SSE[fmt(provider)]
                raise lib.ReadTimeout("read timed out")
        return lib.Response(200, stream=Body(), headers={"content-type": "text/event-stream"})
    return respond


# The end of a successful stream, after FIRST_SSE
LAST_SSE = {
    "openai": b"data: [DONE]\n\n",
    "anthropic": b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\nevent: message_delta\n'
                 b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":'
                 b'{"output_tokens":1}}\n\nevent: message_stop\ndata: {"type":"message_stop"}\n\n',
    "gemini": b"",
}


def ok(provider, stream=False):
    def respond(lib, request):
        if stream:
            return lib.Response(200, content=FIRST_SSE[fmt(provider)] + LAST_SSE[fmt(provider)],
                                headers={"content-type": "text/event-stream"})
        return lib.Response(200, content=json.dumps(REPLY[fmt(provider)]).encode(),
                            headers={"content-type": "application/json"})
    return respond


def use(monkeypatch, provider, respond):
    _, module, lib = PROVIDERS[provider]
    wire = Wire(lib, respond)
    monkeypatch.setattr(module, "HTTP_CLIENT", wire.client())
    return wire


def kind_of(provider, stream=False, timeout=30):
    model = PROVIDERS[provider][0]
    with pytest.raises(ProviderError) as info:
        result = send_request(MESSAGES, REGISTRY[model], stream=stream, max_tokens=100, timeout=timeout,
                              api_key=SECRET)
        if stream:
            list(result)
    return info.value.kind


ALL = list(PROVIDERS)
BOTH = [False, True]


# ---------- 1. every timeout shape is "timeout" ----------

class SlowHandler(BaseHTTPRequestHandler):
    delay = 2.0

    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length", 0)))
        time.sleep(self.delay)                       # longer than the client's timeout
        try:
            self.send_response(200)
            self.end_headers()
        except OSError:
            pass                                     # the client gave up first, as intended

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def slow_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), SlowHandler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1]
    server.shutdown()


def local_client(lib, port):
    """A real HTTP client (real timeouts) whose requests all go to the local slow server."""
    class ToLocal(lib.HTTPTransport):
        def handle_request(self, request):
            request.url = request.url.copy_with(scheme="http", host="127.0.0.1", port=port)
            return super().handle_request(request)
    return lib.Client(transport=ToLocal())


@pytest.mark.parametrize("stream", BOTH)
@pytest.mark.parametrize("provider", ALL)
def test_a_slow_response_past_the_timeout_is_a_timeout(monkeypatch, slow_server, provider, stream):
    _, module, lib = PROVIDERS[provider]
    monkeypatch.setattr(module, "HTTP_CLIENT", local_client(lib, slow_server))
    started = time.perf_counter()
    assert kind_of(provider, stream, timeout=0.3) == "timeout"
    assert time.perf_counter() - started < 1.5       # gave up at the timeout, not after the slow answer


@pytest.mark.parametrize("stream", BOTH)
@pytest.mark.parametrize("provider", ALL)
def test_the_clients_own_timeout_exception_is_a_timeout(monkeypatch, provider, stream):
    use(monkeypatch, provider, times_out)
    assert kind_of(provider, stream) == "timeout"


@pytest.mark.parametrize("provider", ALL)
def test_a_read_timeout_in_the_middle_of_a_stream_is_a_timeout(monkeypatch, provider):
    """The SDKs don't all wrap this one (anthropic hands over httpx2's own ReadTimeout)."""
    use(monkeypatch, provider, times_out_mid_stream(provider))
    assert kind_of(provider, stream=True) == "timeout"


DEADLINES = {
    "groq": [(504, {"error": {"message": "Gateway Timeout"}}), (408, {"error": {"message": "Request Timeout"}})],
    "openai": [(504, {"error": {"message": "Gateway Timeout"}}), (408, {"error": {"message": "Request Timeout"}})],
    "anthropic": [
        (504, {"type": "error", "error": {"type": "timeout_error", "message": "Request timed out"}}),
        # "The API also returns timeout_error with other status codes, such as a 500" (Anthropic's error docs)
        (500, {"type": "error", "error": {"type": "timeout_error", "message": "Request timed out"}}),
        (408, {"type": "error", "error": {"type": "invalid_request_error", "message": "Request Timeout"}}),
    ],
    "gemini": [(504, {"error": {"code": 504, "message": "Deadline expired before operation could complete.",
                                "status": "DEADLINE_EXCEEDED"}}),
               (408, {"error": {"code": 408, "message": "Request Timeout"}})],
}


@pytest.mark.parametrize("stream", BOTH)
@pytest.mark.parametrize("provider, code, body", [(p, c, b) for p, cases in DEADLINES.items() for c, b in cases])
def test_deadline_statuses_are_timeouts(monkeypatch, provider, code, body, stream):
    use(monkeypatch, provider, status(code, body))
    assert kind_of(provider, stream) == "timeout"


def test_anthropics_timeout_error_event_inside_a_stream_is_a_timeout(monkeypatch):
    body = FIRST_SSE["anthropic"] + (b'event: error\ndata: {"type":"error","error":{"type":"timeout_error",'
                                     b'"message":"timed out"}}\n\n')
    use(monkeypatch, "anthropic", lambda lib, r: lib.Response(200, content=body,
                                                              headers={"content-type": "text/event-stream"}))
    assert kind_of("anthropic", stream=True) == "timeout"


# ---------- overloaded and 5xx keep the right kind ----------

BUSY_OR_BROKEN = [
    ("groq", 500, {"error": {"message": "boom"}}, "server"),
    ("groq", 502, {"error": {"message": "bad gateway"}}, "server"),
    ("groq", 503, {"error": {"message": "over capacity"}}, "server"),     # kept as before (d0e5cc3)
    ("groq", 498, {"error": {"message": "flex tier at capacity"}}, "overloaded"),
    ("openai", 500, {"error": {"message": "boom"}}, "server"),
    ("openai", 503, {"error": {"message": "overloaded", "type": "service_unavailable_error",
                               "code": "server_is_overloaded"}}, "overloaded"),
    ("anthropic", 500, {"type": "error", "error": {"type": "api_error", "message": "boom"}}, "server"),
    ("anthropic", 529, {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
     "overloaded"),
    ("gemini", 500, {"error": {"code": 500, "message": "boom", "status": "INTERNAL"}}, "server"),
    ("gemini", 503, {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}}, "overloaded"),
]


@pytest.mark.parametrize("stream", BOTH)
@pytest.mark.parametrize("provider, code, body, kind", BUSY_OR_BROKEN)
def test_overloaded_and_server_errors_keep_their_kind(monkeypatch, provider, code, body, kind, stream):
    use(monkeypatch, provider, status(code, body))
    assert kind_of(provider, stream) == kind


@pytest.mark.parametrize("event, kind", [
    (b'{"error":{"message":"overloaded","type":"service_unavailable_error","code":"server_is_overloaded"}}',
     "overloaded"),
    (b'{"error":{"message":"server error","type":"server_error"}}', "server"),
])
def test_an_openai_format_error_event_inside_a_stream_keeps_its_kind(monkeypatch, event, kind):
    body = FIRST_SSE["openai"] + b"data: " + event + b"\n\n"
    use(monkeypatch, "openai", lambda lib, r: lib.Response(200, content=body,
                                                           headers={"content-type": "text/event-stream"}))
    assert kind_of("openai", stream=True) == kind


# ---------- 2. a timeout moves straight to the next model ----------

def keys_for(*providers):
    return KeyContext("byok", {p: SECRET for p in providers})


@pytest.mark.parametrize("stream", BOTH)
@pytest.mark.parametrize("provider", ALL)
def test_a_timeout_is_tried_once_then_the_next_model_answers(monkeypatch, provider, stream):
    first = use(monkeypatch, provider, times_out)
    other = "anthropic" if provider != "anthropic" else "groq"       # a different adapter (its own mock) answers
    second = use(monkeypatch, other, ok(other, stream))
    chain = [PROVIDERS[provider][0], PROVIDERS[other][0]]
    routing = dataclasses.replace(load_routing(REGISTRY), chains={"medium": chain})
    sleeps = []
    client = ReliableClient(REGISTRY, routing, sleep=sleeps.append, rng=random.Random(0))
    result = client.send(MESSAGES, "medium", stream=stream, keys=keys_for(provider, other))
    if stream:
        list(result.response)
    assert result.model.name == chain[1]
    assert [(a.model_name, a.error_kind) for a in result.attempts] == [(chain[0], "timeout"), (chain[1], None)]
    assert len(first.requests) == 1 and len(second.requests) == 1 and sleeps == []


@pytest.mark.parametrize("stream", BOTH)
@pytest.mark.parametrize("provider", ALL)
def test_a_model_that_only_times_out_is_tried_exactly_once(monkeypatch, provider, stream):
    wire = use(monkeypatch, provider, status(*DEADLINES[provider][0]))
    model = PROVIDERS[provider][0]
    client = ReliableClient(REGISTRY, load_routing(REGISTRY), sleep=lambda s: None)
    with pytest.raises(AllModelsFailed) as info:
        result = client.send_model(MESSAGES, model, stream=stream, keys=keys_for(provider))
        if stream:
            list(result.response)
    assert len(wire.requests) == 1
    assert [a.error_kind for a in info.value.attempts] == ["timeout"]


# ---------- the API: provider_timeout only when every model timed out ----------

def app(script):
    deps = make_deps(script, predictor=FakePredictor(0.4, 0.95))
    return TestClient(create_app(deps=deps, auth=Auth([API_KEY]))), deps


ALL_KEYS = {"x-groq-key": SECRET, "x-openai-key": SECRET, "x-anthropic-key": SECRET, "x-gemini-key": SECRET}
SLOW = FakeBehavior(fail_before="timeout")


def chat(http, headers):
    return http.post("/v1/chat/completions", json={"model": "auto", "messages": ASK},
                     headers={"Authorization": f"Bearer {API_KEY}"} | headers)


def test_every_provider_timing_out_is_provider_timeout_naming_them_all():
    http, _ = app({name: SLOW for name in REGISTRY})
    r = chat(http, ALL_KEYS)
    assert r.status_code == 504 and r.json()["error"]["code"] == "provider_timeout"
    assert r.json()["error"]["message"].startswith("Groq, Gemini, OpenAI and Anthropic took too long to answer.")


@pytest.mark.parametrize("provider", ALL)
def test_a_timeout_then_an_answer_is_a_normal_answer(provider):
    first = PROVIDERS[provider][0]
    http, deps = app({first: SLOW})
    r = chat(http, ALL_KEYS | {"X-Triage-Prefer-Provider": provider})
    assert r.status_code == 200 and r.headers["x-triage-model"] != first
    tried = [c["model"] for c in deps.fake_models.calls if c["kind"] == "answer"]
    assert tried.count(first) == 1                                   # no same-model retry


def test_timeouts_mixed_with_a_server_error_are_not_provider_timeout():
    script = {name: SLOW for name in REGISTRY}
    script["claude-sonnet"] = FakeBehavior(fail_before="server")
    http, _ = app(script)
    assert chat(http, ALL_KEYS).json()["error"]["code"] == "all_models_failed"
