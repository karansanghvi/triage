"""Stopping a stream (the web chat's Stop): the client disconnects part-way. The provider's stream is closed, the
request is logged once with an estimated cost, a partial answer is never cached, and a caller without a key can
read that cost back by request id. No network: fake models and a hand-driven ASGI client."""
import asyncio
import json

from fastapi.testclient import TestClient

from app.api.auth import Auth
from app.api.fake_models import FakeBehavior
from app.api.main import create_app, demo_usage
from app.api.request_log import RequestRow
from app.cache import ResponseCache
from tests.api_helpers import API_KEY, ASK, FakePredictor, make_deps

LONG = " ".join(f"word{i}" for i in range(200))          # 200 pieces, so a stop lands well before the end


def make(tmp_path=None, public=False, script=None):
    deps = make_deps(script or {"groq-small": FakeBehavior(text=LONG, piece_delay=0.01)},
                     predictor=FakePredictor(0.4, 0.95),
                     cache=ResponseCache(tmp_path / "c.sqlite3") if tmp_path else None)
    return create_app(deps=deps, auth=Auth([API_KEY]), public=public), deps


async def stream_then_disconnect(app, after: int, key=API_KEY, body=None) -> list:
    """POST a streamed chat; after `after` body messages, the client disconnects (as a browser's abort does).
    Returns the SSE text received."""
    payload = json.dumps(body or {"model": "auto", "messages": ASK, "stream": True}).encode()
    received, gone = [], asyncio.Event()
    request_sent = False

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": payload, "more_body": False}
        await gone.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            received.append(message["body"].decode())
            if len(received) >= after:
                gone.set()

    headers = [(b"content-type", b"application/json")]
    if key:
        headers.append((b"authorization", f"Bearer {key}".encode()))
    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
             "method": "POST", "scheme": "http", "path": "/v1/chat/completions", "raw_path": b"/v1/chat/completions",
             "query_string": b"", "root_path": "", "headers": headers, "server": ("testserver", 80),
             "client": ("203.0.113.9", 50000)}
    await asyncio.wait_for(app(scope, receive, send), timeout=20)
    return received


def request_id_of(received: list) -> str:
    first = json.loads(received[0].split("data: ", 1)[1])
    return first["triage"]["request_id"]


def test_a_stop_closes_the_providers_stream_and_logs_an_estimated_cost():
    app, deps = make()
    received = asyncio.run(stream_then_disconnect(app, after=4))
    assert deps.fake_models.stopped == ["groq-small"]               # the model was told to stop
    row = deps.log.get(request_id_of(received))
    assert (row.status, row.http_status, row.error_kind) == ("error", 499, "client_disconnected")
    assert row.tokens_estimated is True and 0 < row.output_tokens < len(LONG) // 4
    assert row.price_as_cost > 0 and row.strong_cost > row.price_as_cost and row.saved > 0
    assert row.training_eligible is False
    assert len(deps.log.rows()) == 1                                  # logged once


def test_a_stopped_answer_costs_less_than_the_whole_one():
    app, deps = make()
    stopped = deps.log.get(request_id_of(asyncio.run(stream_then_disconnect(app, after=4))))
    with TestClient(app) as http:
        with http.stream("POST", "/v1/chat/completions", headers={"Authorization": f"Bearer {API_KEY}"},
                         json={"model": "auto", "messages": ASK, "stream": True}) as r:
            "".join(r.iter_text())
            full = deps.log.get(r.headers["x-triage-request-id"])
    assert 0 < stopped.price_as_cost < full.price_as_cost


def test_a_stopped_answer_is_never_cached(tmp_path):
    app, deps = make(tmp_path)
    asyncio.run(stream_then_disconnect(app, after=4))
    with TestClient(app) as http:
        with http.stream("POST", "/v1/chat/completions", headers={"Authorization": f"Bearer {API_KEY}"},
                         json={"model": "auto", "messages": ASK, "stream": True}) as r:
            text = "".join(r.iter_text())
            assert r.headers["x-triage-cache"] == "miss"
    assert "word199" in text


def test_a_demo_stop_can_be_read_back_without_a_key():
    app, deps = make(public=True)
    received = asyncio.run(stream_then_disconnect(app, after=4, key=None))
    request_id = request_id_of(received)
    with TestClient(app) as http:
        body = http.get(f"/v1/requests/{request_id}").json()
    assert body["status"] == "error" and body["error_kind"] == "client_disconnected"
    assert body["tokens_estimated"] is True and body["price_as_cost"] > 0 and body["saved"] > 0
    assert "attempts" not in body and "p_low" not in body and "key_id" not in body     # cost and outcome only


def test_the_demo_settles_on_the_estimate_not_the_reservation():
    row = RequestRow(request_id="r", timestamp="t", key_id="demo", requested_model="auto", stream=True,
                     served_model="groq-small", input_tokens=40, output_tokens=25, tokens_estimated=True,
                     attempts=[{"model": "groq-small", "outcome": "success", "error_kind": None}])
    assert demo_usage(row, reserved=4000) == (65, 1)


# ---------- the real provider streams end their HTTP stream when stopped early ----------

class FakeHttpStream:
    """Stands in for an SDK's streaming response: iterable chunks, and close() ends the connection."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.closed = False

    def __iter__(self):
        for chunk in self._chunks:
            if self.closed:
                return
            yield chunk

    def close(self):
        self.closed = True


def test_an_openai_format_stream_is_closed_when_stopped_early():
    from types import SimpleNamespace
    from app.providers import openai_provider
    chunks = [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=f"w{i} "), finish_reason=None)],
                              usage=None) for i in range(10)]
    http_stream = FakeHttpStream(chunks)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: http_stream)))
    events = openai_provider._stream(client, ASK, "m", "groq", {})
    assert next(events) == "w0 " and next(events) == "w1 "
    events.close()                                                    # the caller stopped
    assert http_stream.closed is True


def test_a_gemini_stream_is_closed_when_stopped_early():
    from types import SimpleNamespace
    from app.providers import gemini_provider
    http_stream = FakeHttpStream([SimpleNamespace(text=f"w{i} ", usage_metadata=None) for i in range(10)])
    client = SimpleNamespace(models=SimpleNamespace(generate_content_stream=lambda **kw: http_stream))
    events = gemini_provider._stream(client, "m", [], None)
    assert next(events) == "w0 "
    events.close()
    assert http_stream.closed is True
