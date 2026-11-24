"""The free demo's per-visitor allowance: exactly one count per user message, whatever happens inside it
(fallbacks, retries, cache hits, streams, stops), never for health or status reads, and given back when the
request failed on the server's side. No network: fake models, an in-memory DemoLimiter with a fake clock."""
import asyncio
import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.api.auth import Auth
from app.api.demo import DemoLimiter, DemoSettings, visitor_id
from app.api.fake_models import FakeBehavior
from app.api.main import create_app, demo_refund
from app.api.request_log import RequestRow
from app.cache import ResponseCache
from tests.api_helpers import API_KEY, ASK, TEST_HASH_SECRET, FakePredictor, make_deps

NOON = datetime(2026, 11, 13, 12, 0, 0, tzinfo=timezone.utc)
IP = "203.0.113.7"
DOWN = FakeBehavior(fail_before="server")


def settings(**overrides):
    base = dict(visitor_daily_requests=10, daily_requests=300, daily_tokens=150_000, daily_model_calls=600,
                per_minute_requests=100, max_prompt_chars=4_000, max_tokens=2_048,
                public_byok_visitor_daily_requests=200, retention_days=7)
    return DemoSettings(**{**base, **overrides})


class CountingLimiter(DemoLimiter):
    """Records every admission and settlement the API makes."""

    def __init__(self, *args, **kwargs):
        self.admits, self.settles = [], []
        super().__init__(*args, **kwargs)

    def admit_demo(self, visitor, estimated_tokens):
        admission = super().admit_demo(visitor, estimated_tokens)
        self.admits.append(("demo", visitor))
        return admission

    def admit_public_byok(self, visitor):
        admission = super().admit_public_byok(visitor)
        self.admits.append(("byok", visitor))
        return admission

    def settle(self, admission, tokens, calls, refund=False):
        self.settles.append({"tokens": tokens, "calls": calls, "refund": refund})
        return super().settle(admission, tokens, calls, refund=refund)


def make(script=None, hops=1, cache=None, **demo):
    limiter = CountingLimiter(settings(**demo), clock=lambda: NOON)
    deps = make_deps(script, predictor=FakePredictor(0.95, 0.95), demo_limiter=limiter, trusted_proxy_hops=hops,
                     cache=cache)
    return create_app(deps=deps, auth=Auth([API_KEY]), public=True), deps, limiter


def chat(http, ip=IP, headers=None, **body):
    payload = {"model": "auto", "messages": ASK} | body
    return http.post("/v1/chat/completions", json=payload, headers={"X-Forwarded-For": ip} | (headers or {}))


def used(limiter, ip=IP):
    return limiter.status(visitor_id(ip, TEST_HASH_SECRET)).used


def every_model(deps, behavior):
    for name in deps.registry:
        deps.fake_models.script[name] = behavior


# ---------- exactly one count per message ----------

def test_an_answer_is_one_count_and_one_settlement():
    app, deps, limiter = make()
    with TestClient(app) as http:
        r = chat(http)
    assert r.status_code == 200 and r.headers["x-triage-demo-remaining"] == "9"
    assert len(limiter.admits) == 1 and len(limiter.settles) == 1 and not limiter.settles[0]["refund"]
    assert used(limiter) == 1


def test_retries_and_a_fallback_are_still_one_count():
    app, deps, limiter = make({"groq-allam": DOWN})            # 3 tries of groq-allam, then the fallback answers
    with TestClient(app) as http:
        r = chat(http)
    assert r.status_code == 200 and r.headers["x-triage-fallback"] != "none"
    assert len(deps.fake_models.calls) == 4                        # four model calls ...
    assert len(limiter.admits) == 1 and used(limiter) == 1         # ... one request from the visitor's allowance
    assert limiter.settles == [{"tokens": limiter.settles[0]["tokens"], "calls": 4, "refund": False}]
    assert _totals(limiter) == (1, limiter.settles[0]["tokens"], 4)


def test_a_cache_hit_is_one_count():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        app, deps, limiter = make(cache=ResponseCache(Path(tmp) / "cache.sqlite3"))
        with TestClient(app) as http:
            assert chat(http).headers["x-triage-cache"] == "miss"
            assert chat(http).headers["x-triage-cache"] == "hit"
    assert len(limiter.admits) == 2 and used(limiter) == 2


def test_a_streamed_answer_is_one_count():
    app, deps, limiter = make()
    with TestClient(app) as http:
        r = chat(http, stream=True)
        assert r.status_code == 200 and "[DONE]" in r.text
    assert len(limiter.admits) == 1 and len(limiter.settles) == 1 and used(limiter) == 1


def test_messages_are_counted_one_by_one_until_the_limit():
    app, deps, limiter = make(visitor_daily_requests=3)
    with TestClient(app) as http:
        remaining = [chat(http).headers["x-triage-demo-remaining"] for _ in range(3)]
        refused = chat(http)
    assert remaining == ["2", "1", "0"] and refused.status_code == 429
    assert len(limiter.admits) == 3 and len(limiter.settles) == 3 and used(limiter) == 3


# ---------- reads never count ----------

def test_health_status_and_request_lookups_never_count():
    app, deps, limiter = make()
    with TestClient(app) as http:
        request_id = chat(http).headers["x-triage-request-id"]
        for _ in range(3):
            http.get("/healthz")
            http.get("/readyz")
            assert http.get("/v1/demo/status", headers={"X-Forwarded-For": IP}).json()["visitor"]["used"] == 1
            http.get(f"/v1/requests/{request_id}", headers={"X-Forwarded-For": IP})
    assert len(limiter.admits) == 1 and len(limiter.settles) == 1 and used(limiter) == 1


@pytest.mark.parametrize("body, headers, status", [
    ({"messages": "not a list"}, {}, 400),                                        # validation
    ({"logit_bias": {"1": 1}}, {}, 400),                                          # unsupported parameter
    ({}, {"X-Triage-Verify": "sync"}, 400),                                       # no sync verify in the demo
    ({"messages": [{"role": "user", "content": "x" * 4001}]}, {}, 413),           # prompt too long
    ({"model": "no-such-model"}, {}, 404),                                        # unknown named model
])
def test_requests_refused_before_admission_are_never_counted(body, headers, status):
    app, deps, limiter = make()
    with TestClient(app) as http:
        assert chat(http, headers=headers, **body).status_code == status
    assert limiter.admits == [] and limiter.settles == [] and used(limiter) == 0


# ---------- server-side failures are given back ----------

@pytest.mark.parametrize("kind, status, code", [
    ("server", 503, "all_models_failed"),
    ("timeout", 504, "provider_timeout"),
    ("overloaded", 503, "provider_overloaded"),
])
def test_every_model_failing_gives_the_request_back(kind, status, code):
    app, deps, limiter = make()
    every_model(deps, FakeBehavior(fail_before=kind))
    with TestClient(app) as http:
        r = chat(http)
    assert r.status_code == status and r.json()["error"]["code"] == code
    assert r.headers["x-triage-demo-remaining"] == "10"            # the error already shows it given back
    assert len(limiter.admits) == 1 and len(limiter.settles) == 1 and limiter.settles[0]["refund"]
    assert used(limiter) == 0
    calls = limiter.settles[0]["calls"]
    assert calls > 0 and _totals(limiter) == (0, limiter.settles[0]["tokens"], calls)


def test_a_model_this_request_cant_use_gives_the_request_back():
    app, deps, limiter = make()
    with TestClient(app) as http:
        r = chat(http, model="claude-opus")                         # needs X-Anthropic-Key
    assert r.status_code == 403 and r.json()["error"]["code"] == "model_not_allowed"
    assert len(limiter.admits) == 1 and limiter.settles[0]["refund"] and used(limiter) == 0
    assert r.headers["x-triage-demo-remaining"] == "10"


def test_a_stream_broken_part_way_gives_the_request_back():
    app, deps, limiter = make()
    every_model(deps, FakeBehavior(fail_after=2))
    with TestClient(app) as http:
        r = chat(http, stream=True)
    assert r.status_code == 200 and "part-way" in r.text
    row = deps.log.rows()[0]
    assert row.error_kind.startswith("stream_interrupted:")
    assert len(limiter.admits) == 1 and limiter.settles[0]["refund"] and used(limiter) == 0


def test_a_refund_is_given_once_and_the_allowance_stays_honest():
    app, deps, limiter = make(visitor_daily_requests=2)
    with TestClient(app) as http:
        assert chat(http).status_code == 200
        assert chat(http, model="claude-opus").status_code == 403   # given back
        assert chat(http).headers["x-triage-demo-remaining"] == "0"
        assert chat(http).status_code == 429
    assert used(limiter) == 2


# ---------- a stopped request still counts ----------

def test_a_stopped_stream_still_counts():
    from tests.test_stream_stop import LONG, stream_then_disconnect
    app, deps, limiter = make({"groq-allam": FakeBehavior(text=LONG, piece_delay=0.01)}, hops=0)
    asyncio.run(stream_then_disconnect(app, after=3, key=None))
    row = deps.log.rows()[0]
    assert (row.status, row.error_kind) == ("stopped", "client_disconnected")
    assert len(limiter.admits) == 1 and limiter.settles == [{"tokens": limiter.settles[0]["tokens"],
                                                             "calls": 1, "refund": False}]
    assert used(limiter, "203.0.113.9") == 1


# ---------- which outcomes are refunded ----------

def row(status, http_status, error_kind=None):
    return RequestRow(request_id="r", timestamp="t", key_id="demo", requested_model="auto", stream=True,
                      status=status, http_status=http_status, error_kind=error_kind)


@pytest.mark.parametrize("status, http_status, error_kind, refunded", [
    ("ok", 200, None, False),
    ("stopped", 499, "client_disconnected", False),
    ("error", 503, "all_models_failed", True),
    ("error", 503, "provider_overloaded", True),
    ("error", 503, "provider_unavailable", True),
    ("error", 504, "provider_timeout", True),
    ("error", 502, "server", True),                            # a model failed while answering (not streamed)
    ("error", 200, "stream_interrupted:server", True),
    ("error", 400, "no_usable_model", True),
    ("error", 403, "model_not_allowed", True),
    ("error", 404, "model_not_found", True),
    ("error", 400, "truncated", True),
    ("error", 401, "invalid_provider_key", False),             # the caller's own key (and BYOK isn't demo)
    ("error", 429, "demo_limit_reached", False),
])
def test_demo_refund(status, http_status, error_kind, refunded):
    assert demo_refund(row(status, http_status, error_kind)) is refunded


# ---------- DemoLimiter.settle(refund=True) ----------

def test_settle_with_refund_gives_back_the_request_but_keeps_tokens_calls_and_the_window():
    limiter = DemoLimiter(settings(), clock=lambda: NOON)
    admission = limiter.admit_demo("v", 1000)
    limiter.settle(admission, tokens=300, calls=3, refund=True)
    assert limiter.status("v").used == 0 and limiter.status("v").remaining == 10
    assert _totals(limiter) == (0, 300, 3)
    assert limiter.requests_in_last_minute() == 1                  # the providers were still asked


def test_a_refund_touches_only_that_visitor():
    limiter = DemoLimiter(settings(), clock=lambda: NOON)
    mine, theirs = limiter.admit_demo("v", 10), limiter.admit_demo("w", 10)
    limiter.settle(mine, tokens=0, calls=0, refund=True)
    limiter.settle(theirs, tokens=0, calls=0)
    assert limiter.status("v").used == 0 and limiter.status("w").used == 1
    assert _totals(limiter)[0] == 1


def test_a_refund_never_takes_a_count_below_zero():
    limiter = DemoLimiter(settings(), clock=lambda: NOON)
    admission = limiter.admit_demo("v", 10)
    for _ in range(2):          # the API settles once per request; even a second refund can't go negative
        limiter.settle(admission, tokens=0, calls=0, refund=True)
    assert limiter.status("v").used == 0 and _totals(limiter)[0] == 0


def test_a_byok_refund_is_a_no_op():
    limiter = DemoLimiter(settings(), clock=lambda: NOON)
    admission = limiter.admit_public_byok("v")
    limiter.settle(admission, tokens=0, calls=0, refund=True)
    assert limiter.admit_public_byok("v").remaining == 198


def _totals(limiter):
    with limiter.engine.connect() as db:
        return limiter._totals(db, "2026-11-13")
