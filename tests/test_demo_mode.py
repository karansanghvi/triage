"""The free demo through the API (Phase 5): limits, errors, headers, accounting. No network.

Demo = TRIAGE_PUBLIC on, no Triage key, no provider keys. The DemoLimiter here is in memory with a fake clock.
"""
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.api.auth import Auth
from app.api.demo import DemoLimiter, DemoSettings, load_demo_settings, visitor_id
from app.api.fake_models import FakeBehavior
from app.api.main import create_app
from app.api.request_log import SQLiteRequestLog
from tests.api_helpers import API_KEY, ASK, TEST_HASH_SECRET, FakePredictor, make_deps

NOON = datetime(2026, 11, 13, 12, 0, 0, tzinfo=timezone.utc)
BEARER = {"Authorization": f"Bearer {API_KEY}"}
DOWN = FakeBehavior(fail_before="server")


class Clock:
    def __init__(self):
        self.now = NOON

    def __call__(self):
        return self.now

    def advance(self, **kw):
        self.now += timedelta(**kw)


def settings(**overrides):
    base = dict(visitor_daily_requests=10, daily_requests=300, daily_tokens=150_000, daily_model_calls=600,
                per_minute_requests=100, max_prompt_chars=8_000, max_tokens=2_048,
                public_byok_visitor_daily_requests=200, retention_days=7)
    return DemoSettings(**{**base, **overrides})


def app(script=None, hops=1, clock=None, **demo):
    limiter = DemoLimiter(settings(**demo), clock=clock or Clock())
    deps = make_deps(script, predictor=FakePredictor(0.95, 0.95), demo_limiter=limiter, trusted_proxy_hops=hops)
    return TestClient(create_app(deps=deps, auth=Auth([API_KEY]), public=True)), deps


def chat(http, ip="203.0.113.7", headers=None, **body):
    payload = {"model": "auto", "messages": ASK} | body
    return http.post("/v1/chat/completions", json=payload, headers={"X-Forwarded-For": ip} | (headers or {}))


# ---------- config ----------

def test_config_file_has_the_approved_defaults():
    assert load_demo_settings() == DemoSettings()


@pytest.mark.parametrize("text, message", [
    ("visitor_daily_requests: 10\n", "missing setting"),
    ("\n".join(f"{k}: 1" for k in DemoSettings.__dataclass_fields__) + "\nnope: 1\n", "unknown setting"),
    ("\n".join(f"{k}: {0 if k == 'daily_tokens' else 1}" for k in DemoSettings.__dataclass_fields__), "above 0"),
    ("\n".join(f"{k}: {'off' if k == 'max_tokens' else 1}" for k in DemoSettings.__dataclass_fields__), "above 0"),
])
def test_bad_demo_config_is_rejected(tmp_path, text, message):
    path = tmp_path / "demo.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match=message):
        load_demo_settings(path)


# ---------- per-visitor limit ----------

def test_ten_requests_per_visitor_then_a_429_suggesting_own_keys():
    http, deps = app()
    remaining = []
    for _ in range(10):
        r = chat(http)
        assert r.status_code == 200 and r.headers["x-triage-mode"] == "demo"
        remaining.append(int(r.headers["x-triage-demo-remaining"]))
    assert remaining == list(range(9, -1, -1))
    assert r.headers["x-triage-demo-limit"] == "10" and r.headers["x-triage-demo-resets"] == "2026-11-14T00:00:00Z"
    r = chat(http)
    assert r.status_code == 429
    err = r.json()["error"]
    assert (err["type"], err["code"]) == ("rate_limit_error", "demo_limit_reached")
    assert "X-Groq-Key" in err["message"] and "00:00 UTC" in err["message"]
    assert r.headers["retry-after"] == str(12 * 3600) and r.headers["x-triage-demo-remaining"] == "0"
    assert len(deps.fake_models.calls) == 10                         # the refused request called nothing


def test_visitors_have_their_own_allowance():
    http, _ = app(visitor_daily_requests=1)
    assert chat(http, ip="203.0.113.7").status_code == 200
    assert chat(http, ip="203.0.113.7").status_code == 429
    assert chat(http, ip="198.51.100.4").status_code == 200


def test_with_zero_trusted_hops_a_faked_forwarded_for_gets_no_new_allowance():
    http, _ = app(hops=0, visitor_daily_requests=1)
    assert chat(http, ip="1.1.1.1").status_code == 200
    assert chat(http, ip="2.2.2.2").status_code == 429               # same connection, different fake header


def test_with_one_trusted_hop_a_faked_left_entry_doesnt_matter():
    http, _ = app(hops=1, visitor_daily_requests=1)
    assert chat(http, ip="9.9.9.9, 203.0.113.7").status_code == 200
    assert chat(http, ip="8.8.8.8, 203.0.113.7").status_code == 429


def test_the_allowance_comes_back_the_next_utc_day():
    clock = Clock()
    http, _ = app(clock=clock, visitor_daily_requests=1)
    chat(http)
    assert chat(http).status_code == 429
    clock.advance(hours=12)
    assert chat(http).status_code == 200


# ---------- demo-wide limits ----------

def test_daily_total_cap_pauses_the_demo_until_tomorrow():
    http, _ = app(daily_requests=2)
    chat(http, ip="203.0.113.1")
    chat(http, ip="203.0.113.2")
    r = chat(http, ip="203.0.113.3")
    assert r.status_code == 429 and r.json()["error"]["code"] == "demo_paused"
    assert "paused until 00:00 UTC" in r.json()["error"]["message"] and r.headers["retry-after"] == str(12 * 3600)


def test_per_minute_limit_is_demo_busy():
    clock = Clock()
    http, _ = app(clock=clock, per_minute_requests=2)
    chat(http, ip="203.0.113.1")
    chat(http, ip="203.0.113.2")
    r = chat(http, ip="203.0.113.3")
    assert r.status_code == 429 and r.json()["error"]["code"] == "demo_busy"
    assert 1 <= int(r.headers["retry-after"]) <= 60
    clock.advance(seconds=61)
    assert chat(http, ip="203.0.113.3").status_code == 200


def test_tokens_are_reserved_then_settled_to_what_was_used():
    estimate = -(-len(ASK[0]["content"]) // 4) + 2048
    http, deps = app(daily_tokens=estimate + 100)                   # room for one reservation, not two
    assert chat(http).status_code == 200                              # settled: it used far fewer than reserved
    assert chat(http, ip="198.51.100.4").status_code == 200           # so the second one fits
    row = deps.log.rows()[0]
    assert row.input_tokens + row.output_tokens < 100


def test_every_model_call_counts_including_retries_and_fallbacks():
    http, deps = app({"groq-allam": DOWN}, daily_model_calls=4)       # 3 tries of groq-allam + ollama-llama = 4
    assert chat(http).status_code == 200
    r = chat(http, ip="198.51.100.4")
    assert r.status_code == 429 and r.json()["error"]["code"] == "demo_paused"
    assert deps.demo_limiter.status("x").paused_reason == "daily_model_calls"


def test_cache_hits_dont_use_the_call_budget(tmp_path):
    from app.cache import ResponseCache
    limiter = DemoLimiter(settings(daily_model_calls=2), clock=Clock())
    deps = make_deps(predictor=FakePredictor(0.95, 0.95), demo_limiter=limiter, trusted_proxy_hops=1,
                     cache=ResponseCache(tmp_path / "cache.sqlite3"))
    http = TestClient(create_app(deps=deps, auth=Auth([API_KEY]), public=True))
    assert chat(http).headers["x-triage-cache"] == "miss"                      # 1 call
    assert chat(http, ip="198.51.100.4").headers["x-triage-cache"] == "hit"     # 0 calls
    assert limiter.status("x").paused is False
    other = [{"role": "user", "content": "A different question"}]
    assert chat(http, ip="198.51.100.5", messages=other).headers["x-triage-cache"] == "miss"   # 2 calls
    assert limiter.status("x").paused_reason == "daily_model_calls"
    assert len(deps.fake_models.calls) == 2


# ---------- per-request caps ----------

def test_a_long_prompt_is_413_and_isnt_counted():
    http, deps = app(max_prompt_chars=50)
    r = chat(http, messages=[{"role": "user", "content": "x" * 51}])
    assert r.status_code == 413
    err = r.json()["error"]
    assert (err["code"], err["param"]) == ("demo_prompt_too_long", "messages")
    assert "50 characters" in err["message"] and "provider key" in err["message"]
    assert deps.fake_models.calls == []
    assert chat(http, messages=[{"role": "system", "content": "x" * 25},
                                {"role": "user", "content": "y" * 25}]).status_code == 200       # exactly 50: fine
    assert deps.demo_limiter.status(visitor_id("203.0.113.7", TEST_HASH_SECRET)).used == 1


@pytest.mark.parametrize("requested, sent", [(None, 2048), (500, 500), (5000, 2048)])
def test_max_tokens_is_quietly_lowered_to_the_demo_cap(requested, sent):
    http, deps = app()
    body = {} if requested is None else {"max_tokens": requested}
    r = chat(http, **body)
    assert r.status_code == 200 and r.headers["x-triage-max-tokens"] == str(sent)
    assert deps.fake_models.calls[0]["max_tokens"] == sent


def test_sync_verification_isnt_available_in_the_demo():
    http, deps = app()
    r = chat(http, headers={"X-Triage-Verify": "sync"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "demo_unsupported"
    assert deps.fake_models.calls == [] and r.headers["x-triage-demo-remaining"] == "10"


# ---------- streams, logging, privacy ----------

def test_streamed_demo_requests_are_counted_and_settled():
    http, deps = app(visitor_daily_requests=2)
    r = chat(http, stream=True)
    assert r.status_code == 200 and r.headers["x-triage-demo-remaining"] == "1"
    assert "[DONE]" in r.text
    assert chat(http, stream=True).status_code == 200
    assert chat(http, stream=True).status_code == 429


def test_refused_requests_are_logged_without_any_ip(tmp_path):
    limiter = DemoLimiter(settings(visitor_daily_requests=1), path=tmp_path / "usage.sqlite3", clock=Clock())
    log = SQLiteRequestLog(tmp_path / "requests.sqlite3")
    deps = make_deps(predictor=FakePredictor(0.95, 0.95), demo_limiter=limiter, trusted_proxy_hops=1, log=log)
    http = TestClient(create_app(deps=deps, auth=Auth([API_KEY]), public=True))
    chat(http, ip="203.0.113.77")
    chat(http, ip="203.0.113.77")
    rows = log.rows()
    assert [(r.mode, r.key_id, r.http_status) for r in rows] == [("demo", "demo", 200), ("demo", "demo", 429)]
    assert rows[1].error_kind == "demo_limit_reached"
    log.close()
    limiter.close()
    for f in tmp_path.iterdir():
        assert b"203.0.113.77" not in f.read_bytes(), f.name


# ---------- api and byok are unchanged ----------

def test_api_requests_have_no_demo_limits_or_headers():
    http, deps = app(visitor_daily_requests=1, daily_requests=1, max_tokens=100)
    for _ in range(3):
        r = chat(http, headers=BEARER)
        assert r.status_code == 200 and r.headers["x-triage-mode"] == "api"
        assert "x-triage-demo-remaining" not in r.headers and "x-triage-max-tokens" not in r.headers
    assert all(c["max_tokens"] == 4000 for c in deps.fake_models.calls)
    assert deps.demo_limiter.status("x").paused is False


def test_byok_requests_with_a_triage_key_have_no_limits():
    http, deps = app(visitor_daily_requests=1, daily_requests=1)
    for _ in range(3):
        r = chat(http, headers=BEARER | {"x-groq-key": "gsk_test-key-123"})
        assert r.status_code == 200 and r.headers["x-triage-mode"] == "byok"
        assert "x-triage-demo-remaining" not in r.headers


# ---------- GET /v1/demo/status ----------

def status(http, ip="203.0.113.7", headers=None):
    return http.get("/v1/demo/status", headers={"X-Forwarded-For": ip} | (headers or {}))


def test_status_for_a_new_visitor_needs_no_key():
    http, _ = app()
    r = status(http)
    assert r.status_code == 200
    assert r.json() == {"enabled": True, "paused": False, "paused_reason": None,
                        "visitor": {"limit": 10, "used": 0, "remaining": 10},
                        "limits": {"max_prompt_chars": 8000, "max_tokens": 2048},
                        "resets_at": "2026-11-14T00:00:00Z"}


def test_status_follows_the_visitors_usage_and_counts_nothing():
    http, _ = app(visitor_daily_requests=3)
    chat(http)
    for _ in range(5):
        status(http)
    assert status(http).json()["visitor"] == {"limit": 3, "used": 1, "remaining": 2}
    assert status(http, ip="198.51.100.4").json()["visitor"]["used"] == 0
    assert chat(http).headers["x-triage-demo-remaining"] == "1"


def test_status_shows_a_paused_demo():
    http, _ = app(daily_requests=1)
    chat(http)
    body = status(http, ip="198.51.100.4").json()
    assert (body["paused"], body["paused_reason"]) == (True, "daily_requests")


def test_status_when_public_access_is_off():
    deps = make_deps(demo_limiter=DemoLimiter(settings(), clock=Clock()))
    http = TestClient(create_app(deps=deps, auth=Auth([API_KEY]), public=False))
    assert status(http).json() == {"enabled": False}


def test_status_ignores_a_triage_key():
    http, _ = app()
    assert status(http, headers=BEARER).json()["enabled"] is True


# ---------- public BYOK: a generous per-visitor cap ----------

GROQ_KEY = {"x-groq-key": "gsk_test-key-123"}


def test_public_byok_has_its_own_per_visitor_cap():
    http, deps = app(public_byok_visitor_daily_requests=2)
    for _ in range(2):
        r = chat(http, headers=GROQ_KEY)
        assert r.status_code == 200 and r.headers["x-triage-mode"] == "byok"
        assert "x-triage-demo-remaining" not in r.headers
    r = chat(http, headers=GROQ_KEY)
    assert r.status_code == 429
    err = r.json()["error"]
    assert err["code"] == "public_limit_reached" and "Triage API key" in err["message"]
    assert r.headers["retry-after"] == str(12 * 3600)
    assert chat(http, ip="198.51.100.4", headers=GROQ_KEY).status_code == 200
    assert chat(http, headers=GROQ_KEY | BEARER).status_code == 200            # with a Triage key: no cap


def test_public_byok_and_the_demo_are_counted_apart():
    http, _ = app(visitor_daily_requests=1, public_byok_visitor_daily_requests=1)
    assert chat(http).status_code == 200
    assert chat(http).status_code == 429                                        # demo used up
    assert chat(http, headers=GROQ_KEY).status_code == 200                      # BYOK still fine
    assert chat(http, headers=GROQ_KEY).status_code == 429


def test_public_byok_doesnt_use_the_demo_budgets():
    http, deps = app(daily_requests=1, daily_tokens=10, daily_model_calls=1, per_minute_requests=1,
                     public_byok_visitor_daily_requests=50)
    for i in range(5):
        assert chat(http, ip=f"198.51.100.{i}", headers=GROQ_KEY).status_code == 200
    assert deps.demo_limiter.status("x").paused is False
    assert deps.demo_limiter.requests_in_last_minute() == 0
