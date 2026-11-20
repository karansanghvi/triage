"""GET /v1/stats/public: the Overview tab's live aggregates (app/api/public_stats.py)."""
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from tests.api_helpers import API_KEY, make_deps
from app.api import public_stats
from app.api.auth import Auth
from app.api.main import create_app
from app.api.request_log import InMemoryRequestLog, RequestRow
from app.registry import load_registry

_ids = iter(range(10**6))


def row(days_ago: float = 0, now=None, **kw) -> RequestRow:
    now = now or datetime.now(timezone.utc)
    base = dict(request_id=f"r{next(_ids)}", timestamp=(now - timedelta(days=days_ago)).isoformat(), key_id="k",
                requested_model="auto", stream=False, status="ok", routed_by="classifier", tier="medium",
                served_model="groq-small", fell_back=False, cache_hit=False, price_as_cost=1.0, strong_cost=3.0,
                latency_ms=100.0)
    base.update(kw)
    return RequestRow(**base)


def client(*rows):
    log = InMemoryRequestLog()
    for r in rows:
        log.write(r)
    deps = make_deps(log=log)
    return TestClient(create_app(deps=deps, auth=Auth([API_KEY]))), deps


NOISE = ([row(requested_model="groq-large", routed_by="requested_model") for _ in range(60)]
         + [row(routed_by="small_talk", tier="low") for _ in range(60)])


def test_below_the_threshold_only_the_count_is_published():
    http, _ = client(*[row() for _ in range(99)], *NOISE)          # named models and small talk don't qualify
    r = http.get("/v1/stats/public")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"enough", "range", "threshold", "qualifying_requests", "generated_at"}
    assert body["enough"] is False and body["qualifying_requests"] == 99 and body["threshold"] == 100
    assert body["range"] == "30d"


def test_at_the_threshold_the_figures_are_published_without_named_models_or_small_talk():
    http, _ = client(*[row() for _ in range(100)], *NOISE)
    body = http.get("/v1/stats/public").json()
    assert body["enough"] is True and body["qualifying_requests"] == 100
    t = body["totals"]
    assert t["requests"] == 100 and t["cost_usd"] == 100.0 and t["saved_usd"] == 200.0
    assert t["saved_share"] == pytest.approx(2 / 3, abs=1e-4) and t["fallback_rate"] == 0.0
    assert body["tier_mix"] == {"medium": 1.0} and body["baseline"] == "always-strongest"
    assert body["bucket"] == "day" and len(body["series"]) == 30 and sum(p["requests"] for p in body["series"]) == 100
    assert "credited_requests" not in t and "streams" not in t


def test_no_auth_is_needed_and_the_response_is_cacheable():
    http, _ = client()
    r = http.get("/v1/stats/public")
    assert r.status_code == 200 and r.headers["cache-control"] == "public, max-age=300"


def test_ranges():
    http, _ = client(*[row(days_ago=10) for _ in range(100)], *[row(days_ago=400) for _ in range(5)])
    assert http.get("/v1/stats/public?range=7d").json()["qualifying_requests"] == 0
    assert http.get("/v1/stats/public?range=30d").json()["qualifying_requests"] == 100
    body = http.get("/v1/stats/public?range=all").json()
    assert body["qualifying_requests"] == 105 and body["bucket"] == "month"
    r = http.get("/v1/stats/public?range=90d")
    assert r.status_code == 400 and r.json()["error"]["param"] == "range"


def test_errors_fallbacks_and_latency_by_model():
    rows = ([row(latency_ms=float(ms)) for ms in range(1, 81)]
            + [row(served_model="groq-large", tier="high", latency_ms=500.0, fell_back=True) for _ in range(19)]
            + [row(served_model="retired-model", latency_ms=50.0) for _ in range(20)]
            + [row(status="error", error_kind="timeout", served_model=None, latency_ms=None) for _ in range(3)])
    http, _ = client(*rows)
    body = http.get("/v1/stats/public").json()
    assert body["errors_by_kind"] == {"timeout": 3}
    assert body["totals"]["fallbacks"] == 19 and body["totals"]["errors"] == 3
    registry = load_registry()
    latency = {m["model"]: m for m in body["latency_by_model"]}
    assert set(latency) == {"groq-small", "retired-model"}               # groq-large: only 19 requests
    assert latency["groq-small"] == {"model": "groq-small", "requests": 80, "p50_ms": 40.0, "p95_ms": 76.0,
                                     "provider": registry["groq-small"].provider,
                                     "tier": registry["groq-small"].tier}
    assert (latency["retired-model"]["provider"], latency["retired-model"]["tier"]) == ("unknown", "unknown")


def test_results_are_cached_per_range():
    http, deps = client(*[row() for _ in range(100)])
    first = http.get("/v1/stats/public").json()
    deps.log.write(row())
    assert http.get("/v1/stats/public").json() == first                     # cached: the new row isn't seen yet
    assert http.get("/v1/stats/public?range=all").json()["qualifying_requests"] == 101


def test_the_cache_expires_and_computes_once_per_range():
    now = [0.0]
    cache = public_stats.PublicStatsCache(ttl=300, clock=lambda: now[0])
    calls = []

    def compute(tag):
        calls.append(tag)
        return {"tag": tag, "n": len(calls)}
    assert cache.get("30d", lambda: compute("a")) == cache.get("30d", lambda: compute("b")) == {"tag": "a", "n": 1}
    cache.get("7d", lambda: compute("c"))
    now[0] = 299.9
    assert cache.get("30d", lambda: compute("d"))["tag"] == "a"
    now[0] = 300.0
    assert cache.get("30d", lambda: compute("e"))["tag"] == "e"
    assert calls == ["a", "c", "e"]


def test_fake_mode_refuses():
    http, deps = client()
    deps.fake_mode = True
    assert http.get("/v1/stats/public").status_code == 404


def test_compute_uses_the_given_time():
    now = datetime(2026, 11, 19, 12, tzinfo=timezone.utc)
    log = InMemoryRequestLog()
    for _ in range(100):
        log.write(row(days_ago=6.4, now=now))
    body = public_stats.compute(log, load_registry(), "7d", now)
    assert body["generated_at"] == now.isoformat() and [p["requests"] for p in body["series"]][0] == 100
