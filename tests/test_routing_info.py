"""What a response tells the website about how it was routed (Phase 5, chat page). No network.

Headers: X-Triage-P-Low, -P-Medium and -Tau (model "auto" only, streams too), X-Triage-Strong-Cost (non-streaming).
Body: a top-level "triage" object on non-streaming responses, the same values as the headers.
"""
import openai
import pytest
from fastapi.testclient import TestClient

from app.api.auth import Auth
from app.api.main import create_app
from tests.api_helpers import API_KEY, ASK, FakePredictor, make_deps, routing_raw

BEARER = {"Authorization": f"Bearer {API_KEY}"}
PROBABILITY_HEADERS = ("x-triage-p-low", "x-triage-p-medium", "x-triage-tau")


def app(predictor=None, **kw):
    deps = make_deps(predictor=predictor or FakePredictor(0.4, 0.95), **kw)
    return TestClient(create_app(deps=deps, auth=Auth([API_KEY]))), deps


def chat(http, body=None, headers=None):
    return http.post("/v1/chat/completions", json=body or {"model": "auto", "messages": ASK},
                     headers=BEARER | (headers or {}))


# ---------- p_low, p_medium, tau ----------

def test_auto_reports_the_probabilities_and_tau_as_numbers_with_6_decimals():
    http, deps = app(FakePredictor(0.4, 0.95123456))
    r = chat(http)
    assert r.status_code == 200 and r.headers["x-triage-tier"] == "medium"
    assert (r.headers["x-triage-p-low"], r.headers["x-triage-p-medium"], r.headers["x-triage-tau"]) == \
        ("0.400000", "0.951235", "0.860000")
    (row,) = deps.log.rows()
    assert float(r.headers["x-triage-p-medium"]) == pytest.approx(row.p_medium, abs=1e-6)


def test_p_medium_is_the_adjusted_value_routing_used():
    http, _ = app(FakePredictor(0.5, 0.3))           # the medium model's own 0.3 is raised to stay >= cheap
    assert chat(http).headers["x-triage-p-medium"] == "0.500000"


@pytest.mark.parametrize("quality, tau", [("0.7", "0.700000"), ("high", "0.890000"), ("cheap", "0.680000"),
                                          ("1", "1.000000"), ("0", "0.000000")])
def test_tau_is_the_one_actually_used(quality, tau):
    http, _ = app()
    assert chat(http, headers={"X-Triage-Quality": quality}).headers["x-triage-tau"] == tau


def test_a_named_model_has_no_probabilities_or_tau():
    http, _ = app()
    r = chat(http, {"model": "groq-small", "messages": ASK}, headers={"X-Triage-Quality": "0.7"})
    assert r.status_code == 200
    assert not set(PROBABILITY_HEADERS) & set(r.headers)


def test_streams_get_the_probabilities_and_tau_at_the_start():
    http, _ = app()
    with http.stream("POST", "/v1/chat/completions", headers=BEARER,
                     json={"model": "auto", "messages": ASK, "stream": True}) as r:
        "".join(r.iter_text())
        h = r.headers
    assert (h["x-triage-p-low"], h["x-triage-p-medium"], h["x-triage-tau"]) == ("0.400000", "0.950000", "0.860000")
    assert "x-triage-strong-cost" not in h


# ---------- strong cost ----------

def test_strong_cost_matches_the_log_and_saved_is_strong_minus_cost():
    http, deps = app()
    h = chat(http).headers
    (row,) = deps.log.rows()
    assert float(h["x-triage-strong-cost"]) == pytest.approx(row.strong_cost, abs=1e-8) and row.strong_cost > 0
    assert float(h["x-triage-saved"]) == pytest.approx(float(h["x-triage-strong-cost"]) - float(h["x-triage-cost"]),
                                                       abs=1e-8)


def test_a_named_model_still_reports_the_strong_cost():
    http, _ = app()
    assert float(chat(http, {"model": "groq-small", "messages": ASK}).headers["x-triage-strong-cost"]) > 0


def test_no_reachable_strong_model_leaves_strong_cost_and_saved_out():
    raw = routing_raw()
    raw["fallback_chains"]["high"] = ["groq-large"]                 # no gemini model in high: no baseline
    http, deps = app(FakePredictor(0.95, 0.95), raw=raw)
    r = chat(http, headers={"x-gemini-key": "sk-test-SECRET123"})
    assert r.status_code == 200
    assert "x-triage-strong-cost" not in r.headers and "x-triage-saved" not in r.headers
    assert "x-triage-cost" in r.headers
    (row,) = deps.log.rows()
    assert row.strong_cost is None


# ---------- the "triage" object in non-streaming bodies ----------

TRIAGE_FIELDS = {"request_id": "x-triage-request-id", "model": "x-triage-model", "tier": "x-triage-tier",
                 "p_low": "x-triage-p-low", "p_medium": "x-triage-p-medium", "tau": "x-triage-tau",
                 "reason": "x-triage-reason", "cost": "x-triage-cost", "strong_cost": "x-triage-strong-cost",
                 "saved": "x-triage-saved", "cache": "x-triage-cache", "fallback": "x-triage-fallback",
                 "mode": "x-triage-mode", "preferred_provider": "x-triage-preferred-provider",
                 "routed_by": "x-triage-routed-by"}
NUMBERS = {"p_low", "p_medium", "tau", "cost", "strong_cost", "saved"}


def matches_headers(triage, headers):
    assert set(triage) == set(TRIAGE_FIELDS)
    for field, header in TRIAGE_FIELDS.items():
        if header not in headers or (field == "preferred_provider" and headers[header] == "none"):
            assert triage[field] is None, field
        elif field in NUMBERS:
            assert isinstance(triage[field], float) and triage[field] == float(headers[header]), field
        else:
            assert triage[field] == headers[header], field


def test_auto_body_has_the_triage_object_matching_the_headers():
    http, _ = app(FakePredictor(0.4, 0.95123456))
    r = chat(http)
    triage = r.json()["triage"]
    matches_headers(triage, r.headers)
    assert (triage["tier"], triage["p_low"], triage["p_medium"], triage["tau"]) == ("medium", 0.4, 0.951235, 0.86)
    assert (triage["cache"], triage["fallback"], triage["mode"]) == ("miss", "none", "api")
    assert triage["request_id"] == r.json()["id"] and triage["model"] == r.json()["model"]
    assert triage["saved"] == pytest.approx(triage["strong_cost"] - triage["cost"], abs=1e-8)


def test_named_model_body_has_null_probabilities_and_tau():
    http, _ = app()
    r = chat(http, {"model": "groq-small", "messages": ASK})
    triage = r.json()["triage"]
    matches_headers(triage, r.headers)
    assert (triage["p_low"], triage["p_medium"], triage["tau"]) == (None, None, None)
    assert triage["reason"] == "requested model groq-small" and triage["strong_cost"] > 0


def test_unknown_strong_cost_and_saved_are_null():
    raw = routing_raw()
    raw["fallback_chains"]["high"] = ["groq-large"]
    http, _ = app(FakePredictor(0.95, 0.95), raw=raw)
    r = chat(http, headers={"x-gemini-key": "sk-test-SECRET123"})
    triage = r.json()["triage"]
    matches_headers(triage, r.headers)
    assert (triage["strong_cost"], triage["saved"], triage["mode"]) == (None, None, "byok")


def test_fallback_and_cache_hit_show_in_the_body(tmp_path):
    from app.api.fake_models import FakeBehavior
    from app.cache import ResponseCache
    http, _ = app(FakePredictor(0.95, 0.95), script={"groq-allam": FakeBehavior(fail_before="server")},
                  cache=ResponseCache(tmp_path / "cache.sqlite3"))
    first = chat(http)
    assert first.json()["triage"]["fallback"].startswith("groq-allam->")
    matches_headers(first.json()["triage"], first.headers)
    second = chat(http).json()["triage"]
    assert second["cache"] == "hit" and second["cost"] == 0.0 and second["saved"] == second["strong_cost"]


def test_openai_sdk_still_parses_the_response():
    http, _ = app()
    client = openai.OpenAI(base_url="http://testserver/v1", api_key=API_KEY, http_client=http, max_retries=0)
    completion = client.chat.completions.create(model="auto", messages=ASK)
    assert completion.choices[0].message.content and completion.model


# ---------- the request id on errors ----------

def test_logged_errors_carry_the_request_id():
    http, deps = app()
    r = chat(http, {"model": "no-such-model", "messages": ASK})
    assert r.status_code == 404
    (row,) = deps.log.rows()
    assert r.headers["x-triage-request-id"] == row.request_id


def test_a_failed_answer_carries_the_request_id_of_its_log_row():
    from app.api.fake_models import FakeBehavior
    http, deps = app(script={name: FakeBehavior(fail_before="server") for name in deps_registry()})
    r = chat(http)
    assert r.status_code == 503 and r.json()["error"]["code"] == "all_models_failed"
    (row,) = deps.log.rows()
    assert r.headers["x-triage-request-id"] == row.request_id and row.status == "error"


def test_the_demo_limit_429_carries_the_request_id(tmp_path):
    from app.api.demo import DemoLimiter, DemoSettings
    settings = DemoSettings(visitor_daily_requests=1)
    deps = make_deps(predictor=FakePredictor(0.95, 0.95), demo_settings=settings,
                     demo_limiter=DemoLimiter(settings, tmp_path / "demo.sqlite3"))
    http = TestClient(create_app(deps=deps, auth=Auth([API_KEY]), public=True))
    assert http.post("/v1/chat/completions", json={"model": "auto", "messages": ASK}).status_code == 200
    r = http.post("/v1/chat/completions", json={"model": "auto", "messages": ASK})
    assert r.status_code == 429 and r.headers["x-triage-request-id"] == deps.log.rows()[-1].request_id


@pytest.mark.parametrize("body, headers", [({"model": "auto", "messages": ASK, "top_p": 1}, {}),
                                           (None, {"X-Triage-Quality": "86%"}),
                                           ({"model": "auto", "messages": []}, {})])
def test_errors_before_a_request_exists_have_no_id(body, headers):
    http, deps = app()
    r = chat(http, body, headers)
    assert r.status_code == 400 and "x-triage-request-id" not in r.headers and deps.log.rows() == []


def test_the_success_id_is_unchanged():
    http, _ = app()
    r = chat(http)
    assert r.headers["x-triage-request-id"] == r.json()["id"] == r.json()["triage"]["request_id"]


def deps_registry():
    from app.registry import load_registry
    return load_registry()
