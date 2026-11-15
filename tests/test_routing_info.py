"""What a response tells the website about how it was routed (Phase 5, chat page). No network.

Headers: X-Triage-P-Low, -P-Medium and -Tau (model "auto" only, streams too), X-Triage-Strong-Cost (non-streaming).
"""
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
