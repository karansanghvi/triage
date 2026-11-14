"""CORS (Phase 5): the website calls Triage from the browser with custom headers. No network."""
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import main as main_module
from app.api.auth import Auth
from app.api.main import CORS_RESPONSE_HEADERS, cors_origins_from_env, create_app
from tests.api_helpers import API_KEY, ASK, FakePredictor, make_deps

SITE = "https://triage.example.com"
KEY_HEADERS = "x-openai-key, x-anthropic-key, x-groq-key, x-gemini-key, x-triage-quality, content-type"


def app(origins):
    deps = make_deps(predictor=FakePredictor(0.95, 0.95))
    return TestClient(create_app(deps=deps, auth=Auth([API_KEY]), public=True, cors_origins=origins))


def preflight(http, origin=SITE, headers=KEY_HEADERS, method="POST"):
    return http.options("/v1/chat/completions", headers={"Origin": origin, "Access-Control-Request-Method": method,
                                                         "Access-Control-Request-Headers": headers})


@pytest.mark.parametrize("value, origins", [(None, []), ("", []), (f"{SITE}/", [SITE]),
                                            (f"{SITE}, http://localhost:5173", [SITE, "http://localhost:5173"])])
def test_origins_from_env(monkeypatch, value, origins):
    if value is None:
        monkeypatch.delenv("TRIAGE_CORS_ORIGINS", raising=False)
    else:
        monkeypatch.setenv("TRIAGE_CORS_ORIGINS", value)
    assert cors_origins_from_env() == origins


def test_preflight_with_key_headers_is_allowed_for_a_listed_origin():
    r = preflight(app([SITE]))
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == SITE
    allowed = r.headers["access-control-allow-headers"].lower()
    for header in ("x-openai-key", "x-anthropic-key", "x-groq-key", "x-gemini-key", "x-triage-quality",
                   "x-triage-task", "x-triage-verify", "authorization", "content-type"):
        assert header in allowed
    assert "POST" in r.headers["access-control-allow-methods"]
    assert "access-control-allow-credentials" not in r.headers


def test_preflight_from_another_origin_is_refused():
    r = preflight(app([SITE]), origin="https://evil.example.net")
    assert r.status_code == 400 and "access-control-allow-origin" not in r.headers


def test_preflight_with_an_unknown_header_is_refused():
    r = preflight(app([SITE]), headers="x-something-else")
    assert r.status_code == 400


def test_responses_expose_the_triage_headers():
    http = app([SITE])
    r = http.post("/v1/chat/completions", json={"model": "auto", "messages": ASK}, headers={"Origin": SITE})
    assert r.status_code == 200 and r.headers["access-control-allow-origin"] == SITE
    exposed = {h.strip().lower() for h in r.headers["access-control-expose-headers"].split(",")}
    for header in ("x-triage-mode", "x-triage-demo-remaining", "x-triage-model", "x-triage-cost", "retry-after"):
        assert header in exposed


def test_error_responses_carry_cors_headers_too():
    http = app([SITE])
    r = http.post("/v1/chat/completions", json={"model": "nope", "messages": ASK}, headers={"Origin": SITE})
    assert r.status_code == 404 and r.headers["access-control-allow-origin"] == SITE


def test_no_origins_means_no_cross_origin_access():
    http = app([])
    assert "access-control-allow-origin" not in preflight(http).headers
    r = http.post("/v1/chat/completions", json={"model": "auto", "messages": ASK}, headers={"Origin": SITE})
    assert r.status_code == 200 and "access-control-allow-origin" not in r.headers


def test_every_triage_response_header_is_exposed():
    """A new X-Triage-* response header must be added to CORS_RESPONSE_HEADERS, or the website can't read it."""
    source = Path(main_module.__file__).read_text()
    sent = set(re.findall(r'"(X-Triage-[A-Za-z-]+)"', source)) - {"X-Triage-Quality", "X-Triage-Task",
                                                                   "X-Triage-Verify"}
    assert sent <= set(CORS_RESPONSE_HEADERS), sent - set(CORS_RESPONSE_HEADERS)
