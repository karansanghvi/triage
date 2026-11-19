"""CORS (Phase 5): the website calls Triage from the browser with custom headers. No network."""
import re
from pathlib import Path

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from app.api import main as main_module
from app.api.auth import Auth
from app.api.main import CORS_RESPONSE_HEADERS, cors_origins_from_env, create_app
from tests.api_helpers import API_KEY, ASK, FakePredictor, make_deps

SITE = "https://triage.example.com"
KEY_HEADERS = ("x-openai-key, x-anthropic-key, x-groq-key, x-gemini-key, x-triage-quality, "
               "x-triage-prefer-provider, content-type")


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
                   "x-triage-task", "x-triage-verify", "x-triage-prefer-provider", "authorization", "content-type"):
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
    for header in ("x-triage-mode", "x-triage-demo-remaining", "x-triage-model", "x-triage-cost", "retry-after",
                   "x-triage-p-low", "x-triage-p-medium", "x-triage-tau", "x-triage-strong-cost",
                   "x-triage-preferred-provider"):
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
                                                                   "X-Triage-Verify", "X-Triage-Prefer-Provider"}
    assert sent <= set(CORS_RESPONSE_HEADERS), sent - set(CORS_RESPONSE_HEADERS)


# ---------- unhandled errors: a JSON 500 the browser can read ----------

SECRET = "sk-test-SECRET123"


@pytest.fixture
def broken_router(monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError(f"something broke near {SECRET}")
    monkeypatch.setattr(main_module.core, "route_and_stream", boom)


def test_an_unhandled_error_is_a_json_500_with_cors_headers(broken_router, caplog):
    http = app([SITE])
    r = http.post("/v1/chat/completions", json={"model": "auto", "messages": ASK},
                  headers={"Origin": SITE, "Authorization": f"Bearer {API_KEY}"})
    assert r.status_code == 500 and r.headers["access-control-allow-origin"] == SITE
    assert r.json()["error"]["code"] == "internal_error" and r.json()["error"]["type"] == "server_error"
    assert r.headers["x-triage-mode"] == "api" and r.headers["x-triage-request-id"].startswith("chatcmpl-triage-")
    assert "broke" not in r.text and SECRET not in r.text                 # the exception's text stays out
    assert "RuntimeError" in caplog.text and SECRET not in caplog.text    # logged, redacted


def test_the_json_500_works_without_cors_too(broken_router):
    r = app([]).post("/v1/chat/completions", json={"model": "auto", "messages": ASK},
                     headers={"Authorization": f"Bearer {API_KEY}"})
    assert r.status_code == 500 and r.json()["error"]["code"] == "internal_error"
    assert "access-control-allow-origin" not in r.headers


# ---------- every endpoint's method passes a preflight ----------

def every_route(routes, prefix=""):
    """(path, method) for each endpoint, including routers added with include_router (which FastAPI may wrap)."""
    for route in routes:
        inner = getattr(route, "original_router", None)
        if inner is not None:
            context = getattr(route, "include_context", None)
            yield from every_route(inner.routes, prefix + (getattr(context, "prefix", "") or ""))
        elif isinstance(route, APIRoute):
            for method in sorted(route.methods - {"HEAD"}):
                yield prefix + route.path, method


def api_app():
    deps = make_deps(predictor=FakePredictor(0.95, 0.95))
    return create_app(deps=deps, auth=Auth([API_KEY]), public=True, cors_origins=[SITE], accounts=True)


ROUTES = sorted(set(every_route(api_app().routes)))


def test_the_route_list_covers_the_whole_api():
    """If FastAPI changes how included routers are stored, the walk above must not silently find fewer routes."""
    application = api_app()
    documented = {(path, method.upper()) for path, item in application.openapi()["paths"].items() for method in item}
    assert documented <= set(ROUTES), documented - set(ROUTES)
    paths = {path for path, _ in ROUTES}
    assert any(p.startswith("/v1/account") for p in paths) and any(p.startswith("/v1/auth") for p in paths)
    assert ("/v1/account/provider-keys/{provider}", "PUT") in ROUTES


@pytest.mark.parametrize("path, method", ROUTES, ids=[f"{m} {p}" for p, m in ROUTES])
def test_every_endpoint_method_passes_a_browser_preflight(path, method):
    """A new endpoint with a method CORS doesn't allow fails here: the browser would block it with a network error."""
    url = re.sub(r"\{[^}]+\}", "x", path)
    r = TestClient(api_app()).options(url, headers={"Origin": SITE, "Access-Control-Request-Method": method,
                                                    "Access-Control-Request-Headers": "authorization, content-type"})
    assert r.status_code == 200, f"{method} {path}: {r.text}"
    assert r.headers["access-control-allow-origin"] == SITE
    allowed = {m.strip() for m in r.headers["access-control-allow-methods"].split(",")}
    assert method in allowed


def test_a_method_no_endpoint_uses_is_still_refused():
    r = TestClient(api_app()).options("/v1/account", headers={"Origin": SITE, "Access-Control-Request-Method": "TRACE"})
    assert r.status_code == 400
