"""The OpenAI-compatible API (Phase 4 Step 5): fake models only, no network.

The official openai SDK calls Triage in-process: the sync client uses FastAPI's TestClient as its HTTP client,
the async client an httpx.AsyncClient on an ASGITransport. route_and_stream is swapped for the reference
version in tests/api_helpers.py, so these tests cover the API around it.
"""
import asyncio

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from app.api import core
from app.api import deps as deps_module
from app.api.auth import Auth
from app.api.config_state import NOT_PERSISTED_NOTE
from app.api.fake_models import FakeBehavior, FakeModels, fake_text
from app.api.main import create_app
from app.cache import ResponseCache
from app.providers.common import ProviderError
from app.reliability import DEFAULT_ROUTING_PATH, ReliableClient, load_routing, routing_from_dict
from app.registry import load_registry
from tests.api_helpers import (ADMIN_KEY, API_KEY, ASK, OTHER_KEY, FakePredictor, make_deps,
                               reference_route_and_stream, routing_raw)

AUTH = Auth([API_KEY, OTHER_KEY], [ADMIN_KEY])


@pytest.fixture(autouse=True)
def reference_core(monkeypatch):
    monkeypatch.setattr(core, "route_and_stream", reference_route_and_stream)


def make_app(script=None, **kw):
    deps = make_deps(script, **kw)
    return create_app(deps=deps, auth=AUTH), deps


@pytest.fixture
def app_deps(tmp_path):
    return make_app(cache=ResponseCache(tmp_path / "cache.sqlite3"), predictor=FakePredictor(0.4, 0.95))


@pytest.fixture
def http(app_deps):
    return TestClient(app_deps[0])


def sdk(app, key=API_KEY, headers=None):
    return openai.OpenAI(base_url="http://testserver/v1", api_key=key, http_client=TestClient(app),
                         default_headers=headers or {}, max_retries=0)


def post(http, body=None, key=API_KEY, headers=None):
    h = {"Authorization": f"Bearer {key}"} if key else {}
    return http.post("/v1/chat/completions", json=body or {"model": "auto", "messages": ASK}, headers=h | (headers or {}))


def auth(key=API_KEY):
    return {"Authorization": f"Bearer {key}"}


# ---------- the openai SDK, in-process ----------

def test_sdk_auto_non_streaming(app_deps):
    app, deps = app_deps
    raw = sdk(app).chat.completions.with_raw_response.create(model="auto", messages=ASK)
    completion = raw.parse()
    assert completion.choices[0].message.content == fake_text("groq-small")     # p_low 0.4 < 0.86: medium
    assert completion.model == "groq-small" and completion.usage.total_tokens > 0
    assert raw.headers["X-Triage-Tier"] == "medium" and raw.headers["X-Triage-Model"] == "groq-small"


def test_sdk_auto_streaming_with_usage(app_deps):
    app, _ = app_deps
    chunks = list(sdk(app).chat.completions.create(model="auto", messages=ASK, stream=True,
                                                   stream_options={"include_usage": True}))
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert text == fake_text("groq-small")
    assert chunks[0].choices[0].delta.role == "assistant"
    assert [c.choices[0].finish_reason for c in chunks if c.choices][-1] == "stop"
    assert chunks[-1].choices == [] and chunks[-1].usage.completion_tokens > 0
    assert len({c.id for c in chunks}) == 1


def test_sdk_streaming_without_usage_has_no_usage_chunk(app_deps):
    chunks = list(sdk(app_deps[0]).chat.completions.create(model="auto", messages=ASK, stream=True))
    assert all(c.choices for c in chunks) and all(c.usage is None for c in chunks)


@pytest.mark.parametrize("stream", [False, True])
def test_sdk_pass_through_model(app_deps, stream):
    app, deps = app_deps
    client = sdk(app)
    if stream:
        text = "".join(c.choices[0].delta.content or "" for c in
                       client.chat.completions.create(model="groq-large", messages=ASK, stream=True) if c.choices)
    else:
        text = client.chat.completions.create(model="groq-large", messages=ASK).choices[0].message.content
    assert text == fake_text("groq-large")
    assert deps.predictor.calls == []


def test_async_sdk_streaming_and_not(app_deps):
    app, _ = app_deps

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as h:
            client = openai.AsyncOpenAI(base_url="http://testserver/v1", api_key=API_KEY, http_client=h, max_retries=0)
            done = await client.chat.completions.create(model="auto", messages=ASK)
            stream = await client.chat.completions.create(model="groq-allam", messages=ASK, stream=True)
            text = "".join([c.choices[0].delta.content or "" async for c in stream if c.choices])
            return done.choices[0].message.content, text
    full, streamed = asyncio.run(go())
    assert full == fake_text("groq-small") and streamed == fake_text("groq-allam")


def test_sdk_errors_map_to_openai_exceptions(app_deps):
    app, _ = app_deps
    with pytest.raises(openai.NotFoundError):
        sdk(app).chat.completions.create(model="no-such-model", messages=ASK)
    with pytest.raises(openai.AuthenticationError):
        sdk(app, key="wrong").chat.completions.create(model="auto", messages=ASK)
    with pytest.raises(openai.BadRequestError):
        sdk(app, headers={"X-Triage-Quality": "max"}).chat.completions.create(model="auto", messages=ASK)


def test_sdk_lists_models(app_deps):
    ids = [m.id for m in sdk(app_deps[0]).models.list()]
    assert ids[0] == "auto" and {"groq-allam", "groq-small", "groq-large"} <= set(ids)


# ---------- response headers ----------

TRIAGE_HEADERS = {"x-triage-request-id", "x-triage-model", "x-triage-tier", "x-triage-reason", "x-triage-fallback",
                  "x-triage-cache", "x-triage-verified", "x-triage-escalated"}


def test_non_streaming_headers_have_real_cost_and_savings(http, app_deps):
    r = post(http)
    assert r.status_code == 200
    h = r.headers
    assert TRIAGE_HEADERS | {"x-triage-cost", "x-triage-saved"} <= set(h)
    assert h["x-triage-request-id"] == r.json()["id"]
    assert (h["x-triage-verified"], h["x-triage-escalated"], h["x-triage-fallback"]) == ("off", "false", "none")
    assert "tau 0.86" in h["x-triage-reason"]
    row = app_deps[1].log.get(h["x-triage-request-id"])
    assert float(h["x-triage-cost"]) == pytest.approx(row.price_as_cost) and row.price_as_cost > 0
    assert float(h["x-triage-saved"]) == pytest.approx(row.strong_cost - row.price_as_cost) and row.saved > 0


def test_cache_header_miss_then_hit_and_bypass(http):
    assert post(http).headers["x-triage-cache"] == "miss"
    second = post(http)
    assert second.headers["x-triage-cache"] == "hit" and float(second.headers["x-triage-cost"]) == 0
    assert float(second.headers["x-triage-saved"]) > 0
    warm = post(http, {"model": "auto", "messages": ASK, "temperature": 0.7})
    assert warm.headers["x-triage-cache"] == "bypass"


def test_streaming_headers_leave_out_cost_and_report_it_by_request_id(http):
    with http.stream("POST", "/v1/chat/completions", headers=auth(),
                     json={"model": "auto", "messages": ASK, "stream": True}) as r:
        body = "".join(r.iter_text())
        h = r.headers
    assert TRIAGE_HEADERS <= set(h)
    assert "x-triage-cost" not in h and "x-triage-saved" not in h
    assert h["x-triage-cache"] == "bypass" and h["content-type"].startswith("text/event-stream")
    assert body.rstrip().endswith("data: [DONE]")
    row = http.get(f"/v1/requests/{h['x-triage-request-id']}", headers=auth()).json()
    assert row["status"] == "ok" and row["price_as_cost"] > 0 and row["saved"] > 0 and row["stream"] is True


def test_fallback_header_and_model(tmp_path):
    app, _ = make_app({"groq-small": FakeBehavior(fail_before="server")}, predictor=FakePredictor(0.4, 0.95))
    r = post(TestClient(app))
    assert r.headers["x-triage-fallback"] == "groq-small->gemini-flash"
    assert r.headers["x-triage-model"] == "gemini-flash" and r.json()["model"] == "gemini-flash"


# ---------- request headers: bad values are 400s ----------

@pytest.mark.parametrize("value", ["max", "1.5", "-0.1", "nan", "abc"])
def test_bad_quality_header_is_400(http, value):
    r = post(http, headers={"X-Triage-Quality": value})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["param"] == "X-Triage-Quality" and "cheap" in err["message"] and err["type"] == "invalid_request_error"


@pytest.mark.parametrize("value,tau", [("cheap", 0.68), ("HIGH", 0.89), ("0.3", 0.3), ("1", 1.0)])
def test_good_quality_header_sets_tau(http, app_deps, value, tau):
    r = post(http, headers={"X-Triage-Quality": value})
    assert r.status_code == 200
    assert app_deps[1].log.get(r.headers["x-triage-request-id"]).tau == tau


def test_task_header_is_normalized_and_bad_ones_are_400(http, app_deps):
    r = post(http, headers={"X-Triage-Task": "  Coding "})
    assert r.status_code == 200 and app_deps[1].predictor.calls[-1][1] == "coding"
    bad = post(http, headers={"X-Triage-Task": "poetry-slam"})
    assert bad.status_code == 400 and bad.json()["error"]["param"] == "X-Triage-Task"


def test_verify_header(http):
    assert post(http, headers={"X-Triage-Verify": "off"}).status_code == 200
    sync = post(http, headers={"X-Triage-Verify": "sync"})
    assert sync.status_code == 501 and "not available yet" in sync.json()["error"]["message"]
    both = post(http, {"model": "auto", "messages": ASK, "stream": True}, headers={"X-Triage-Verify": "sync"})
    assert both.status_code == 400 and both.json()["error"]["code"] == "unsupported_combination"
    assert post(http, headers={"X-Triage-Verify": "later"}).status_code == 400


# ---------- request body ----------

def test_unknown_model_is_an_openai_style_404(http):
    r = post(http, {"model": "gpt-17-ultra", "messages": ASK})
    assert r.status_code == 404
    assert r.json()["error"] == {"message": r.json()["error"]["message"], "type": "invalid_request_error",
                                 "param": "model", "code": "model_not_found"}
    assert "gpt-17-ultra" in r.json()["error"]["message"]


@pytest.mark.parametrize("body,param", [
    ({"model": "auto", "messages": ASK, "tools": [{"type": "function"}]}, "tools"),
    ({"model": "auto", "messages": ASK, "n": 2}, "n"),
    ({"model": "auto", "messages": []}, "messages"),
    ({"model": "auto", "messages": [{"role": "robot", "content": "hi"}]}, "messages.0.role"),
    ({"model": "auto", "messages": ASK, "temperature": 3}, "temperature"),
    ({"messages": ASK}, "model"),
])
def test_bad_bodies_are_400_in_openai_format(http, body, param):
    r = post(http, body)
    assert r.status_code == 400 and r.json()["error"]["param"] == param


def test_all_models_failing_is_503():
    app, deps = make_app({"groq-small": FakeBehavior(fail_before="server"),
                          "gemini-flash": FakeBehavior(fail_before="server")}, predictor=FakePredictor(0.4, 0.95))
    r = post(TestClient(app))
    assert r.status_code == 503 and r.json()["error"]["code"] == "all_models_failed"
    assert deps.log.rows()[-1].status == "error"


def test_mid_stream_failure_ends_the_stream_with_an_error(tmp_path):
    app, deps = make_app({"groq-small": FakeBehavior(fail_after=2)}, predictor=FakePredictor(0.4, 0.95))
    got = []
    with pytest.raises(openai.APIError, match="part-way"):
        for c in sdk(app).chat.completions.create(model="auto", messages=ASK, stream=True):
            got.append(c.choices[0].delta.content or "")
    assert "".join(got) == "Fake answer "                          # two pieces arrived before the error
    row = deps.log.rows()[-1]
    assert row.status == "error" and row.error_kind.startswith("stream_interrupted") and row.cost is None
    assert set(c["model"] for c in deps.fake_models.calls) == {"groq-small"}   # no fallback once text was sent


# ---------- auth ----------

def test_auth_missing_or_wrong_key_is_401_and_never_echoes_the_key(http):
    assert post(http, key=None).status_code == 401
    r = post(http, key="sk-secret-attempt")
    assert r.status_code == 401 and "sk-secret-attempt" not in r.text
    assert http.get("/v1/models").status_code == 401


def test_log_rows_hold_a_key_id_never_the_key(http, app_deps):
    post(http)
    row = app_deps[1].log.rows()[-1]
    assert API_KEY not in repr(row.to_dict()) and len(row.key_id) == 8


def test_admin_endpoints_need_an_admin_key(http):
    assert http.get("/v1/routing-config", headers=auth()).status_code == 403
    assert http.put("/v1/routing-config", headers=auth(), json={"quality": {"default": "cheap"}}).status_code == 403
    assert http.get("/v1/routing-config", headers=auth(ADMIN_KEY)).status_code == 200


def test_no_keys_configured_refuses_to_start(monkeypatch):
    monkeypatch.delenv("TRIAGE_API_KEYS", raising=False)
    monkeypatch.delenv("TRIAGE_ADMIN_KEYS", raising=False)
    with pytest.raises(RuntimeError, match="no API keys"):
        with TestClient(create_app(deps=make_deps())):
            pass


# ---------- live routing config ----------

def test_routing_config_update_applies_without_restart(http, app_deps):
    r = http.put("/v1/routing-config", headers=auth(ADMIN_KEY),
                 json={"quality": {"presets": {"cheap": 0.3, "balanced": 0.86, "high": 0.89}}})
    assert r.status_code == 200
    out = r.json()
    assert out["version"] == 2 and out["persisted"] is False and out["note"] == NOT_PERSISTED_NOTE
    assert out["config"]["quality"]["presets"]["cheap"] == 0.3
    cheap = post(http, headers={"X-Triage-Quality": "cheap"})        # p_low 0.4 >= 0.3: now low
    assert cheap.headers["x-triage-tier"] == "low"


def test_invalid_config_update_is_rejected_and_the_old_one_kept(http, app_deps):
    before = app_deps[1].config.snapshot()
    for patch in ({"quality": {"presets": {"cheap": 1.5}}}, {"quality": {"default": "turbo"}},
                  {"fallback_chains": {"medium": ["groq-small", "no-such-model"]}},
                  {"fallback_chains": {"medium": ["groq-small", "groq-small"]}},
                  {"timeout_seconds": 1}, {}, {"quality": "cheap"}):
        r = http.put("/v1/routing-config", headers=auth(ADMIN_KEY), json=patch)
        assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_config", patch
    assert app_deps[1].config.snapshot() is before


def test_first_model_of_a_chain_cannot_change(http, app_deps):
    r = http.put("/v1/routing-config", headers=auth(ADMIN_KEY),
                 json={"fallback_chains": {"medium": ["gemini-flash", "groq-small"]}})
    assert r.status_code == 400
    msg = r.json()["error"]["message"]
    assert "groq-small" in msg and "classifier" in msg and "exact model" in msg


def test_backup_models_can_change_and_the_client_uses_them():
    # a client error: not retried and doesn't trip the provider-wide breaker, so the groq backup is tried
    app, deps = make_app({"groq-small": FakeBehavior(fail_before="client")}, predictor=FakePredictor(0.4, 0.95))
    http = TestClient(app)
    r = http.put("/v1/routing-config", headers=auth(ADMIN_KEY),
                 json={"fallback_chains": {"medium": ["groq-small", "groq-large"]}})
    assert r.status_code == 200 and deps.client.routing.chains["medium"] == ["groq-small", "groq-large"]
    assert post(http).headers["x-triage-model"] == "groq-large"


def test_config_update_is_validated_like_startup():
    registry = load_registry()
    raw = routing_raw()
    assert routing_from_dict(raw, registry) == load_routing(registry, DEFAULT_ROUTING_PATH)


# ---------- models, stats, requests ----------

def test_models_lists_auto_and_prices(http):
    data = http.get("/v1/models", headers=auth()).json()
    assert data["object"] == "list" and data["data"][0]["id"] == "auto"
    small = next(m for m in data["data"] if m["id"] == "groq-small")
    assert small["triage"]["tier"] == "medium"
    assert (small["triage"]["price_per_1m_input_usd"], small["triage"]["price_per_1m_output_usd"]) == (0.075, 0.30)


def test_stats_summarize_savings_per_key(http):
    post(http)
    post(http, {"model": "groq-large", "messages": ASK})
    post(http, key=OTHER_KEY)
    mine = http.get("/v1/stats", headers=auth()).json()
    assert mine["scope"] == "your key" and mine["requests"] == 2 and mine["ok"] == 2
    assert mine["tier_mix"] == {"high": 0.5, "medium": 0.5}
    assert mine["strong_model_cost_usd"] >= mine["cost_usd"] > 0
    assert mine["saved_usd"] == pytest.approx(mine["strong_model_cost_usd"] - mine["cost_usd"])
    everyone = http.get("/v1/stats", headers=auth(ADMIN_KEY)).json()
    assert everyone["scope"] == "all keys" and everyone["requests"] == 3


def test_request_lookup_is_scoped_to_the_callers_key(http):
    rid = post(http).headers["x-triage-request-id"]
    assert http.get(f"/v1/requests/{rid}", headers=auth()).status_code == 200
    assert http.get(f"/v1/requests/{rid}", headers=auth(OTHER_KEY)).status_code == 404
    assert http.get("/v1/requests/nope", headers=auth()).status_code == 404


def test_log_row_fields(http, app_deps):
    rid = post(http, headers={"X-Triage-Task": "qa"}).headers["x-triage-request-id"]
    row = app_deps[1].log.get(rid).to_dict()
    assert row["requested_model"] == "auto" and row["tier"] == "medium" and row["served_model"] == "groq-small"
    assert row["classifier"] == "classifier_c_logreg_cv v2" and row["config_version"] == 1
    assert (row["p_low"], row["p_medium"]) == (0.4, 0.95) and len(row["prompt_hmac"]) == 64
    assert "prompt_sha256" not in row
    for later in ("checker_verdict", "decided_by", "checker_version", "mode"):
        assert later in row and row[later] is None


# ---------- settings_match_training ----------

def test_settings_match_training_is_logged_for_auto_only(http, app_deps):
    log = app_deps[1].log

    def row_for(body):
        return log.get(post(http, body).headers["x-triage-request-id"])
    default = row_for({"model": "auto", "messages": ASK})
    assert (default.temperature, default.max_tokens, default.settings_match_training) == (0.0, 4000, True)
    warm = row_for({"model": "auto", "messages": ASK, "temperature": 0.5})
    assert warm.settings_match_training is False and warm.temperature == 0.5
    short = row_for({"model": "auto", "messages": ASK, "max_tokens": 256})
    assert short.settings_match_training is False and short.max_tokens == 256
    assert app_deps[1].fake_models.calls[-1]["max_tokens"] == 256                # honoured, not rejected
    assert row_for({"model": "groq-small", "messages": ASK, "temperature": 0.5}).settings_match_training is None


# ---------- startup: predictor loaded once, safety checks ----------

def test_startup_refuses_a_predictor_that_fails_its_checks(monkeypatch):
    import app.router.predictor as predictor
    monkeypatch.setattr(predictor, "load_classifier_config", lambda: ("classifier_c_logreg_cv", 99))
    with pytest.raises(RuntimeError, match="refusing to start"):
        deps_module.load_predictor_checked()


def test_startup_loads_the_predictor_once_with_fake_models(monkeypatch, tmp_path):
    monkeypatch.setenv("TRIAGE_LOG_DB", str(tmp_path / "requests.sqlite3"))     # never the real data/logs
    monkeypatch.setenv("TRIAGE_HASH_SECRET", "startup-test-secret")
    loads = []
    monkeypatch.setattr(deps_module, "load_predictor_checked",
                        lambda: loads.append(1) or (FakePredictor(), "fake v0"))
    monkeypatch.setenv("TRIAGE_FAKE_MODELS", "1")
    monkeypatch.setenv("TRIAGE_CACHE", "off")
    monkeypatch.setenv("TRIAGE_API_KEYS", API_KEY)
    with TestClient(create_app()) as http:
        for _ in range(3):
            assert post(http).status_code == 200
        assert isinstance(http.app.state.deps.fake_models, FakeModels)
    assert loads == [1]


def test_real_predictor_loads_and_passes_its_checks():
    predictor, label = deps_module.load_predictor_checked()
    assert label == "classifier_c_logreg_cv v2"
    p = predictor.predict_messages(ASK)
    assert 0 <= p.p_low <= p.p_medium <= 1


# ---------- reliability: the first-token limit and pass-through ----------

def client_with(script, first_token=None, no_limit=False):
    registry = load_registry()
    raw = routing_raw(first_token)
    if no_limit:
        raw.pop("streaming", None)
    return ReliableClient(registry, routing_from_dict(raw, registry),
                          send_fn=FakeModels(script), sleep=lambda s: None)


def test_first_token_limit_in_the_reliable_client():
    client = client_with({"groq-small": FakeBehavior(first_token_delay=0.5)}, first_token=0.05)
    result = client.send(ASK, "medium", stream=True)
    assert result.model.name == "gemini-flash"
    assert [a.error_kind for a in result.attempts if a.outcome == "error"] == ["timeout"]


def test_no_first_token_limit_means_wait():
    client = client_with({"groq-small": FakeBehavior(first_token_delay=0.1)}, no_limit=True)
    assert client.routing.first_token_timeout_for("groq-small") is None
    assert client.send(ASK, "medium", stream=True).model.name == "groq-small"


def test_routing_yaml_first_token_limits():
    routing = load_routing(load_registry())
    assert routing.first_token_timeout_for("groq-small") == 8
    assert routing.first_token_timeout_for("groq-large") == 10
    assert routing.first_token_timeout_for("ollama-llama") == 32


@pytest.mark.parametrize("streaming", [{"first_token_timeout_seconds": 0}, {"bogus": 1},
                                       {"model_first_token_timeouts": {"nope": 3}}])
def test_bad_streaming_config_is_rejected(streaming):
    raw = routing_raw()
    raw["streaming"] = streaming
    with pytest.raises(ValueError):
        routing_from_dict(raw, load_registry())


def test_send_model_is_one_model_without_fallback():
    client = client_with({"groq-small": FakeBehavior(fail_before="client")})
    with pytest.raises(Exception, match="groq-small"):
        client.send_model(ASK, "groq-small")
    assert client.send_model(ASK, "groq-large").model.name == "groq-large"
    with pytest.raises(ValueError):
        client.send_model(ASK, "nope")


def test_mid_stream_error_reaches_the_caller_of_the_reliable_client():
    client = client_with({"groq-small": FakeBehavior(fail_after=1)})
    result = client.send(ASK, "medium", stream=True)
    with pytest.raises(ProviderError):
        list(result.response)
    assert result.model.name == "groq-small"
