"""Public access (Phase 5, demo mode): which request mode each request gets, and auth. No network.

    api   valid Triage key, no provider keys    server keys, no demo limits (unchanged)
    byok  provider key headers                  the user's keys, with or without a Triage key
    demo  no Triage key, no provider keys       server keys, demo limits (only with TRIAGE_PUBLIC on)
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.auth import Auth, public_from_env
from app.api.main import create_app
from app.router.predictor import Prediction
from tests.api_helpers import ADMIN_KEY, API_KEY, ASK, FakePredictor, make_deps

SECRET = "sk-test-SECRET123"
GROQ_KEY = {"x-groq-key": SECRET}
BEARER = {"Authorization": f"Bearer {API_KEY}"}
BAD_BEARER = {"Authorization": "Bearer not-a-real-triage-key"}


class VectorPredictor(FakePredictor):
    """Returns the classifier inputs too, so a test can see whether they would be stored."""

    def predict_messages(self, messages, task_type=None):
        p = super().predict_messages(messages, task_type)
        return Prediction(p.p_low, p.p_medium, p.raw_p_medium, p.task_type, p.task_type_guessed,
                          features=tuple([0.0] * 14), embedding=np.zeros(384, dtype="float32"))


def app(public=True, **kw):
    deps = make_deps(predictor=VectorPredictor(0.95, 0.95), **kw)
    return TestClient(create_app(deps=deps, auth=Auth([API_KEY], [ADMIN_KEY]), public=public)), deps


def chat(http, headers=None, body=None):
    return http.post("/v1/chat/completions", json=body or {"model": "auto", "messages": ASK}, headers=headers or {})


# ---------- the switch ----------

@pytest.mark.parametrize("value, on", [(None, False), ("", False), ("off", False), ("on", True), ("1", True),
                                       ("TRUE", True), ("yes", True), ("no", False)])
def test_public_switch_from_env(monkeypatch, value, on):
    if value is None:
        monkeypatch.delenv("TRIAGE_PUBLIC", raising=False)
    else:
        monkeypatch.setenv("TRIAGE_PUBLIC", value)
    assert public_from_env() is on


@pytest.mark.parametrize("headers", [{}, GROQ_KEY])
def test_public_off_still_needs_a_triage_key(headers):
    http, deps = app(public=False)
    r = chat(http, headers)
    assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_api_key"
    assert deps.fake_models.calls == [] and deps.log.rows() == []


# ---------- choosing the mode ----------

@pytest.mark.parametrize("public", [True, False])
def test_valid_key_without_provider_keys_is_api(public):
    http, deps = app(public=public)
    r = chat(http, BEARER)
    assert r.status_code == 200 and r.headers["x-triage-mode"] == "api"
    (row,) = deps.log.rows()
    assert row.mode == "api" and row.key_id != "demo" and len(row.key_id) == 8


def test_no_key_at_all_is_demo():
    http, deps = app()
    r = chat(http)
    assert r.status_code == 200 and r.headers["x-triage-mode"] == "demo"
    assert deps.fake_models.last_api_key is None                  # server keys
    (row,) = deps.log.rows()
    assert (row.mode, row.key_id) == ("demo", "demo")


def test_provider_keys_without_a_triage_key_is_public_byok():
    http, deps = app()
    r = chat(http, GROQ_KEY)
    assert r.status_code == 200 and r.headers["x-triage-mode"] == "byok"
    assert deps.fake_models.last_api_key == SECRET
    (row,) = deps.log.rows()
    assert (row.mode, row.key_id) == ("byok", "public")


def test_provider_keys_with_a_triage_key_is_byok_under_that_key():
    http, deps = app()
    r = chat(http, BEARER | GROQ_KEY)
    assert r.headers["x-triage-mode"] == "byok"
    (row,) = deps.log.rows()
    assert row.mode == "byok" and row.key_id not in ("public", "demo")


@pytest.mark.parametrize("headers", [BAD_BEARER, BAD_BEARER | GROQ_KEY, {"Authorization": ""},
                                     {"Authorization": "Basic abc"}, {"Authorization": "Bearer "}])
def test_an_invalid_triage_key_is_401_never_demo(headers):
    http, deps = app()
    r = chat(http, headers)
    assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_api_key"
    assert deps.fake_models.calls == [] and deps.log.rows() == []


def test_errors_carry_the_mode_header():
    http, _ = app()
    r = chat(http, body={"model": "no-such-model", "messages": ASK})
    assert r.status_code == 404 and r.headers["x-triage-mode"] == "demo"
    assert chat(http, BEARER, {"model": "no-such-model", "messages": ASK}).headers["x-triage-mode"] == "api"


@pytest.mark.parametrize("stream", [False, True])
def test_mode_header_on_streams_too(stream):
    http, _ = app()
    r = chat(http, body={"model": "auto", "messages": ASK, "stream": stream})
    assert r.headers["x-triage-mode"] == "demo"


# ---------- everything else still needs a Triage key ----------

@pytest.mark.parametrize("method, path", [("get", "/v1/models"), ("get", "/v1/stats"),
                                          ("get", "/v1/requests/chatcmpl-x"), ("get", "/v1/routing-config")])
def test_other_endpoints_still_need_a_key_with_public_on(method, path):
    http, _ = app()
    assert getattr(http, method)(path).status_code == 401


def test_admin_endpoints_still_need_an_admin_key():
    http, _ = app()
    assert http.get("/v1/routing-config", headers=BEARER).status_code == 403
    assert http.get("/v1/routing-config", headers={"Authorization": f"Bearer {ADMIN_KEY}"}).status_code == 200


# ---------- data rules: demo prompts are strangers' ----------

def test_demo_rows_are_never_training_data_and_store_no_vectors():
    http, deps = app()
    chat(http)
    chat(http, BEARER)
    demo_row, api_row = deps.log.rows()
    assert (demo_row.mode, demo_row.training_eligible, demo_row.features, demo_row.embedding) == \
        ("demo", False, None, None)
    assert api_row.mode == "api" and api_row.training_eligible is True and api_row.embedding is not None


@pytest.mark.parametrize("stream", [False, True])
def test_demo_requests_get_no_background_labels(stream):
    http, deps = app(sample_rate=1.0)
    chat(http, body={"model": "auto", "messages": ASK, "stream": stream})
    assert deps.log.labels() == []
    chat(http, BEARER, {"model": "auto", "messages": ASK, "stream": stream})
    assert len(deps.log.labels()) == 1
