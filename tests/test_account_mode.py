"""Account mode (Phase 5, accounts PR 6): chat with a user's trk_ key or session, using their saved provider
keys. No network: fake models, an in-memory SQLite database."""
import logging
import secrets
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app import db
from app.accounts.encryption import KeyRing
from app.accounts.limits import AccountLimiter, AccountLimits
from app.accounts.verify_keys import Verdict
from app.api.auth import Auth
from app.api.fake_models import FakeBehavior
from app.api.main import create_app
from app.api.request_log import SQLRequestLog
from tests.api_helpers import API_KEY, FakePredictor, make_deps

PASSWORD = "correct horse battery"
SAVED_GROQ, SAVED_OPENAI = "gsk_saved_SECRET_groq_1111", "sk-saved-SECRET-openai-2222"
HEADER_GROQ = "gsk_header_SECRET_groq_3333"
ASK = {"model": "auto", "messages": [{"role": "user", "content": "What is 2 + 3?"}]}
ACCEPT = Verdict("valid", None, True, 200)


def app(script=None, engine=None, predictor=None, public=False, account_limits=None, **kw):
    engine = engine or db.open_engine("sqlite://")
    deps = make_deps(script, engine=engine, log=SQLRequestLog(engine), predictor=predictor or FakePredictor(0.95, 0.95),
                     keyring=KeyRing([("k1", secrets.token_bytes(32))]), key_verifier=lambda p, k: ACCEPT,
                     trusted_proxy_hops=1, account_limits=account_limits, **kw)
    return TestClient(create_app(deps=deps, auth=Auth([API_KEY]), accounts=True, public=public)), deps


def signup(http, email="ada@example.com", n=1):
    r = http.post("/v1/auth/signup", json={"email": email, "password": PASSWORD},
                  headers={"X-Forwarded-For": f"203.0.113.{n}"})
    return {"Authorization": f"Bearer {r.json()['session_token']}"}, r.json()["user"]["id"]


def save(http, auth, provider, key):
    assert http.put(f"/v1/account/provider-keys/{provider}", json={"key": key}, headers=auth).status_code == 200


def trk(http, auth, name="laptop"):
    return {"Authorization": f"Bearer {http.post('/v1/account/api-keys', json={'name': name}, headers=auth).json()['key']}"}


def chat(http, headers, body=None, **extra):
    return http.post("/v1/chat/completions", json=body or ASK, headers=headers | extra)


# ---------- chat with a trk_ key or a session ----------

@pytest.mark.parametrize("credential", ["trk", "session"])
def test_chat_runs_in_account_mode_with_the_saved_key(credential):
    http, deps = app()
    auth, user_id = signup(http)
    save(http, auth, "groq", SAVED_GROQ)
    headers = trk(http, auth) if credential == "trk" else auth
    r = chat(http, headers)
    assert r.status_code == 200 and r.headers["x-triage-mode"] == "account" and r.json()["triage"]["mode"] == "account"
    assert deps.fake_models.last_api_key == SAVED_GROQ
    (row,) = deps.log.rows()
    assert (row.mode, row.user_id) == ("account", user_id)
    assert row.key_id == ("session" if credential == "session" else headers["Authorization"][7:19])


def test_header_keys_replace_the_saved_key_for_their_provider():
    http, deps = app()
    auth, _ = signup(http)
    save(http, auth, "groq", SAVED_GROQ)
    r = chat(http, auth, **{"x-groq-key": HEADER_GROQ})
    assert r.status_code == 200 and r.headers["x-triage-mode"] == "account"
    assert deps.fake_models.last_api_key == HEADER_GROQ


def test_header_keys_add_providers_too():
    http, deps = app(predictor=FakePredictor(0.95, 0.95))
    auth, _ = signup(http)
    save(http, auth, "groq", SAVED_GROQ)
    r = chat(http, auth, **{"x-openai-key": "sk-header-openai-4444", "X-Triage-Prefer-Provider": "openai"})
    assert r.json()["model"] == "openai-luna" and deps.fake_models.last_api_key == "sk-header-openai-4444"


def test_no_usable_keys_is_a_400_pointing_to_account_settings():
    http, deps = app()
    auth, user_id = signup(http)
    r = chat(http, auth)
    assert r.status_code == 400 and r.json()["error"]["code"] == "no_provider_keys"
    assert "/v1/account/provider-keys" in r.json()["error"]["message"] and deps.fake_models.calls == []
    save(http, auth, "groq", SAVED_GROQ)                         # an invalid saved key doesn't count either
    deps.provider_keys.set_status(user_id, "groq", "invalid", "rejected", verified=True)
    assert chat(http, auth).json()["error"]["code"] == "no_provider_keys"


def test_unverified_saved_keys_are_used():
    http, deps = app()
    auth, user_id = signup(http)
    save(http, auth, "groq", SAVED_GROQ)
    deps.provider_keys.set_status(user_id, "groq", "unverified", "timeout", verified=False)
    assert chat(http, auth).status_code == 200 and deps.fake_models.last_api_key == SAVED_GROQ


# ---------- a provider rejecting a saved key ----------

def test_a_rejected_saved_key_is_a_401_and_marked_invalid():
    http, deps = app({"groq-allam": FakeBehavior(fail_before="auth", echo_key=True)})
    auth, user_id = signup(http)
    save(http, auth, "groq", SAVED_GROQ)
    r = chat(http, auth, body={"model": "groq-allam", "messages": ASK["messages"]})
    assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_provider_key"
    assert "saved key" in r.json()["error"]["message"] and "account settings" in r.json()["error"]["message"]
    (saved,) = http.get("/v1/account/provider-keys", headers=auth).json()
    assert saved["status"] == "invalid" and saved["status_detail"].startswith("Groq rejected it during a chat request")
    assert SAVED_GROQ not in saved["status_detail"] and SAVED_GROQ not in r.text
    assert chat(http, auth).json()["error"]["code"] == "no_provider_keys"          # skipped from now on


def test_a_saved_key_rejected_before_a_fallback_answered_is_still_marked_invalid():
    http, deps = app({"groq-allam": FakeBehavior(fail_before="auth")})
    auth, _ = signup(http)
    save(http, auth, "groq", SAVED_GROQ)
    save(http, auth, "openai", SAVED_OPENAI)
    r = chat(http, auth)                                          # groq-allam first, then the chain falls back
    assert r.status_code == 200 and r.json()["model"] != "groq-allam"
    statuses = {k["provider"]: k["status"] for k in http.get("/v1/account/provider-keys", headers=auth).json()}
    assert statuses == {"openai": "valid", "groq": "invalid"}


def test_a_rejected_header_key_never_touches_the_saved_one():
    http, _ = app({"groq-allam": FakeBehavior(fail_before="auth")})
    auth, _ = signup(http)
    save(http, auth, "groq", SAVED_GROQ)
    r = chat(http, auth, body={"model": "groq-allam", "messages": ASK["messages"]}, **{"x-groq-key": HEADER_GROQ})
    assert r.status_code == 401 and "X-Groq-Key" in r.json()["error"]["message"]
    assert http.get("/v1/account/provider-keys", headers=auth).json()[0]["status"] == "valid"


# ---------- preferred provider ----------

def test_the_saved_preference_is_the_default_and_the_header_overrides_it():
    http, deps = app()
    auth, _ = signup(http)
    save(http, auth, "groq", SAVED_GROQ)
    save(http, auth, "openai", SAVED_OPENAI)
    assert chat(http, auth).json()["model"] == "groq-allam"                        # no preference: as usual
    http.patch("/v1/account", json={"preferred_provider": "openai"}, headers=auth)
    r = chat(http, auth)
    assert r.json()["model"] == "openai-luna" and r.headers["x-triage-preferred-provider"] == "openai"
    r = chat(http, auth, **{"X-Triage-Prefer-Provider": "groq"})
    assert r.json()["model"] == "groq-allam" and r.headers["x-triage-preferred-provider"] == "groq"
    (_, saved_pref, header_pref) = deps.log.rows()
    assert (saved_pref.preferred_provider, header_pref.preferred_provider) == ("openai", "groq")


def test_a_saved_preference_without_a_key_for_it_is_ignored():
    http, _ = app()
    auth, _ = signup(http)
    save(http, auth, "groq", SAVED_GROQ)
    http.patch("/v1/account", json={"preferred_provider": "anthropic"}, headers=auth)
    r = chat(http, auth)
    assert r.status_code == 200 and r.headers["x-triage-preferred-provider"] == "none"


def test_a_named_model_ignores_the_saved_preference():
    http, _ = app()
    auth, _ = signup(http)
    save(http, auth, "groq", SAVED_GROQ)
    save(http, auth, "openai", SAVED_OPENAI)
    http.patch("/v1/account", json={"preferred_provider": "openai"}, headers=auth)
    r = chat(http, auth, body={"model": "groq-small", "messages": ASK["messages"]})
    assert r.json()["model"] == "groq-small" and r.headers["x-triage-preferred-provider"] == "none"


# ---------- limits ----------

def test_sixty_a_minute_then_429():
    http, deps = app(account_limits=AccountLimits(requests_per_minute=3, requests_per_day=100))
    auth, _ = signup(http)
    save(http, auth, "groq", SAVED_GROQ)
    for _ in range(3):
        assert chat(http, auth).status_code == 200
    r = chat(http, auth)
    assert r.status_code == 429 and r.json()["error"]["code"] == "account_rate_limited"
    assert 0 < int(r.headers["retry-after"]) <= 60 and "minute" in r.json()["error"]["message"]
    assert len(deps.fake_models.calls) == 3 and deps.log.rows()[-1].error_kind == "account_rate_limited"


def test_ten_thousand_a_day_then_429_until_midnight():
    now = [datetime(2026, 11, 16, 23, 0, tzinfo=timezone.utc)]
    http, deps = app()
    deps.account_limiter = AccountLimiter(AccountLimits(requests_per_minute=100, requests_per_day=2), deps.engine,
                                          clock=lambda: now[0])
    auth, _ = signup(http)
    save(http, auth, "groq", SAVED_GROQ)
    assert chat(http, auth).status_code == 200 and chat(http, auth).status_code == 200
    r = chat(http, auth)
    assert r.status_code == 429 and int(r.headers["retry-after"]) == 3600 and "day" in r.json()["error"]["message"]
    now[0] += timedelta(hours=1)
    assert chat(http, auth).status_code == 200


def test_the_limits_are_per_account_and_there_are_no_demo_limits():
    from app.api.demo import DemoLimiter, DemoSettings
    limiter = DemoLimiter(DemoSettings(visitor_daily_requests=1, public_byok_visitor_daily_requests=1))
    http, deps = app(account_limits=AccountLimits(requests_per_minute=2, requests_per_day=100), public=True,
                     demo_limiter=limiter)
    ada, _ = signup(http)
    bob, _ = signup(http, "bob@example.com", 2)
    for auth in (ada, bob):
        save(http, auth, "groq", SAVED_GROQ)
        assert chat(http, auth).status_code == 200 and chat(http, auth).status_code == 200   # 2 > demo's 1
    assert chat(http, ada).status_code == 429
    assert limiter.status("whoever").used == 0                                    # never counted as demo


# ---------- per-user isolation ----------

def test_users_see_only_their_own_requests_and_stats():
    http, deps = app()
    ada, ada_id = signup(http)
    bob, _ = signup(http, "bob@example.com", 2)
    for auth in (ada, bob):
        save(http, auth, "groq", SAVED_GROQ)
    ada_key = trk(http, ada)
    from_key = chat(http, ada_key).headers["x-triage-request-id"]
    from_session = chat(http, ada).headers["x-triage-request-id"]
    bobs = chat(http, bob).headers["x-triage-request-id"]
    env_row = chat(http, {"Authorization": f"Bearer {API_KEY}"}).headers["x-triage-request-id"]
    for credential in (ada_key, ada, trk(http, ada, "another")):           # any of ada's credentials
        assert http.get(f"/v1/requests/{from_key}", headers=credential).json()["user_id"] == ada_id
        assert http.get(f"/v1/requests/{from_session}", headers=credential).status_code == 200
        for other in (bobs, env_row):
            assert http.get(f"/v1/requests/{other}", headers=credential).status_code == 404
        assert http.get("/v1/stats", headers=credential).json()["requests"] == 2
    assert http.get(f"/v1/requests/{bobs}", headers=bob).status_code == 200
    env = {"Authorization": f"Bearer {API_KEY}"}
    assert http.get(f"/v1/requests/{env_row}", headers=env).status_code == 200
    assert http.get(f"/v1/requests/{from_key}", headers=env).status_code == 404    # an env key can't see accounts'


def test_a_revoked_key_is_401_and_a_logged_out_session_too():
    http, _ = app()
    auth, _ = signup(http)
    save(http, auth, "groq", SAVED_GROQ)
    created = http.post("/v1/account/api-keys", json={"name": "x"}, headers=auth).json()
    http.delete(f"/v1/account/api-keys/{created['id']}", headers=auth)
    r = chat(http, {"Authorization": f"Bearer {created['key']}"})
    assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_api_key"
    http.post("/v1/auth/logout", headers=auth)
    assert chat(http, auth).status_code == 401


def test_using_a_key_for_chat_records_last_used():
    http, _ = app()
    auth, _ = signup(http)
    save(http, auth, "groq", SAVED_GROQ)
    chat(http, trk(http, auth))
    assert http.get("/v1/account/api-keys", headers=auth).json()[0]["last_used_at"] is not None


# ---------- data rules ----------

@pytest.mark.parametrize("stream", [False, True])
def test_account_rows_are_never_training_data_vectors_or_labels(stream):
    http, deps = app(sample_rate=1.0)
    auth, _ = signup(http)
    save(http, auth, "groq", SAVED_GROQ)
    body = ASK | {"stream": stream}
    if stream:
        with http.stream("POST", "/v1/chat/completions", json=body, headers=auth) as r:
            "".join(r.iter_text())
    else:
        chat(http, auth, body=body)
    (row,) = deps.log.rows()
    assert (row.mode, row.training_eligible, row.features, row.embedding) == ("account", False, None, None)
    assert deps.log.labels() == []


def test_decrypted_keys_appear_nowhere(tmp_path, caplog):
    engine = db.open_engine(db.sqlite_url(tmp_path / "t.sqlite3"))
    http, deps = app({"groq-allam": FakeBehavior(fail_before="auth", echo_key=True)}, engine=engine)
    texts = []
    with caplog.at_level(logging.DEBUG):
        auth, _ = signup(http)
        save(http, auth, "groq", SAVED_GROQ)
        save(http, auth, "openai", SAVED_OPENAI)
        for r in (chat(http, auth), chat(http, trk(http, auth)),
                  chat(http, auth, body={"model": "groq-allam", "messages": ASK["messages"]}),
                  http.get("/v1/stats", headers=auth), http.get("/v1/account/provider-keys", headers=auth)):
            texts.append(r.text)
        for row in deps.log.rows():
            texts.append(http.get(f"/v1/requests/{row.request_id}", headers=auth).text)
    for secret in (SAVED_GROQ, SAVED_OPENAI):
        for text in texts + [caplog.text]:
            assert secret not in text and "SECRET" not in text
        assert all(secret not in str(row.to_dict()) for row in deps.log.rows())
    assert not any(isinstance(v, str) and "SECRET" in v for v in vars(deps).values())
    engine.dispose()
    raw = b"".join(p.read_bytes() for p in tmp_path.iterdir() if p.name.startswith("t.sqlite3"))
    assert b"SECRET" not in raw


# ---------- everything else unchanged ----------

def test_env_keys_byok_and_demo_are_unchanged():
    http, deps = app(public=True)
    assert chat(http, {"Authorization": f"Bearer {API_KEY}"}).headers["x-triage-mode"] == "api"
    assert chat(http, {"x-groq-key": HEADER_GROQ}).headers["x-triage-mode"] == "byok"
    assert chat(http, {}).headers["x-triage-mode"] == "demo"
    assert [(r.mode, r.user_id) for r in deps.log.rows()] == [("api", None), ("byok", None), ("demo", None)]


def test_with_accounts_off_trk_and_trs_are_unknown_keys():
    engine = db.open_engine("sqlite://")
    deps = make_deps(engine=engine, log=SQLRequestLog(engine))
    http = TestClient(create_app(deps=deps, auth=Auth([API_KEY]), accounts=False))
    for token in ("trk_" + "a" * 43, "trs_" + "a" * 43):
        assert chat(http, {"Authorization": f"Bearer {token}"}).status_code == 401
