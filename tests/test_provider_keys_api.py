"""Saved provider keys over HTTP (/v1/account/provider-keys). No network: verification goes to a fake or to
verify_key with an httpx MockTransport."""
import base64
import logging
import secrets

import httpx
import pytest
from fastapi.testclient import TestClient

from app import db
from app.accounts.encryption import EncryptionConfigError, KeyRing
from app.accounts.verify_keys import Verdict, verify_key
from app.api.auth import Auth
from app.api.main import create_app
from app.api.request_log import SQLRequestLog
from tests.api_helpers import API_KEY, make_deps

SECRET = "sk-test-SECRET123"
PASSWORD = "correct horse battery"
RAW_KEY = secrets.token_bytes(32)
RING = KeyRing([("k1", RAW_KEY)])


class FakeVerifier:
    def __init__(self, verdict=Verdict("valid", None, True)):
        self.verdict, self.calls = verdict, []

    def __call__(self, provider, key):
        self.calls.append((provider, key))
        return self.verdict


def app(verifier=None, engine=None, keyring=RING, **kw):
    engine = engine or db.open_engine("sqlite://")
    deps = make_deps(engine=engine, log=SQLRequestLog(engine), keyring=keyring,
                     key_verifier=verifier or FakeVerifier(), trusted_proxy_hops=1, **kw)
    http = TestClient(create_app(deps=deps, auth=Auth([API_KEY]), accounts=True))
    return http, deps


def session(http, email="ada@example.com", n=1):
    r = http.post("/v1/auth/signup", json={"email": email, "password": PASSWORD},
                  headers={"X-Forwarded-For": f"203.0.113.{n}"})
    return {"Authorization": f"Bearer {r.json()['session_token']}"}


def put(http, auth, provider="openai", key=SECRET):
    return http.put(f"/v1/account/provider-keys/{provider}", json={"key": key}, headers=auth)


# ---------- save, list, delete ----------

def test_save_then_list_shows_only_the_summary():
    http, deps = app()
    auth = session(http)
    r = put(http, auth, key=f"  {SECRET}\n")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"provider", "last4", "status", "verified_at", "created_at", "updated_at"}
    assert (body["provider"], body["last4"], body["status"]) == ("openai", "T123", "valid") and body["verified_at"]
    assert http.get("/v1/account/provider-keys", headers=auth).json() == [body]
    assert deps.key_verifier.calls == [("openai", SECRET)]                    # trimmed before checking


def test_a_rejected_key_is_400_and_nothing_is_stored():
    http, deps = app(FakeVerifier(Verdict("invalid", "OpenAI answered 401: Incorrect API key provided", True)))
    auth = session(http)
    r = put(http, auth)
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_provider_key"
    assert "401" in r.json()["error"]["message"]
    assert http.get("/v1/account/provider-keys", headers=auth).json() == []


def test_a_rejected_key_leaves_an_earlier_saved_key_alone():
    verifier = FakeVerifier()
    http, deps = app(verifier)
    auth = session(http)
    put(http, auth, key="sk-test-FIRSTKEY-aaaa")
    verifier.verdict = Verdict("invalid", "OpenAI answered 401", True)
    assert put(http, auth).status_code == 400
    (saved,) = http.get("/v1/account/provider-keys", headers=auth).json()
    assert saved["last4"] == "aaaa"


@pytest.mark.parametrize("detail", ["OpenAI didn't answer within 10 s", "OpenAI answered 503", "OpenAI answered 429"])
def test_no_clear_answer_is_stored_as_unverified(detail):
    http, _ = app(FakeVerifier(Verdict("unverified", detail, False)))
    auth = session(http)
    r = put(http, auth)
    assert r.status_code == 200 and r.json()["status"] == "unverified" and r.json()["verified_at"] is None


def test_saving_again_replaces_and_each_provider_is_separate():
    http, _ = app()
    auth = session(http)
    put(http, auth, "openai", "sk-test-FIRSTKEY-aaaa")
    put(http, auth, "OpenAI", "sk-test-SECONDKEY-bbbb")
    put(http, auth, "groq", "gsk_testkey_cccc")
    assert [(k["provider"], k["last4"]) for k in http.get("/v1/account/provider-keys", headers=auth).json()] == \
        [("openai", "bbbb"), ("groq", "cccc")]


def test_delete_is_204_and_idempotent():
    http, _ = app()
    auth = session(http)
    put(http, auth)
    assert http.delete("/v1/account/provider-keys/openai", headers=auth).status_code == 204
    assert http.delete("/v1/account/provider-keys/openai", headers=auth).status_code == 204
    assert http.get("/v1/account/provider-keys", headers=auth).json() == []


@pytest.mark.parametrize("provider", ["ollama", "openrouter", "openai2", "%20"])
def test_unknown_providers_are_400(provider):
    http, _ = app()
    auth = session(http)
    for r in (put(http, auth, provider), http.delete(f"/v1/account/provider-keys/{provider}", headers=auth),
              http.post(f"/v1/account/provider-keys/{provider}/verify", headers=auth)):
        assert r.status_code == 400 and r.json()["error"]["param"] == "provider"


@pytest.mark.parametrize("key", ["", "   ", "x" * 513])
def test_empty_or_absurd_keys_are_400_without_calling_the_provider(key):
    http, deps = app()
    auth = session(http)
    r = put(http, auth, key=key)
    assert r.status_code == 400 and r.json()["error"]["param"] == "key" and deps.key_verifier.calls == []


def test_users_only_see_their_own_keys():
    http, _ = app()
    ada, bob = session(http), session(http, "bob@example.com", 2)
    put(http, ada)
    assert http.get("/v1/account/provider-keys", headers=bob).json() == []
    assert http.post("/v1/account/provider-keys/openai/verify", headers=bob).status_code == 404


# ---------- re-verify ----------

def test_verify_rechecks_the_saved_key_and_updates_its_status():
    verifier = FakeVerifier(Verdict("unverified", "OpenAI answered 503", False))
    http, _ = app(verifier)
    auth = session(http)
    put(http, auth)
    verifier.verdict = Verdict("valid", None, True)
    r = http.post("/v1/account/provider-keys/openai/verify", headers=auth)
    assert r.status_code == 200 and r.json()["status"] == "valid" and r.json()["verified_at"]
    assert verifier.calls[-1] == ("openai", SECRET)                         # the decrypted saved key
    verifier.verdict = Verdict("invalid", "OpenAI answered 401", True)
    r = http.post("/v1/account/provider-keys/openai/verify", headers=auth)
    assert r.status_code == 200 and r.json()["status"] == "invalid"        # stays saved, marked invalid


def test_verify_without_a_saved_key_is_404():
    http, deps = app()
    r = http.post("/v1/account/provider-keys/groq/verify", headers=session(http))
    assert r.status_code == 404 and deps.key_verifier.calls == []


def test_a_key_whose_encryption_key_was_removed_is_409():
    engine = db.open_engine("sqlite://")
    http, deps = app(engine=engine)
    auth = session(http)
    put(http, auth)
    other = KeyRing([("k2", secrets.token_bytes(32))])
    deps.keyring = deps.provider_keys.keyring = other                       # k1 dropped from the server
    r = http.post("/v1/account/provider-keys/openai/verify", headers=auth)
    assert r.status_code == 409 and r.json()["error"]["code"] == "key_unreadable"


# ---------- rate limit ----------

def test_saves_and_rechecks_share_twenty_per_user_per_hour():
    http, deps = app()
    ada, bob = session(http), session(http, "bob@example.com", 2)
    for i in range(15):
        assert put(http, ada, key=f"sk-test-KEY-{i:04d}").status_code == 200
    for _ in range(5):
        assert http.post("/v1/account/provider-keys/openai/verify", headers=ada).status_code == 200
    r = put(http, ada)
    assert r.status_code == 429 and r.json()["error"]["code"] == "too_many_attempts" and int(r.headers["retry-after"]) > 0
    assert http.post("/v1/account/provider-keys/openai/verify", headers=ada).status_code == 429
    assert len(deps.key_verifier.calls) == 20                               # the provider wasn't called again
    assert put(http, bob).status_code == 200                                # another user is unaffected
    assert http.get("/v1/account/provider-keys", headers=ada).status_code == 200    # listing isn't limited


# ---------- session only, the switch, startup ----------

@pytest.mark.parametrize("method, path", [("get", "/v1/account/provider-keys"),
                                          ("put", "/v1/account/provider-keys/openai"),
                                          ("post", "/v1/account/provider-keys/openai/verify"),
                                          ("delete", "/v1/account/provider-keys/openai")])
def test_provider_key_endpoints_need_a_session_token(method, path):
    http, deps = app()
    session(http)
    for headers in ({}, {"Authorization": f"Bearer {API_KEY}"}, {"x-openai-key": SECRET}):
        r = http.request(method.upper(), path, json={"key": SECRET}, headers=headers)
        assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_session"
    assert deps.key_verifier.calls == []


def test_accounts_on_refuses_to_start_without_encryption_keys(monkeypatch):
    deps = make_deps()
    with pytest.raises(EncryptionConfigError, match="TRIAGE_ENCRYPTION_KEYS is not set"):
        with TestClient(create_app(deps=deps, auth=Auth([API_KEY]), accounts=True)):
            pass


def test_accounts_on_refuses_a_short_key_without_printing_it(monkeypatch):
    short = base64.b64encode(secrets.token_bytes(16)).decode()
    monkeypatch.setenv("TRIAGE_ENCRYPTION_KEYS", f"k1:{short}")
    with pytest.raises(EncryptionConfigError, match="exactly 32 bytes") as error:
        with TestClient(create_app(deps=make_deps(), auth=Auth([API_KEY]), accounts=True)):
            pass
    assert short not in str(error.value)


def test_accounts_on_starts_with_valid_keys_and_off_needs_none(monkeypatch):
    monkeypatch.setenv("TRIAGE_ENCRYPTION_KEYS", f"k1:{base64.b64encode(RAW_KEY).decode()}")
    deps = make_deps()
    with TestClient(create_app(deps=deps, auth=Auth([API_KEY]), accounts=True)):
        assert deps.keyring.primary_id == "k1" and deps.provider_keys.keyring is deps.keyring
    monkeypatch.delenv("TRIAGE_ENCRYPTION_KEYS")
    with TestClient(create_app(deps=make_deps(), auth=Auth([API_KEY]), accounts=False)):
        pass


# ---------- account deletion ----------

def test_deleting_the_account_deletes_its_provider_keys():
    engine = db.open_engine("sqlite://")
    http, _ = app(engine=engine)
    ada, bob = session(http), session(http, "bob@example.com", 2)
    put(http, ada)
    put(http, bob, "groq", "gsk_bobs_key_dddd")
    assert http.request("DELETE", "/v1/account", json={"password": PASSWORD}, headers=ada).status_code == 204
    with engine.connect() as c:
        assert [r.provider for r in c.execute(db.provider_keys_table.select())] == ["groq"]


# ---------- the key never leaks ----------

def test_the_key_appears_nowhere_but_as_ciphertext(tmp_path, caplog):
    """Responses, logs, error messages, the database file and the request log: only ciphertext."""
    engine = db.open_engine(db.sqlite_url(tmp_path / "t.sqlite3"))

    def provider(request):                      # a provider that echoes the key back in its errors
        return httpx.Response(503, json={"error": {"message": f"overloaded, key {SECRET} queued"}})

    mock = httpx.Client(transport=httpx.MockTransport(provider))
    http, deps = app(lambda p, k: verify_key(p, k, client=mock), engine=engine)
    texts = []
    with caplog.at_level(logging.DEBUG):
        auth = session(http)
        for r in (put(http, auth), http.get("/v1/account/provider-keys", headers=auth),
                  http.post("/v1/account/provider-keys/openai/verify", headers=auth),
                  put(http, auth, "groq", SECRET)):
            texts.append(r.text)
        mock2 = httpx.Client(transport=httpx.MockTransport(
            lambda r: httpx.Response(401, json={"error": {"message": f"Incorrect API key provided: {SECRET}"}})))
        deps.key_verifier = lambda p, k: verify_key(p, k, client=mock2)
        texts.append(put(http, auth, "anthropic").text)                    # a rejected key's 400
    assert '"invalid_provider_key"' in texts[-1]
    for text in texts + [caplog.text]:
        assert SECRET not in text and "SECRET123" not in text
    with engine.connect() as c:
        rows = c.execute(db.provider_keys_table.select()).all()
    assert all(SECRET not in (r.status_detail or "") for r in rows) and len(rows) == 2
    assert all(SECRET not in str(row.to_dict()) for row in deps.log.rows())
    engine.dispose()
    raw = b"".join(p.read_bytes() for p in tmp_path.iterdir() if p.name.startswith("t.sqlite3"))
    assert SECRET.encode() not in raw and b"SECRET123" not in raw
