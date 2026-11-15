"""Accounts over HTTP (app/api/account_routes.py): sign up, log in, log out, me; sessions only; rate limits;
nothing secret echoed or logged. No network: an in-memory SQLite database."""
import logging
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.accounts import credentials
from app.accounts.ratelimit import AuthRateLimiter
from app.accounts.store import AccountStore
from app.api.auth import Auth
from app.api.main import create_app
from tests.api_helpers import API_KEY, make_deps

EMAIL, PASSWORD = "Ada.Lovelace+triage@Example.com", "correct horse battery"
SITE = "https://triage.example.com"


class Clock:
    def __init__(self):
        self.now = datetime.now(timezone.utc)

    def __call__(self):
        return self.now

    def advance(self, **kw):
        self.now += timedelta(**kw)


def app(accounts=True, clock=None, **kw):
    deps = make_deps(trusted_proxy_hops=1, **kw)
    if clock is not None:
        deps.accounts, deps.auth_limiter = AccountStore(deps.engine, clock), AuthRateLimiter(deps.engine, clock)
    return TestClient(create_app(deps=deps, auth=Auth([API_KEY]), accounts=accounts,
                                 cors_origins=[SITE])), deps


def ip(n=1):
    return {"X-Forwarded-For": f"203.0.113.{n}"}


def signup(http, email=EMAIL, password=PASSWORD, headers=None, **extra):
    return http.post("/v1/auth/signup", json={"email": email, "password": password, **extra},
                     headers=headers or ip())


def login(http, email=EMAIL, password=PASSWORD, headers=None):
    return http.post("/v1/auth/login", json={"email": email, "password": password}, headers=headers or ip())


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


# ---------- the switch ----------

@pytest.mark.parametrize("method, path", [("post", "/v1/auth/signup"), ("post", "/v1/auth/login"),
                                          ("post", "/v1/auth/logout"), ("get", "/v1/auth/me")])
def test_accounts_off_means_the_paths_dont_exist(method, path):
    http, _ = app(accounts=False)
    assert getattr(http, method)(path).status_code == 404


def test_the_switch_reads_triage_accounts(monkeypatch):
    from app.api.auth import accounts_from_env
    assert accounts_from_env() is False
    monkeypatch.setenv("TRIAGE_ACCOUNTS", "on")
    assert accounts_from_env() is True


# ---------- sign up ----------

def test_signup_returns_the_user_and_a_session_once():
    http, deps = app()
    r = signup(http, display_name="  Ada  ")
    assert r.status_code == 201
    body = r.json()
    assert body["user"]["email"] == "ada.lovelace+triage@example.com" and body["user"]["display_name"] == "Ada"
    assert body["user"]["preferred_provider"] is None and set(body["user"]) == {
        "id", "email", "display_name", "preferred_provider", "created_at"}
    assert credentials.looks_like_session_token(body["session_token"]) and body["expires_at"].endswith("Z")
    assert http.get("/v1/auth/me", headers=bearer(body["session_token"])).json() == body["user"]
    stored = deps.accounts.user_by_email("ada.lovelace+triage@example.com")
    assert stored.password_hash.startswith("$argon2id$") and PASSWORD not in stored.password_hash


def test_a_taken_email_is_409_even_with_different_case_and_spaces():
    http, _ = app()
    signup(http)
    r = signup(http, email="  ADA.LOVELACE+TRIAGE@example.COM ", headers=ip(2))
    assert r.status_code == 409 and r.json()["error"]["code"] == "email_taken"


@pytest.mark.parametrize("body, param", [
    ({"email": "not-an-email", "password": PASSWORD}, "email"),
    ({"email": EMAIL, "password": "short"}, "password"),
    ({"email": EMAIL, "password": "x" * 129}, "password"),
    ({"email": EMAIL, "password": PASSWORD, "display_name": "x" * 81}, "display_name"),
])
def test_invalid_signups_are_400_naming_the_field(body, param):
    http, deps = app()
    r = http.post("/v1/auth/signup", json=body, headers=ip())
    assert r.status_code == 400 and r.json()["error"]["param"] == param
    assert deps.accounts.user_by_email("ada.lovelace+triage@example.com") is None


def test_a_display_name_is_optional():
    http, _ = app()
    assert signup(http).json()["user"]["display_name"] is None


# ---------- log in ----------

def test_login_returns_a_new_session():
    http, _ = app()
    first = signup(http).json()["session_token"]
    r = login(http, email="ada.lovelace+triage@example.com")
    assert r.status_code == 200 and r.json()["session_token"] != first
    assert http.get("/v1/auth/me", headers=bearer(r.json()["session_token"])).status_code == 200


@pytest.mark.parametrize("email, password", [
    (EMAIL, "wrong password here"),                 # wrong password
    ("nobody@example.com", PASSWORD),               # unknown email
    ("not an email", PASSWORD),                     # malformed
    (EMAIL, "x"),                                   # too short to be anyone's
])
def test_every_login_failure_is_the_same_401(email, password):
    http, _ = app()
    signup(http)
    r = login(http, email, password)
    assert r.status_code == 401
    assert r.json() == {"error": {"message": "Incorrect email or password.", "type": "invalid_request_error",
                                  "param": None, "code": "invalid_credentials"}}


def test_an_unknown_email_still_runs_one_argon2_check(monkeypatch):
    http, _ = app()
    calls = []
    real = credentials.verify_password
    monkeypatch.setattr("app.api.account_routes.verify_password",
                        lambda h, p: calls.append(h) or real(h, p))
    login(http, "nobody@example.com", PASSWORD)
    login(http, "not an email", PASSWORD)
    assert calls == [None, None]                    # None: checked against the dummy hash


def test_an_old_hash_is_upgraded_at_login():
    from argon2 import PasswordHasher
    http, deps = app()
    user = deps.accounts.user_by_id(signup(http).json()["user"]["id"])
    weak = PasswordHasher(time_cost=1, memory_cost=8 * 1024, parallelism=1).hash(PASSWORD)
    deps.accounts.update_user(user.id, password_hash=weak)
    assert login(http).status_code == 200
    assert deps.accounts.user_by_id(user.id).password_hash.startswith("$argon2id$v=19$m=19456,t=2,p=1$")


# ---------- sessions: logout, me, expiry, session-only ----------

def test_logout_revokes_only_that_session():
    http, _ = app()
    a = signup(http).json()["session_token"]
    b = login(http).json()["session_token"]
    assert http.post("/v1/auth/logout", headers=bearer(a)).status_code == 204
    r = http.get("/v1/auth/me", headers=bearer(a))
    assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_session"
    assert http.get("/v1/auth/me", headers=bearer(b)).status_code == 200


def test_idle_and_absolute_expiry_over_http():
    clock = Clock()
    http, _ = app(clock=clock)
    token = signup(http).json()["session_token"]
    clock.advance(days=7, seconds=1)
    assert http.get("/v1/auth/me", headers=bearer(token)).status_code == 401           # 7 days unused
    token = login(http).json()["session_token"]
    for _ in range(29):
        clock.advance(days=1)
        assert http.get("/v1/auth/me", headers=bearer(token)).status_code == 200
    clock.advance(days=1)
    assert http.get("/v1/auth/me", headers=bearer(token)).status_code == 401           # 30 days after login


@pytest.mark.parametrize("authorization", [None, "", f"Bearer {API_KEY}", "Bearer trs_short",
                                           "Bearer trs_" + "A" * 43, "Basic abc", "trs_" + "A" * 43])
def test_account_endpoints_need_a_session_token(authorization):
    http, _ = app()
    headers = {} if authorization is None else {"Authorization": authorization}
    for method, path in (("get", "/v1/auth/me"), ("post", "/v1/auth/logout")):
        r = getattr(http, method)(path, headers=headers)
        assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_session", (authorization, path)


def test_a_session_token_on_chat_is_an_account_caller():
    http, _ = app()
    token = signup(http).json()["session_token"]
    r = http.post("/v1/chat/completions", json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
                  headers=bearer(token))
    assert r.status_code == 403 and r.json()["error"]["code"] == "account_mode_unavailable"   # until account chat


# ---------- rate limits ----------

def test_signup_is_limited_to_five_per_ip_per_hour():
    clock = Clock()
    http, _ = app(clock=clock)
    for i in range(5):
        assert signup(http, email=f"user{i}@example.com").status_code == 201
    r = signup(http, email="user5@example.com")
    assert r.status_code == 429 and r.json()["error"]["code"] == "too_many_attempts"
    assert 3000 < int(r.headers["retry-after"]) <= 3600
    assert signup(http, email="user5@example.com", headers=ip(2)).status_code == 201   # another IP
    clock.advance(hours=1)
    assert signup(http, email="user6@example.com").status_code == 201


def test_login_is_limited_to_ten_per_ip_per_fifteen_minutes():
    clock = Clock()
    http, _ = app(clock=clock)
    signup(http)
    for i in range(10):
        assert login(http, email=f"nobody{i}@example.com").status_code == 401
    r = login(http)                                        # even the right password
    assert r.status_code == 429 and 0 < int(r.headers["retry-after"]) <= 900
    assert login(http, headers=ip(2)).status_code == 200
    clock.advance(minutes=15)
    assert login(http).status_code == 200


def test_five_failures_for_one_email_lock_it_with_the_same_401():
    clock = Clock()
    http, _ = app(clock=clock)
    signup(http)
    for n in range(5):
        assert login(http, password="wrong password here", headers=ip(10 + n)).status_code == 401
    locked = login(http, headers=ip(20))                    # right password, new IP: still the same 401
    assert locked.status_code == 401 and locked.json()["error"]["code"] == "invalid_credentials"
    assert "retry-after" not in locked.headers
    assert login(http, email="someone-else@example.com", headers=ip(21)).status_code == 401
    clock.advance(minutes=15)
    assert login(http, headers=ip(22)).status_code == 200


def test_a_locked_email_still_takes_an_argon2_check(monkeypatch):
    http, _ = app()
    signup(http)
    for n in range(5):
        login(http, password="wrong password here", headers=ip(10 + n))
    calls = []
    real = credentials.verify_password
    monkeypatch.setattr("app.api.account_routes.verify_password", lambda h, p: calls.append(1) or real(h, p))
    login(http, headers=ip(30))
    assert calls == [1]


# ---------- nothing secret echoed or logged ----------

SECRET_PASSWORD = "Zebra-Password-77310"


@pytest.mark.parametrize("body", [
    {"email": EMAIL, "password": SECRET_PASSWORD, "unexpected": SECRET_PASSWORD},     # unknown field
    {"email": EMAIL, "password": [SECRET_PASSWORD]},                                   # wrong type
    {"email": SECRET_PASSWORD, "password": SECRET_PASSWORD},                           # bad email
    {"email": EMAIL, "password": SECRET_PASSWORD[:5]},                                 # too short
    {"email": EMAIL, "password": SECRET_PASSWORD, "display_name": SECRET_PASSWORD * 9},
])
@pytest.mark.parametrize("path", ["/v1/auth/signup", "/v1/auth/login"])
def test_errors_never_echo_what_was_sent(body, path):
    http, _ = app()
    r = http.post(path, json=body, headers=ip())
    assert r.status_code in (400, 401)
    assert SECRET_PASSWORD not in r.text and SECRET_PASSWORD[:5] not in r.text and "Ada.Lovelace" not in r.text


def test_logs_hold_no_email_password_or_token(caplog):
    http, _ = app()
    with caplog.at_level(logging.DEBUG):
        token = signup(http, password=SECRET_PASSWORD).json()["session_token"]
        login(http, password=SECRET_PASSWORD)
        login(http, password="wrong password here")
        login(http, email="nobody@example.com")
        http.get("/v1/auth/me", headers=bearer(token))
        http.post("/v1/auth/logout", headers=bearer(token))
    text = caplog.text.lower()
    for secret in (SECRET_PASSWORD.lower(), "ada.lovelace", "nobody@example.com", token.lower(), "203.0.113"):
        assert secret not in text, secret
    auth_lines = [r.getMessage() for r in caplog.records if r.name == "triage.auth"]
    assert '"outcome":"failed"' in "".join(auth_lines) and all('"visitor":' in l for l in auth_lines)


# ---------- CORS ----------

@pytest.mark.parametrize("method", ["GET", "POST", "PATCH", "DELETE"])
def test_the_website_can_call_account_endpoints_with_an_authorization_header(method):
    http, _ = app()
    r = http.options("/v1/auth/me", headers={"Origin": SITE, "Access-Control-Request-Method": method,
                                             "Access-Control-Request-Headers": "authorization, content-type"})
    assert r.status_code == 200 and r.headers["access-control-allow-origin"] == SITE
    assert method in r.headers["access-control-allow-methods"]
    assert "access-control-allow-credentials" not in r.headers           # bearer tokens, never cookies
    token = signup(http).json()["session_token"]
    me = http.get("/v1/auth/me", headers=bearer(token) | {"Origin": SITE})
    assert me.status_code == 200 and me.headers["access-control-allow-origin"] == SITE


# ---------- account settings ----------

def patch(http, token, body):
    return http.patch("/v1/account", json=body, headers=bearer(token))


def test_patch_changes_only_the_fields_sent_and_null_clears():
    http, _ = app()
    token = signup(http, display_name="Ada").json()["session_token"]
    r = patch(http, token, {"preferred_provider": " Anthropic "})
    assert r.status_code == 200 and (r.json()["display_name"], r.json()["preferred_provider"]) == ("Ada", "anthropic")
    r = patch(http, token, {"display_name": "Countess"})
    assert (r.json()["display_name"], r.json()["preferred_provider"]) == ("Countess", "anthropic")
    r = patch(http, token, {"display_name": None, "preferred_provider": None})
    assert (r.json()["display_name"], r.json()["preferred_provider"]) == (None, None)
    assert patch(http, token, {}).json() == http.get("/v1/auth/me", headers=bearer(token)).json()
    assert patch(http, token, {"display_name": "   "}).json()["display_name"] is None


@pytest.mark.parametrize("body, param", [({"preferred_provider": "ollama"}, "preferred_provider"),
                                         ({"preferred_provider": "openai,groq"}, "preferred_provider"),
                                         ({"display_name": "x" * 81}, "display_name")])
def test_patch_refuses_bad_values(body, param):
    http, _ = app()
    token = signup(http).json()["session_token"]
    r = patch(http, token, body)
    assert r.status_code == 400 and r.json()["error"]["param"] == param


@pytest.mark.parametrize("body", [{"email": "new@example.com"}, {"password_hash": "x"}, {"id": "other"}])
def test_patch_cant_change_anything_else(body):
    http, _ = app()
    token = signup(http).json()["session_token"]
    assert patch(http, token, body).status_code == 400
    assert http.get("/v1/auth/me", headers=bearer(token)).json()["email"] == "ada.lovelace+triage@example.com"


def test_changing_the_password_revokes_every_other_session():
    http, _ = app()
    current = signup(http).json()["session_token"]
    others = [login(http, headers=ip(n)).json()["session_token"] for n in (2, 3)]
    r = http.post("/v1/account/password", json={"current_password": PASSWORD, "new_password": "a brand new secret"},
                  headers=bearer(current))
    assert r.status_code == 204
    assert http.get("/v1/auth/me", headers=bearer(current)).status_code == 200
    assert all(http.get("/v1/auth/me", headers=bearer(t)).status_code == 401 for t in others)
    assert login(http).status_code == 401
    assert login(http, password="a brand new secret").status_code == 200


def test_a_wrong_current_password_is_403_and_changes_nothing():
    http, _ = app()
    token = signup(http).json()["session_token"]
    other = login(http).json()["session_token"]
    r = http.post("/v1/account/password", json={"current_password": "not my password", "new_password": "a brand new secret"},
                  headers=bearer(token))
    assert r.status_code == 403 and r.json()["error"]["code"] == "incorrect_password"
    assert "not my password" not in r.text
    assert http.get("/v1/auth/me", headers=bearer(other)).status_code == 200 and login(http).status_code == 200


def test_a_too_short_new_password_is_400():
    http, _ = app()
    token = signup(http).json()["session_token"]
    r = http.post("/v1/account/password", json={"current_password": PASSWORD, "new_password": "short"},
                  headers=bearer(token))
    assert r.status_code == 400 and r.json()["error"]["param"] == "new_password" and "short" not in r.text


def delete_account(http, token, password=PASSWORD):
    return http.request("DELETE", "/v1/account", json={"password": password}, headers=bearer(token))


def test_deleting_the_account_keeps_request_rows_without_the_user():
    from app import db
    from app.api.request_log import RequestRow, SQLRequestLog
    engine = db.open_engine("sqlite://")
    log = SQLRequestLog(engine)
    http, deps = app(engine=engine, log=log)
    token = signup(http).json()["session_token"]
    other = login(http).json()["session_token"]
    user_id = http.get("/v1/auth/me", headers=bearer(token)).json()["id"]
    bob = signup(http, email="bob@example.com", headers=ip(2)).json()
    for i, owner in enumerate((user_id, user_id, bob["user"]["id"])):
        log.write(RequestRow(request_id=f"r{i}", timestamp="t", key_id="k", requested_model="auto", stream=False,
                             user_id=owner))
    assert delete_account(http, token, "wrong password!!").status_code == 403
    assert delete_account(http, token).status_code == 204
    assert [(r.request_id, r.user_id) for r in log.rows()] == [("r0", None), ("r1", None), ("r2", bob["user"]["id"])]
    assert deps.accounts.user_by_id(user_id) is None
    for t in (token, other):
        assert http.get("/v1/auth/me", headers=bearer(t)).status_code == 401
    assert login(http).status_code == 401
    assert signup(http, headers=ip(3)).status_code == 201                       # the email is free again
    assert http.get("/v1/auth/me", headers=bearer(bob["session_token"])).status_code == 200


@pytest.mark.parametrize("method, path, body", [
    ("patch", "/v1/account", {"display_name": "x"}),
    ("post", "/v1/account/password", {"current_password": PASSWORD, "new_password": "a brand new secret"}),
    ("delete", "/v1/account", {"password": PASSWORD}),
])
def test_account_settings_need_a_session_token(method, path, body):
    http, _ = app()
    signup(http)
    r = http.request(method.upper(), path, json=body, headers={"Authorization": f"Bearer {API_KEY}"})
    assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_session"


def test_auth_events_are_printed_by_the_server():
    """Not only visible to tests: the server prints the auth lines (to stderr, like the request lines)."""
    app()
    auth_logger = logging.getLogger("triage.auth")
    assert auth_logger.isEnabledFor(logging.INFO)
    assert any(isinstance(h, logging.StreamHandler) for h in auth_logger.handlers)


# ---------- current-password checks share the per-email failure limit ----------

def wrong_password_change(http, token):
    return http.post("/v1/account/password", json={"current_password": "wrong password here",
                                                    "new_password": "a brand new secret"}, headers=bearer(token))


@pytest.mark.parametrize("endpoint", ["password", "delete"])
def test_five_wrong_current_passwords_then_429_with_retry_after(endpoint):
    clock = Clock()
    http, deps = app(clock=clock)
    token = signup(http).json()["session_token"]

    def attempt(password):
        if endpoint == "password":
            return http.post("/v1/account/password", json={"current_password": password,
                                                            "new_password": "a brand new secret"},
                             headers=bearer(token))
        return delete_account(http, token, password)

    for _ in range(5):
        assert attempt("wrong password here").status_code == 403
    r = attempt(PASSWORD)                                          # even the right one: no more checks for now
    assert r.status_code == 429 and r.json()["error"]["code"] == "too_many_attempts"
    assert 0 < int(r.headers["retry-after"]) <= 900 and PASSWORD not in r.text
    assert http.get("/v1/auth/me", headers=bearer(token)).status_code == 200      # still logged in
    clock.advance(minutes=15)
    assert attempt(PASSWORD).status_code == 204


def test_an_over_the_limit_attempt_runs_no_password_check(monkeypatch):
    http, _ = app()
    token = signup(http).json()["session_token"]
    for _ in range(5):
        wrong_password_change(http, token)
    calls = []
    monkeypatch.setattr("app.api.account_routes.verify_password", lambda h, p: calls.append(1) or True)
    assert wrong_password_change(http, token).status_code == 429 and calls == []


def test_settings_failures_and_login_failures_share_one_count():
    http, _ = app()
    token = signup(http).json()["session_token"]
    for n in range(3):
        assert login(http, password="wrong password here", headers=ip(10 + n)).status_code == 401
    for _ in range(2):
        assert wrong_password_change(http, token).status_code == 403
    assert wrong_password_change(http, token).status_code == 429             # 3 + 2 = 5: full
    assert login(http, headers=ip(20)).status_code == 401                    # login is locked too (same 401)


def test_correct_current_passwords_dont_count():
    http, _ = app()
    token = signup(http).json()["session_token"]
    for i in range(6):
        new = f"new long secret {i}"
        old = PASSWORD if i == 0 else f"new long secret {i - 1}"
        assert http.post("/v1/account/password", json={"current_password": old, "new_password": new},
                         headers=bearer(token)).status_code == 204


def test_another_users_failures_dont_lock_this_account():
    http, _ = app()
    ada = signup(http).json()["session_token"]
    bob = signup(http, email="bob@example.com", headers=ip(2)).json()["session_token"]
    for _ in range(5):
        wrong_password_change(http, bob)
    assert http.post("/v1/account/password", json={"current_password": PASSWORD, "new_password": "a brand new secret"},
                     headers=bearer(ada)).status_code == 204
