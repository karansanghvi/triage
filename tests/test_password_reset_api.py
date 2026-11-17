"""Password reset over HTTP (app/api/account_routes.py): request, verify, complete; the same answer whatever the
account; cooldown, expiry, the 5-try lock, single-use tokens, sessions revoked, rate limits, 503 without email,
nothing secret in logs or the database. No network: emails go to a FakeEmailSender or a mocked Resend."""
import json
import logging
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect, select

from app import db
from app.accounts.email import FakeEmailSender, ResendSender, UnavailableSender
from app.accounts.password_reset import ResetStore
from app.accounts.ratelimit import AuthRateLimiter
from app.accounts.store import AccountStore
from app.api.auth import Auth
from app.api.main import create_app
from tests.api_helpers import API_KEY, make_deps

EMAIL, PASSWORD, NEW_PASSWORD = "Ada.Lovelace@Example.com", "correct horse battery", "a brand new password"
UNKNOWN = "nobody@example.com"
REQUEST, VERIFY, COMPLETE = ("/v1/auth/password-reset/request", "/v1/auth/password-reset/verify",
                             "/v1/auth/password-reset/complete")


class Clock:
    def __init__(self):
        self.now = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def advance(self, **kw):
        self.now += timedelta(**kw)


def app(sender="fake", clock=None):
    deps = make_deps(trusted_proxy_hops=1)
    clock = clock or Clock()
    deps.accounts = AccountStore(deps.engine, clock)
    deps.auth_limiter = AuthRateLimiter(deps.engine, clock)
    deps.password_resets = ResetStore(deps.engine, deps.hash_secret, clock)
    deps.email_sender = FakeEmailSender() if sender == "fake" else sender
    return TestClient(create_app(deps=deps, auth=Auth([API_KEY]), accounts=True)), deps, clock


def ip(n=1):
    return {"X-Forwarded-For": f"203.0.113.{n}"}


def signup(http, email=EMAIL, password=PASSWORD):
    r = http.post("/v1/auth/signup", json={"email": email, "password": password}, headers=ip(200))
    assert r.status_code == 201
    return r.json()


def request_code(http, email=EMAIL, n=1):
    return http.post(REQUEST, json={"email": email}, headers=ip(n))


def verify(http, code, email=EMAIL, n=1):
    return http.post(VERIFY, json={"email": email, "code": code}, headers=ip(n))


def complete(http, token, password=NEW_PASSWORD, n=1):
    return http.post(COMPLETE, json={"reset_token": token, "new_password": password}, headers=ip(n))


def last_code(deps):
    return deps.email_sender.sent[-1]["code"]


def wrong(code):
    return f"{(int(code) + 1) % 1_000_000:06d}"


def reset_token(http, deps):
    assert request_code(http).status_code == 202
    r = verify(http, last_code(deps))
    assert r.status_code == 200
    return r.json()["reset_token"]


# ---------- email not configured ----------

@pytest.mark.parametrize("sender", [None, UnavailableSender(("RESEND_API_KEY",))])
@pytest.mark.parametrize("path, body", [(REQUEST, {"email": EMAIL}), (VERIFY, {"email": EMAIL, "code": "123456"}),
                                        (COMPLETE, {"reset_token": "rst_x", "new_password": NEW_PASSWORD})])
def test_without_email_every_reset_endpoint_is_503(sender, path, body):
    http, deps, _ = app(sender=sender)
    r = http.post(path, json=body, headers=ip())
    assert r.status_code == 503 and r.json()["error"]["code"] == "email_unavailable"


def test_the_endpoints_dont_exist_without_accounts():
    deps = make_deps()
    http = TestClient(create_app(deps=deps, auth=Auth([API_KEY]), accounts=False))
    assert http.post(REQUEST, json={"email": EMAIL}).status_code == 404


# ---------- request ----------

def test_existing_and_unknown_emails_get_the_same_answer():
    http, deps, _ = app()
    signup(http)
    known, unknown = request_code(http, EMAIL, n=1), request_code(http, UNKNOWN, n=2)
    assert known.status_code == unknown.status_code == 202
    assert known.content == unknown.content
    drop = {"date", "content-length"}
    assert {k: v for k, v in known.headers.items() if k not in drop} == \
        {k: v for k, v in unknown.headers.items() if k not in drop}
    assert [m["to"] for m in deps.email_sender.sent] == ["ada.lovelace@example.com"]
    code = last_code(deps)
    assert len(code) == 6 and code.isdigit()


def test_a_malformed_email_is_400():
    http, _, _ = app()
    r = request_code(http, "not an email")
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_value"
    assert r.json()["error"]["param"] == "email"


def test_inside_the_cooldown_the_answer_is_the_same_but_nothing_is_sent():
    http, deps, clock = app()
    signup(http)
    first = request_code(http)
    clock.advance(seconds=30)
    second = request_code(http)
    assert first.content == second.content and len(deps.email_sender.sent) == 1
    assert verify(http, deps.email_sender.sent[0]["code"]).status_code == 200    # the emailed code still works


def test_after_the_cooldown_a_new_code_replaces_the_old_one():
    http, deps, clock = app()
    signup(http)
    request_code(http)
    clock.advance(seconds=60)
    request_code(http)
    first, second = (m["code"] for m in deps.email_sender.sent)
    if first != second:
        assert verify(http, first).status_code == 400
    assert verify(http, second).status_code == 200


def test_a_failed_send_still_answers_202_and_logs_only_the_kind(caplog):
    resend = ResendSender("re_" + "k" * 30, "Triage <r@triage.example.com>",
                          transport=httpx.MockTransport(lambda r: httpx.Response(500, text="boom")))
    http, _, _ = app(sender=resend)
    signup(http)
    with caplog.at_level(logging.INFO, logger="triage.auth"):
        r = request_code(http)
    assert r.status_code == 202
    line = [json.loads(m) for m in caplog.messages if "password_reset_request" in m][-1]
    assert line["outcome"] == "send_failed" and line["detail"] == "resend_500"


def test_a_full_reset_through_mocked_resend():
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "e1"})
    http, deps, _ = app(sender=ResendSender("re_" + "k" * 30, "Triage <r@triage.example.com>",
                                            transport=httpx.MockTransport(handler)))
    signup(http)
    request_code(http)
    (email,) = sent
    assert email["to"] == ["ada.lovelace@example.com"]
    code = email["text"].split("code is: ")[1][:6]
    token = verify(http, code).json()["reset_token"]
    assert complete(http, token).status_code == 200


# ---------- verify ----------

def test_wrong_unknown_and_expired_codes_get_the_same_400():
    http, deps, clock = app()
    signup(http)
    request_code(http)
    code = last_code(deps)
    bodies = [verify(http, wrong(code), n=1), verify(http, code, email=UNKNOWN, n=2)]
    clock.advance(minutes=10)
    bodies.append(verify(http, code, n=3))
    assert {r.status_code for r in bodies} == {400}
    assert len({r.content for r in bodies}) == 1
    assert bodies[0].json()["error"]["code"] == "invalid_code"


def test_five_wrong_codes_kill_it():
    http, deps, _ = app()
    signup(http)
    request_code(http)
    code = last_code(deps)
    for _ in range(5):
        assert verify(http, wrong(code)).status_code == 400
    assert verify(http, code).status_code == 400


def test_a_code_works_once():
    http, deps, _ = app()
    signup(http)
    request_code(http)
    assert verify(http, last_code(deps)).status_code == 200
    assert verify(http, last_code(deps)).status_code == 400


def test_verify_returns_a_short_lived_rst_token():
    http, deps, clock = app()
    signup(http)
    request_code(http)
    r = verify(http, last_code(deps))
    body = r.json()
    assert body["reset_token"].startswith("rst_") and r.headers["cache-control"] == "no-store"
    assert body["expires_at"] == (clock.now + timedelta(minutes=15)).isoformat().replace("+00:00", "Z")


# ---------- complete ----------

def test_complete_sets_the_password_revokes_sessions_and_logs_in():
    http, deps, _ = app()
    old_sessions = [signup(http)["session_token"]]
    old_sessions.append(http.post("/v1/auth/login", json={"email": EMAIL, "password": PASSWORD},
                                  headers=ip(201)).json()["session_token"])
    user_id = deps.accounts.user_by_email("ada.lovelace@example.com").id
    api_key, _ = deps.api_keys.create(user_id, "laptop")
    r = complete(http, reset_token(http, deps))
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"user", "session_token", "expires_at"} and body["user"]["id"] == user_id
    me = http.get("/v1/auth/me", headers={"Authorization": f"Bearer {body['session_token']}"})
    assert me.status_code == 200
    for token in old_sessions:
        assert http.get("/v1/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401
    login = lambda pw: http.post("/v1/auth/login", json={"email": EMAIL, "password": pw}, headers=ip(202))
    assert login(PASSWORD).status_code == 401 and login(NEW_PASSWORD).status_code == 200
    assert deps.api_keys.lookup(api_key) is not None                    # account API keys stay valid


def test_a_reset_token_works_once():
    http, deps, _ = app()
    signup(http)
    token = reset_token(http, deps)
    assert complete(http, token).status_code == 200
    r = complete(http, token, password="yet another password")
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_reset_token"


def test_a_reset_token_expires_after_fifteen_minutes():
    http, deps, clock = app()
    signup(http)
    token = reset_token(http, deps)
    clock.advance(minutes=15)
    assert complete(http, token).json()["error"]["code"] == "invalid_reset_token"


@pytest.mark.parametrize("password", ["short", "x" * 129])
def test_the_new_password_follows_the_signup_rules(password):
    http, deps, _ = app()
    signup(http)
    token = reset_token(http, deps)
    r = complete(http, token, password=password)
    assert r.status_code == 400 and r.json()["error"]["param"] == "new_password"
    assert complete(http, token).status_code == 200                      # the token wasn't used up


@pytest.mark.parametrize("token", ["", "rst_short", "trs_" + "A" * 43, "rst_" + "A" * 43])
def test_malformed_and_unknown_tokens_are_400(token):
    http, _, _ = app()
    r = complete(http, token)
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_reset_token"


# ---------- rate limits ----------

def assert_429(r):
    assert r.status_code == 429 and r.json()["error"]["code"] == "too_many_attempts"
    assert int(r.headers["Retry-After"]) > 0


@pytest.mark.parametrize("email", [EMAIL, UNKNOWN])
def test_three_requests_per_email_per_hour(email):
    http, _, clock = app()
    signup(http)
    for n in range(3):
        assert request_code(http, email, n=n + 1).status_code == 202
        clock.advance(seconds=61)
    assert_429(request_code(http, email, n=9))
    clock.advance(hours=1)
    assert request_code(http, email, n=9).status_code == 202


def test_ten_requests_per_ip_per_hour():
    http, _, _ = app()
    for n in range(10):
        assert request_code(http, f"user{n}@example.com", n=7).status_code == 202
    assert_429(request_code(http, "user99@example.com", n=7))
    assert request_code(http, "user99@example.com", n=8).status_code == 202


def test_ten_verifies_per_ip_per_15_minutes():
    http, _, clock = app()
    for n in range(10):
        assert verify(http, "123456", email=f"user{n}@example.com", n=5).status_code == 400
    assert_429(verify(http, "123456", email="user99@example.com", n=5))
    clock.advance(minutes=15)
    assert verify(http, "123456", email="user99@example.com", n=5).status_code == 400


def test_ten_wrong_codes_per_email_per_day_across_codes_and_ips():
    http, deps, clock = app()
    signup(http)
    misses = 0
    for round_ in range(2):                                              # two codes, 5 misses each
        request_code(http, n=50 + round_)
        code = last_code(deps)
        for _ in range(5):
            assert verify(http, wrong(code), n=100 + misses).status_code == 400
            misses += 1
        clock.advance(seconds=61)
    clock.advance(hours=1)                                               # past the per-email request limit
    request_code(http, n=60)
    assert_429(verify(http, last_code(deps), n=150))                     # even the right code
    clock.advance(days=1)
    request_code(http, n=61)
    assert verify(http, last_code(deps), n=151).status_code == 200


# ---------- nothing secret anywhere ----------

def test_no_code_token_email_or_password_in_logs_or_the_database(caplog):
    http, deps, _ = app()
    with caplog.at_level(logging.DEBUG):
        signup(http)
        request_code(http)
        request_code(http, UNKNOWN, n=2)
        code = last_code(deps)
        verify(http, wrong(code))
        token = verify(http, code).json()["reset_token"]
        session = complete(http, token).json()["session_token"]
    logs = caplog.text + " ".join(caplog.messages)
    with deps.engine.connect() as c:
        names = [n for n in inspect(deps.engine).get_table_names() if n in db.metadata.tables and n != "users"]
        stored = " ".join(str(row) for name in names for row in c.execute(select(db.metadata.tables[name])).all())
    for secret in (code, token, session, NEW_PASSWORD, PASSWORD, "ada.lovelace", "nobody@"):
        assert secret not in logs, secret
        assert secret not in stored, secret
