"""Reset-code email senders (app/accounts/email.py). No network: Resend is an httpx.MockTransport."""
import json
import logging

import httpx
import pytest
from fastapi.testclient import TestClient

from app.accounts.email import (ConsoleDevSender, EmailSendError, FakeEmailSender, ResendSender, UnavailableSender,
                                dev_logger, reset_email, sender_from_env)
from app.accounts.encryption import KeyRing
from app.api.auth import Auth
from app.api.main import create_app
from tests.api_helpers import API_KEY, make_deps

API_KEY_VALUE = "re_test_" + "x" * 24
FROM = "Triage <reset@triage.example.com>"
TO = "ada@example.com"
CODE = "042917"


def resend(handler):
    return ResendSender(API_KEY_VALUE, FROM, transport=httpx.MockTransport(handler))


# ---------- the message ----------

def test_the_email_has_the_code_the_expiry_and_the_ignore_line():
    subject, text, html = reset_email(CODE)
    for part in (text, html):
        assert CODE in part and "10 minutes" in part and "didn't ask for this, ignore this email" in part.replace(
            "&#x27;", "'")
    assert "Triage" in subject and CODE not in subject


# ---------- Resend ----------

def test_resend_gets_the_documented_request():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"id": "49a3999c-0ce1-4ea6-ab68-afcd6dc2e794"})
    resend(handler).send_reset_code(TO, "user-1", CODE, "code-id-1")
    (request,) = seen
    assert request.method == "POST" and str(request.url) == "https://api.resend.com/emails"
    assert request.headers["Authorization"] == f"Bearer {API_KEY_VALUE}"
    assert request.headers["Content-Type"] == "application/json"
    assert request.headers["User-Agent"] == "triage/0.5"
    assert request.headers["Idempotency-Key"] == "code-id-1"
    body = json.loads(request.content)
    assert set(body) == {"from", "to", "subject", "text", "html"}
    assert body["from"] == FROM and body["to"] == [TO] and CODE in body["text"] and CODE in body["html"]
    assert "user-1" not in request.content.decode()                     # the user id isn't sent


@pytest.mark.parametrize("status", [400, 401, 403, 422, 429, 500, 503])
def test_a_resend_error_is_a_short_kind_with_nothing_secret(status):
    sender = resend(lambda r: httpx.Response(status, json={"message": f"bad key {API_KEY_VALUE} for {TO}"}))
    with pytest.raises(EmailSendError) as caught:
        sender.send_reset_code(TO, "user-1", CODE, "code-id-1")
    assert caught.value.kind == f"resend_{status}" and str(caught.value) == f"resend_{status}"


@pytest.mark.parametrize("error, kind", [(httpx.ReadTimeout("slow"), "timeout"),
                                         (httpx.ConnectError("no route"), "network")])
def test_network_failures_are_short_kinds(error, kind):
    def handler(request):
        raise error
    with pytest.raises(EmailSendError) as caught:
        resend(handler).send_reset_code(TO, "user-1", CODE, "code-id-1")
    assert caught.value.kind == kind and caught.value.__cause__ is None


def test_the_sender_never_shows_its_key():
    assert API_KEY_VALUE not in repr(resend(lambda r: httpx.Response(200)))


# ---------- the other senders ----------

def test_the_fake_sender_remembers_and_sends_nothing():
    fake = FakeEmailSender()
    fake.send_reset_code(TO, "user-1", CODE, "k")
    assert fake.sent == [{"to": TO, "user_id": "user-1", "code": CODE, "idempotency_key": "k"}]


def test_the_console_sender_logs_the_user_id_and_code_but_no_address():
    sender, records = ConsoleDevSender(), []
    handler = logging.Handler()
    handler.emit = records.append
    dev_logger.addHandler(handler)
    try:
        sender.send_reset_code(TO, "user-1", CODE, "k")
    finally:
        dev_logger.removeHandler(handler)
    (record,) = records
    assert record.getMessage() == f"reset code for user user-1: {CODE}" and TO not in record.getMessage()
    assert dev_logger.propagate is False


def test_the_unavailable_sender_refuses():
    with pytest.raises(EmailSendError):
        UnavailableSender(("RESEND_API_KEY",)).send_reset_code(TO, "u", CODE, "k")


# ---------- choosing one ----------

RESEND_ENV = {"RESEND_API_KEY": API_KEY_VALUE, "TRIAGE_EMAIL_FROM": FROM}


@pytest.mark.parametrize("environ, fake, expected", [
    ({**RESEND_ENV, "TRIAGE_DEV_EMAIL_LOG": "on"}, False, "console"),           # 1 beats everything
    ({**RESEND_ENV, "TRIAGE_DEV_EMAIL_LOG": "on"}, True, "console"),
    ({**RESEND_ENV}, True, "fake"),                                             # 2 beats Resend
    ({**RESEND_ENV}, False, "resend"),
    ({**RESEND_ENV, "TRIAGE_DEV_EMAIL_LOG": "off"}, False, "resend"),
    ({"RESEND_API_KEY": API_KEY_VALUE}, False, "unavailable"),
    ({"TRIAGE_EMAIL_FROM": FROM}, False, "unavailable"),
    ({}, False, "unavailable"),
    ({"TRIAGE_DEV_EMAIL_LOG": "on", "DATABASE_URL": "sqlite:///data/x.sqlite3",
      "TRIAGE_CORS_ORIGINS": "http://localhost:5173,http://127.0.0.1:3000", "TRIAGE_TRUSTED_PROXY_HOPS": "0"},
     False, "console"),
])
def test_the_sender_order(environ, fake, expected):
    assert sender_from_env(fake, environ).name == expected


def test_unavailable_names_what_is_missing():
    assert sender_from_env(False, {"TRIAGE_EMAIL_FROM": FROM}).missing == ("RESEND_API_KEY",)
    assert sender_from_env(False, {}).missing == ("RESEND_API_KEY", "TRIAGE_EMAIL_FROM")


@pytest.mark.parametrize("extra, reason", [
    ({"DATABASE_URL": "postgresql+psycopg://u:secretpw@db.example.com/triage"}, "DATABASE_URL isn't a SQLite file"),
    ({"TRIAGE_PUBLIC": "on"}, "TRIAGE_PUBLIC is on"),
    ({"TRIAGE_CORS_ORIGINS": "http://localhost:5173,https://triage.example.com"}, "TRIAGE_CORS_ORIGINS"),
    ({"TRIAGE_TRUSTED_PROXY_HOPS": "1"}, "TRIAGE_TRUSTED_PROXY_HOPS is above 0"),
])
def test_the_dev_log_is_refused_outside_local_development(extra, reason):
    with pytest.raises(RuntimeError, match="refusing to start") as caught:
        sender_from_env(True, {"TRIAGE_DEV_EMAIL_LOG": "on", **extra})
    message = str(caught.value)
    assert reason in message and "secretpw" not in message and "triage.example.com" not in message


# ---------- at startup ----------

def startup_app(monkeypatch, **env):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    deps = make_deps(keyring=KeyRing.from_env("k1:" + "A" * 43 + "="))
    return create_app(deps=deps, auth=Auth([API_KEY]), accounts=True), deps


def test_the_server_starts_without_email_settings_and_warns_by_name(monkeypatch, caplog):
    application, deps = startup_app(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="triage.api"), TestClient(application):
        pass
    assert deps.email_sender.name == "unavailable"
    assert "password reset is unavailable: RESEND_API_KEY and TRIAGE_EMAIL_FROM not set" in caplog.text


def test_the_server_picks_resend_from_the_environment(monkeypatch, caplog):
    application, deps = startup_app(monkeypatch, **RESEND_ENV)
    with caplog.at_level(logging.INFO), TestClient(application):
        pass
    assert deps.email_sender.name == "resend" and API_KEY_VALUE not in caplog.text


def test_the_server_refuses_the_dev_log_behind_a_proxy(monkeypatch):
    application, _ = startup_app(monkeypatch, TRIAGE_DEV_EMAIL_LOG="on", TRIAGE_TRUSTED_PROXY_HOPS="1")
    with pytest.raises(RuntimeError, match="TRIAGE_DEV_EMAIL_LOG=on"), TestClient(application):
        pass


def test_fake_mode_never_uses_resend_even_with_a_key(monkeypatch):
    application, deps = startup_app(monkeypatch, TRIAGE_FAKE_MODELS="1", **RESEND_ENV)
    with TestClient(application):
        pass
    assert deps.email_sender.name == "fake"
