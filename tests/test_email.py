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

def test_the_email_has_the_code_and_no_code_in_the_subject():
    subject, text, html = reset_email(CODE)
    assert CODE in text and CODE in html
    assert subject == "Reset your Triage password" and CODE not in subject


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
    assert fake.sent == [{"to": TO, "user_id": "user-1", "code": CODE, "idempotency_key": "k", "display_name": None}]


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


def test_the_server_starts_when_the_logo_host_is_down(monkeypatch, caplog):
    application, deps = startup_app(monkeypatch)
    def down(request):
        if request.method == "HEAD":
            raise httpx.ConnectError("no such host")
        return httpx.Response(200, json={"id": "e1"})
    deps.email_sender = ResendSender(API_KEY_VALUE, "Triage <no-reply@send.example.dev>",
                                     transport=httpx.MockTransport(down),
                                     logo_url="https://cdn.example.dev/logo.png")
    with caplog.at_level(logging.WARNING, logger="triage.api"), TestClient(application) as http:
        assert http.get("/v1/models").status_code in (200, 401)
    assert deps.email_sender.available and deps.email_sender.logo_url is None
    assert "emails will show the plain logo mark: TRIAGE_EMAIL_LOGO_URL is unreachable" in caplog.text

def test_the_server_refuses_the_dev_log_behind_a_proxy(monkeypatch):
    application, _ = startup_app(monkeypatch, TRIAGE_DEV_EMAIL_LOG="on", TRIAGE_TRUSTED_PROXY_HOPS="1")
    with pytest.raises(RuntimeError, match="TRIAGE_DEV_EMAIL_LOG=on"), TestClient(application):
        pass


def test_fake_mode_never_uses_resend_even_with_a_key(monkeypatch):
    application, deps = startup_app(monkeypatch, TRIAGE_FAKE_MODELS="1", **RESEND_ENV)
    with TestClient(application):
        pass
    assert deps.email_sender.name == "fake"


# ---------- TRIAGE_EMAIL_FROM: .env quoting and validation ----------

from dotenv import dotenv_values                                     # noqa: E402

from app.accounts.email import SenderError, parse_sender            # noqa: E402

SENDER = "Triage <no-reply@send.example.dev>"


def from_env_file(tmp_path, line: str, final_newline: bool = True) -> dict:
    """The environment a .env containing `line` gives, parsed by python-dotenv as the server does."""
    path = tmp_path / ".env"
    path.write_text(f"RESEND_API_KEY={API_KEY_VALUE}\n{line}" + ("\n" if final_newline else ""), encoding="utf-8")
    return dict(dotenv_values(path))


@pytest.mark.parametrize("line", [
    f'TRIAGE_EMAIL_FROM="{SENDER}"',                     # double quotes
    f"TRIAGE_EMAIL_FROM='{SENDER}'",                     # single quotes
    f"TRIAGE_EMAIL_FROM={SENDER}",                       # unquoted
    f'TRIAGE_EMAIL_FROM=  "{SENDER}"  ',                 # whitespace around the quotes
    f'TRIAGE_EMAIL_FROM=" {SENDER} "',                   # whitespace inside the quotes
    f'export TRIAGE_EMAIL_FROM="{SENDER}"',
    f'TRIAGE_EMAIL_FROM="{SENDER}"\r',                   # CRLF line ending
])
@pytest.mark.parametrize("final_newline", [True, False])            # the sender as the file's last line
def test_dotenv_values_give_a_clean_sender(tmp_path, line, final_newline):
    sender = sender_from_env(False, from_env_file(tmp_path, line, final_newline))
    assert sender.name == "resend" and sender.sender == SENDER


@pytest.mark.parametrize("raw", [f'"{SENDER}"', f"'{SENDER}'", f'  "{SENDER}"  ', f"  {SENDER}\t"])
def test_quotes_and_whitespace_a_host_passes_through_are_removed(raw):
    """docker --env-file and some hosts' dashboards keep a value's quotes: one matching pair is unwrapped."""
    sender = sender_from_env(False, {**RESEND_ENV, "TRIAGE_EMAIL_FROM": raw})
    assert sender.name == "resend" and sender.sender == SENDER


@pytest.mark.parametrize("value", ["no-reply@send.example.dev", "Triage Team <no-reply@send.example.dev>",
                                   "Triage<a+reset@mail.example.co.uk>"])
def test_both_sender_forms_are_accepted(value):
    assert parse_sender(value) == value


@pytest.mark.parametrize("value, problem", [
    (f'{SENDER}"', "quote character left"),                       # what reached Resend: the 422
    (f'"{SENDER}', "quote character left"),
    (f'"{SENDER}\'', "quote character left"),                     # mismatched pair
    (f'""{SENDER}""', "quote character left"),                    # only one pair is removed
    ('"Triage, Inc" <no-reply@send.example.dev>', "quote character left"),
    ("Triage <no-reply@send.example.dev", "isn't a valid sender"),
    ("no-reply@localhost", "isn't a valid sender"),
    ("Triage", "isn't a valid sender"),
    ("<no-reply@send.example.dev>", "isn't a valid sender"),
    ("no-reply@send.example.dev\r\nBcc: someone@example.com", "isn't a valid sender"),
    ('""', "is empty"),
])
def test_malformed_senders_are_refused_without_echoing_them(value, problem):
    with pytest.raises(SenderError) as caught:
        parse_sender(value)
    assert problem in str(caught.value) and "example" not in str(caught.value)
    sender = sender_from_env(False, {**RESEND_ENV, "TRIAGE_EMAIL_FROM": value})
    assert sender.name == "unavailable" and "TRIAGE_EMAIL_FROM" in sender.reason


def test_a_stray_quote_in_env_makes_email_unavailable_at_startup(tmp_path, monkeypatch, caplog):
    env = from_env_file(tmp_path, f'TRIAGE_EMAIL_FROM={SENDER}"')     # the opening quote lost: dotenv keeps the "
    assert env["TRIAGE_EMAIL_FROM"].endswith('"')
    application, deps = startup_app(monkeypatch, **env)
    with caplog.at_level(logging.WARNING, logger="triage.api"), TestClient(application) as http:
        r = http.post("/v1/auth/password-reset/request", json={"email": TO})
    assert r.status_code == 503 and r.json()["error"]["code"] == "email_unavailable"
    warnings = [m for m in caplog.messages if "password reset is unavailable" in m]
    assert warnings == ["password reset is unavailable: TRIAGE_EMAIL_FROM has a quote character left in it "
                        "(check the quotes around the value in .env or your host's settings)"]
    assert SENDER not in caplog.text and API_KEY_VALUE not in caplog.text
