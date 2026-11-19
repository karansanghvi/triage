"""The welcome email (app/accounts/email_template.py, app/api/account_routes.py): content, links, escaping, the
email-client rules, TRIAGE_APP_URL / TRIAGE_PUBLIC_API_URL at startup, and sending it once after signup in the
background. No network: a FakeEmailSender or a mocked Resend."""
import html as htmllib
import json
import logging
import re
from types import SimpleNamespace

import anyio
import httpx
import pytest

from app.accounts.email import (EmailSendError, FakeEmailSender, ResendSender, UnavailableSender, email_urls,
                                sender_from_env)
from app.accounts.email_template import PublicUrlError, parse_public_url, reset_email, welcome_email
from app.api.main import set_up_email
from tests.test_password_reset_api import EMAIL, PASSWORD, app, ip, signup

APP = "https://app.example.dev"
API = "https://api.example.dev"
LOGO = "https://cdn.example.dev/triage-logo.png"
USER_ID = "5f0c1d2e-0000-4000-8000-000000000001"
TOKEN_SHAPES = re.compile(r"(rst|trs)_[0-9A-Za-z]{8,}|trk_[0-9A-Za-z]{8,}|re_[0-9A-Za-z]{8,}")
RESEND_ENV = {"RESEND_API_KEY": "re_test_" + "x" * 24, "TRIAGE_EMAIL_FROM": "Triage <no-reply@send.example.dev>"}


def visible(html: str) -> str:
    """The HTML's text as a reader sees it: tags removed, entities decoded, whitespace collapsed."""
    return " ".join(htmllib.unescape(re.sub(r"<[^>]+>", " ", html)).split())


def flat(text: str) -> str:
    return " ".join(text.split())



def code_in_html(html: str) -> str:
    """The dark code block's text, line by line."""
    cell = re.search(r'background-color: #0F172A;[^>]*>(.*?)</td>', html).group(1)
    return htmllib.unescape(re.sub(r"<[^>]+>", "", cell.replace("<br>", "\n"))).replace("\xa0", " ")


def code_in_text(text: str) -> str:
    """The plain text's code sample (indented 7 spaces)."""
    return "\n".join(line[7:] for line in text.split("\n") if line.startswith("       "))

CONTENT = [
    "Welcome to Triage",
    "You now have an OpenAI-compatible router that sends every prompt to the cheapest model that can answer it "
    "well, so you pay less without your answers getting worse.",
    "Get your first request through in three steps:",
    "Add a provider key",
    "Save your OpenAI, Anthropic, Groq or Gemini key. It's encrypted the moment you save it, and only the last 4 "
    "characters are ever shown.",
    "Create a Triage API key",
    "Your app sends this key to Triage instead of your provider keys. It's shown once, so save it somewhere safe.",
    "Point your app at Triage",
    "Keep your OpenAI code and change two lines:",
    "What happens next: for each request, Triage predicts how hard the prompt is, picks the cheapest model that "
    "clears your quality bar, and falls back automatically if a provider has a bad day. Every response tells you "
    "which model answered, why, and how much you saved, and your dashboard adds it all up.",
    "Triage · Every prompt, routed to the right model.",
    "You're receiving this because you created a Triage account with this email. If that wasn't you, use Forgot "
    "password on the log in page to take over the account, then delete it in Settings → Account.",
]
LINKS = ["Add a provider key →", "Create an API key →", "Open your settings"]


# ---------- content ----------

@pytest.mark.parametrize("name, hello", [("Ada", "Hi Ada,"), (None, "Hi,"), ("  ", "Hi,"),
                                         ("  Ada \n Lovelace ", "Hi Ada Lovelace,")])
@pytest.mark.parametrize("logo", [None, LOGO])
def test_html_and_text_have_the_same_content(name, hello, logo):
    subject, text, html = welcome_email(name, logo, APP, API)
    assert subject == "Welcome to Triage"
    for part in (visible(html), flat(text)):
        assert hello in part
        for line in CONTENT + LINKS:
            assert line in part, line
    expected = ('client = OpenAI(\n    base_url="https://api.example.dev/v1",\n    api_key="trk_...",\n)\n'
                '# then use model="auto"')
    assert code_in_html(html) == expected and code_in_text(text) == expected


def test_the_preheader_is_first_and_hidden():
    _, _, html = welcome_email("Ada", None, APP, API)
    body = html.split("<body", 1)[1]
    first_div = re.search(r"<div style=\"([^\"]*)\">([^<]*)</div>", body)
    assert first_div.group(2) == "Your first routed request is three steps away."
    assert "display: none" in first_div.group(1) and "mso-hide: all" in first_div.group(1)
    assert body.index("three steps away") < body.index("Welcome to Triage</h1>")


def test_every_link_goes_to_an_app_url_page():
    _, text, html = welcome_email("Ada", LOGO, APP, API)
    assert re.findall(r'href="([^"]+)"', html) == [f"{APP}/settings/provider-keys", f"{APP}/settings/api-keys",
                                                   f"{APP}/settings/get-started", APP]
    for url in (f"{APP}/settings/provider-keys", f"{APP}/settings/api-keys", f"{APP}/settings/get-started"):
        assert url in text
    assert set(re.findall(r"https?://[^\s\"<>]+", text)) == {f"{APP}/settings/provider-keys",
                                                             f"{APP}/settings/api-keys",
                                                             f"{APP}/settings/get-started", APP, f"{API}/v1"}


def test_without_an_app_url_there_are_no_links_or_button():
    _, text, html = welcome_email("Ada", LOGO, None, API)
    assert "href=" not in html and "<a " not in html
    for label in LINKS:
        assert label not in html and label not in text
    assert "https://app" not in text and "Add a provider key" in text and "Welcome to Triage" in visible(html)


def test_the_code_sample_uses_the_public_api_url_or_the_default():
    _, text, html = welcome_email("Ada", None, APP, None)
    assert 'base_url="https://api.gettriage.dev/v1"' in text
    assert '<span style="color: #86EFAC;">"https://api.gettriage.dev/v1"</span>' in html
    assert '<span style="color: #86EFAC;">"trk_..."</span>' in html
    assert "background-color: #0F172A" in html and "font-size: 13px" in html and "monospace" in html


@pytest.mark.parametrize("name", ["<script>alert(1)</script>", '"><img src=x onerror=alert(1)>', "Ada & <b>Co</b>"])
def test_the_display_name_is_escaped(name):
    _, text, html = welcome_email(name, None, APP, API)
    assert "<script" not in html and "<b>" not in html and "onerror=alert(1)>" not in html
    assert htmllib.escape(name, quote=True) in html
    assert f"Hi {name}," in visible(html) and f"Hi {name}," in text


def test_email_client_rules_and_the_design():
    _, _, html = welcome_email("Ada", LOGO, APP, API)
    lowered = html.lower()
    for banned in ("<style", "<link", "<script", "<svg", "@import", "@font-face", "fonts.googleapis", "class=",
                   "?utm", "pixel"):
        assert banned not in lowered, banned
    assert "max-width: 600px" in html and 'role="presentation"' in html and "background-color: #F4F5F8" in html
    assert "-apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif" in html
    # three 26px step circles
    circles = re.findall(r'<td width="26" height="26"[^>]*>(\d)</td>', html)
    assert circles == ["1", "2", "3"]
    for style in ("width: 26px; min-width: 26px; height: 26px; background-color: #EFF4FF; border: 1px solid #C7D7FE",
                  "color: #1D4ED8", 'bgcolor="#2563EB" style="background-color: #2563EB; border-radius: 10px;"',
                  "font-size: 12px", "color: #6B7280"):
        assert style in html, style
    assert re.findall(r'<img [^>]*src="([^"]+)"', html) == [LOGO]


@pytest.mark.parametrize("render", [lambda logo: welcome_email("Ada", logo, APP, API)[2],
                                    lambda logo: reset_email("042917", "Ada", logo, APP)[2]])
def test_both_emails_use_alt_empty_on_the_logo(render):
    (tag,) = re.findall(r"<img [^>]*>", render(LOGO))
    assert ' alt="" ' in tag and 'width="36" height="36" border="0"' in tag and "display: block" in tag
    assert "<img" not in render(None) and 'bgcolor="#2563EB"' in render(None)


# ---------- the two URLs at startup ----------

@pytest.mark.parametrize("value, expected", [
    (None, None), ("", None), ("  ", None), ("https://gettriage.dev", "https://gettriage.dev"),
    ("https://gettriage.dev/", "https://gettriage.dev"), ('"https://gettriage.dev"', "https://gettriage.dev"),
    ("http://localhost:5173", "http://localhost:5173"), ("http://127.0.0.1:5173/", "http://127.0.0.1:5173"),
    ("https://example.dev/app/", "https://example.dev/app")])
def test_app_urls_are_cleaned(value, expected):
    assert parse_public_url(value, "TRIAGE_APP_URL") == expected


@pytest.mark.parametrize("value, expected", [("https://api.gettriage.dev/v1", "https://api.gettriage.dev"),
                                             ("https://api.gettriage.dev/v1/", "https://api.gettriage.dev"),
                                             ("http://localhost:8000", "http://localhost:8000")])
def test_the_api_url_drops_a_trailing_v1(value, expected):
    assert parse_public_url(value, "TRIAGE_PUBLIC_API_URL", strip_v1=True) == expected


@pytest.mark.parametrize("value", ["http://gettriage.dev", "ftp://gettriage.dev", "gettriage.dev", "javascript:x",
                                   "https://gettriage.dev/?utm=1", "https://gettriage.dev/#x",
                                   "https://user:pw@gettriage.dev", 'https://gettriage.dev/" onclick="x',
                                   "https://getti age.dev", "https://", "https://gettriage.dev:99999"])
def test_bad_urls_are_refused_without_echoing_them(value):
    with pytest.raises(PublicUrlError) as caught:
        parse_public_url(value, "TRIAGE_APP_URL")
    assert "TRIAGE_APP_URL" in str(caught.value) and "gettriage" not in str(caught.value)


def test_sender_from_env_reads_both_urls():
    sender = sender_from_env(False, {**RESEND_ENV, "TRIAGE_APP_URL": APP + "/",
                                     "TRIAGE_PUBLIC_API_URL": API + "/v1"})
    assert sender.app_url == APP and sender.api_url == API and sender.problems == []


def test_an_unset_app_url_means_one_startup_warning(caplog):
    sender = sender_from_env(False, RESEND_ENV)
    assert sender.app_url is None and sender.api_url == "https://api.gettriage.dev"
    with caplog.at_level(logging.WARNING, logger="triage.api"):
        set_up_email(SimpleNamespace(email_sender=sender))
    assert caplog.messages == ["emails are sent without links: TRIAGE_APP_URL is not set"]


def test_invalid_urls_warn_by_name_only():
    _, _, problems = email_urls({"TRIAGE_APP_URL": "http://insecure.example.dev",
                                 "TRIAGE_PUBLIC_API_URL": "ftp://api.example.dev"})
    assert problems == (
        "emails are sent without links: TRIAGE_APP_URL must be an https:// URL (http:// only for localhost)",
        "the welcome email shows https://api.gettriage.dev: TRIAGE_PUBLIC_API_URL must be an https:// URL "
        "(http:// only for localhost)")


# ---------- what reaches Resend ----------

@pytest.mark.parametrize("name", ["Ada", None])
@pytest.mark.parametrize("app_url", [APP, None])
def test_no_address_user_id_or_token_in_the_message(name, app_url):
    seen = []

    def handler(request):
        seen.append((request.headers, json.loads(request.content)))
        return httpx.Response(200, json={"id": "e1"})
    sender = ResendSender(RESEND_ENV["RESEND_API_KEY"], RESEND_ENV["TRIAGE_EMAIL_FROM"],
                          transport=httpx.MockTransport(handler), logo_url=LOGO, app_url=app_url, api_url=API)
    sender.send_welcome("ada@example.com", USER_ID, "idem-key-1", display_name=name)
    ((headers, body),) = seen
    assert body["to"] == ["ada@example.com"] and body["subject"] == "Welcome to Triage"
    assert body["from"] == "Triage <no-reply@send.example.dev>" and headers["idempotency-key"] == "idem-key-1"
    for part in (body["subject"], body["text"], body["html"]):
        assert "ada@example.com" not in part and "ada@" not in part and USER_ID not in part
        assert "idem-key-1" not in part and not TOKEN_SHAPES.search(part)


# ---------- after signup ----------

def test_sent_once_after_signup_with_the_display_name(caplog):
    http, deps, _ = app()
    with caplog.at_level(logging.INFO, logger="triage.auth"):
        r = http.post("/v1/auth/signup", json={"email": EMAIL, "password": PASSWORD, "display_name": "Ada"},
                      headers=ip(200))
    assert r.status_code == 201
    (welcome,) = deps.email_sender.welcomed
    assert welcome["to"] == "ada.lovelace@example.com" and welcome["display_name"] == "Ada"
    assert welcome["user_id"] == r.json()["user"]["id"]
    assert '"event":"welcome_email","outcome":"sent"' in caplog.text
    assert "ada.lovelace" not in caplog.text.lower() and welcome["user_id"] not in caplog.text


def test_only_at_signup():
    http, deps, _ = app()
    signup(http)
    assert http.post("/v1/auth/signup", json={"email": EMAIL, "password": PASSWORD},
                     headers=ip(201)).status_code == 409
    assert http.post("/v1/auth/login", json={"email": EMAIL, "password": PASSWORD},
                     headers=ip(202)).status_code == 200
    http.post("/v1/auth/password-reset/request", json={"email": EMAIL}, headers=ip(203))
    assert len(deps.email_sender.welcomed) == 1 and len(deps.email_sender.sent) == 1


def test_sent_after_the_response_has_gone_out():
    http, deps, _ = app()
    events = []

    class Recording(FakeEmailSender):
        def send_welcome(self, *args, **kwargs):
            events.append("welcome sent")
    deps.email_sender = Recording()
    application = http.app
    body = json.dumps({"email": EMAIL, "password": PASSWORD}).encode()
    requests = [{"type": "http.request", "body": body, "more_body": False}]

    async def receive():
        return requests.pop(0) if requests else {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.start":
            events.append(f"status {message['status']}")
        elif message["type"] == "http.response.body" and not message.get("more_body"):
            events.append("response sent")

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "http",
             "path": "/v1/auth/signup", "raw_path": b"/v1/auth/signup", "query_string": b"", "root_path": "",
             "headers": [(b"host", b"testserver"), (b"content-type", b"application/json"),
                         (b"x-forwarded-for", b"203.0.113.200")],
             "client": ("127.0.0.1", 50000), "server": ("testserver", 80)}
    anyio.run(application, scope, receive, send)
    assert events == ["status 201", "response sent", "welcome sent"]


class Failing(FakeEmailSender):
    def __init__(self, error):
        super().__init__()
        self.error = error

    def send_welcome(self, *args, **kwargs):
        raise self.error


@pytest.mark.parametrize("error, outcome, detail", [(EmailSendError("resend_422"), "send_failed", "resend_422"),
                                                    (EmailSendError("timeout"), "send_failed", "timeout"),
                                                    (RuntimeError("db said ada.lovelace@example.com"), "error",
                                                     "RuntimeError")])
def test_signup_still_succeeds_when_sending_fails(error, outcome, detail, caplog):
    http, deps, _ = app(sender=Failing(error))
    with caplog.at_level(logging.INFO, logger="triage.auth"):
        body = signup(http)
    assert body["session_token"]
    (line,) = [json.loads(m) for m in caplog.messages if '"welcome_email"' in m]
    assert line["outcome"] == outcome and line["detail"] == detail
    assert "ada.lovelace" not in caplog.text.lower()
    assert http.get("/v1/auth/me", headers={"Authorization": f"Bearer {body['session_token']}"}).status_code == 200


def test_signup_still_succeeds_without_email_configured(caplog):
    http, deps, _ = app(sender=UnavailableSender(("RESEND_API_KEY", "TRIAGE_EMAIL_FROM")))
    with caplog.at_level(logging.INFO, logger="triage.auth"):
        signup(http)
    assert "welcome_email" not in caplog.text


def test_signup_through_mocked_resend_sends_the_welcome_email():
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "e1"})
    sender = ResendSender(RESEND_ENV["RESEND_API_KEY"], RESEND_ENV["TRIAGE_EMAIL_FROM"],
                          transport=httpx.MockTransport(handler), app_url=APP, api_url=API)
    http, _, _ = app(sender=sender)
    signup(http)
    (email,) = sent
    assert email["to"] == ["ada.lovelace@example.com"] and email["subject"] == "Welcome to Triage"
    assert f'href="{APP}/settings/get-started"' in email["html"]
