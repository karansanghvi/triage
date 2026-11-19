"""The password reset email (app/accounts/email_template.py): content, escaping, email-client rules, the logo,
and what reaches Resend. No network."""
import html as htmllib
import json
import logging
import re
from types import SimpleNamespace

import httpx
import pytest

from app.accounts.email import ResendSender, logo_url_problem, sender_from_env
from app.api.main import set_up_email
from app.accounts.email_template import LogoUrlError, parse_logo_url, reset_email

CODE = "042917"
LOGO = "https://cdn.example.dev/triage-logo.png"
EMAIL, USER_ID = "ada@example.com", "5f0c1d2e-0000-4000-8000-000000000001"
TOKEN_SHAPES = re.compile(r"(rst|trs|trk)_[0-9A-Za-z]{8,}|re_[0-9A-Za-z]{8,}")


def visible(html: str) -> str:
    """The HTML's text as a reader sees it: tags removed, entities decoded, whitespace collapsed."""
    return " ".join(htmllib.unescape(re.sub(r"<[^>]+>", " ", html)).split())


def flat(text: str) -> str:
    return " ".join(text.split())


CONTENT = ["Reset your password",
           "We received a request to reset the password for your Triage account. Enter this code to continue:",
           "This code expires in 10 minutes and can be used only once.",
           "Didn't ask for this? You can safely ignore this email; your password won't change.",
           "Triage will never ask you for this code by email, phone or chat.",
           "Triage · Every prompt, routed to the right model.",
           "You're receiving this because a password reset was requested for your account."]


@pytest.mark.parametrize("name, hello", [("Ada", "Hi Ada,"), (None, "Hi,"), ("", "Hi,"), ("   ", "Hi,"),
                                         ("  Ada \n Lovelace ", "Hi Ada Lovelace,")])
@pytest.mark.parametrize("logo", [None, LOGO])
def test_html_and_text_have_the_same_content(name, hello, logo):
    subject, text, html = reset_email(CODE, name, logo)
    assert subject == "Reset your Triage password" and CODE not in subject
    for part in (visible(html), flat(text)):
        assert CODE in part and hello in part
        for line in CONTENT:
            assert line in part, line
    assert "https://gettriage.dev" in text and 'href="https://gettriage.dev"' in html


def test_the_preheader_is_first_and_hidden():
    _, _, html = reset_email(CODE)
    body = html.split("<body", 1)[1]
    first_div = re.search(r"<div style=\"([^\"]*)\">([^<]*)</div>", body)
    assert first_div.group(2) == "Your code expires in 10 minutes."
    assert "display: none" in first_div.group(1) and "mso-hide: all" in first_div.group(1)
    assert body.index("Your code expires") < body.index("Reset your password")


@pytest.mark.parametrize("name", ["<script>alert(1)</script>", '"><img src=x onerror=alert(1)>', "Ada & <b>Co</b>"])
def test_the_display_name_is_escaped(name):
    _, text, html = reset_email(CODE, name)
    assert "<script" not in html and "<b>" not in html and "onerror=alert(1)>" not in html
    assert htmllib.escape(name, quote=True) in html
    assert f"Hi {name}," in visible(html) and f"Hi {name}," in text     # shown literally, not run


def test_email_client_rules():
    _, _, html = reset_email(CODE, "Ada", LOGO)
    lowered = html.lower()
    for banned in ("<style", "<link", "<script", "<svg", "@import", "@font-face", "fonts.googleapis", "class="):
        assert banned not in lowered, banned
    assert "max-width: 600px" in html and 'role="presentation"' in html
    assert "background-color: #F4F5F8" in html
    assert "-apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif" in html
    # The design's colours and the code box
    for style in ("border: 1px solid #E5E7EB; border-radius: 16px; padding: 40px", "font-size: 24px",
                  "color: #111827", "background-color: #EFF4FF; border: 1px solid #C7D7FE; border-radius: 12px",
                  "font-size: 34px", "letter-spacing: 10px", "color: #1D4ED8", "monospace",
                  "font-size: 12px", "color: #6B7280", "#2563EB"):
        assert style in html, style
    # Only one image (the logo), one link (the site), no tracking
    assert re.findall(r'<img [^>]*src="([^"]+)"', html) == [LOGO]
    assert re.findall(r'href="([^"]+)"', html) == ["https://gettriage.dev"]
    assert "?utm" not in html and "pixel" not in lowered


def test_without_a_logo_url_the_blue_square_stays_and_there_is_no_image():
    _, _, html = reset_email(CODE, "Ada", None)
    assert "<img" not in html
    assert 'bgcolor="#2563EB"' in html and "width: 36px; height: 36px" in html and ">Triage</td>" in html


def test_with_a_logo_url_it_sits_in_the_36px_blue_square():
    _, _, html = reset_email(CODE, "Ada", LOGO)
    assert 'bgcolor="#2563EB"' in html
    (tag,) = re.findall(r"<img [^>]*>", html)
    attrs = dict(re.findall(r'(\w[\w-]*)="([^"]*)"', tag))
    assert attrs["src"] == LOGO and attrs["src"].startswith("https://")
    assert attrs["width"] == "36" and attrs["height"] == "36" and attrs["border"] == "0"
    assert attrs["alt"] == "Triage"                                     # shown when images are blocked
    assert "display: block" in attrs["style"]


# ---------- the logo URL ----------

@pytest.mark.parametrize("value, expected", [(None, None), ("", None), ("  ", None), (LOGO, LOGO),
                                             (f'"{LOGO}"', LOGO), (f"  {LOGO} ", LOGO)])
def test_logo_urls_are_cleaned(value, expected):
    assert parse_logo_url(value) == expected


@pytest.mark.parametrize("value", ["http://cdn.example.dev/logo.png", "javascript:alert(1)", "logo.png",
                                   'https://cdn.example.dev/a.png" onerror="x', "https://cdn.example.dev/a b.png",
                                   "https:///nohost.png"])
def test_unsafe_logo_urls_are_refused_without_echoing_them(value):
    with pytest.raises(LogoUrlError) as caught:
        parse_logo_url(value)
    assert "example" not in str(caught.value) and "alert" not in str(caught.value)


RESEND_ENV = {"RESEND_API_KEY": "re_test_" + "x" * 24, "TRIAGE_EMAIL_FROM": "Triage <no-reply@send.example.dev>"}


def test_the_logo_url_comes_from_the_environment():
    assert sender_from_env(False, {**RESEND_ENV, "TRIAGE_EMAIL_LOGO_URL": LOGO}).logo_url == LOGO
    assert sender_from_env(False, RESEND_ENV).logo_url is None


def test_a_bad_logo_url_still_sends_with_the_plain_mark():
    sender = sender_from_env(False, {**RESEND_ENV, "TRIAGE_EMAIL_LOGO_URL": "http://insecure.example.dev/l.png"})
    assert sender.name == "resend" and sender.logo_url is None
    assert "TRIAGE_EMAIL_LOGO_URL" in sender.logo_problem and "insecure" not in sender.logo_problem


# ---------- what reaches Resend ----------

@pytest.mark.parametrize("name", ["Ada", None])
@pytest.mark.parametrize("logo", [LOGO, None])
def test_no_address_user_id_or_token_in_the_message(name, logo):
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "e1"})
    sender = ResendSender(RESEND_ENV["RESEND_API_KEY"], RESEND_ENV["TRIAGE_EMAIL_FROM"],
                          transport=httpx.MockTransport(handler), logo_url=logo)
    sender.send_reset_code(EMAIL, USER_ID, CODE, "idem-key-1", display_name=name)
    (body,) = seen
    assert body["to"] == [EMAIL] and body["subject"] == "Reset your Triage password"
    for part in (body["subject"], body["text"], body["html"]):
        assert EMAIL not in part and "ada@" not in part and USER_ID not in part and "idem-key-1" not in part
        assert not TOKEN_SHAPES.search(part)
    assert CODE in body["text"] and CODE in body["html"]


# ---------- the startup check of the logo URL (one HEAD, mocked) ----------

def logo_server(respond):
    """A mock transport: `respond(request)` answers the logo HEAD; Resend sends get a 200. Records every request."""
    seen = []

    def handler(request):
        seen.append(request)
        if request.url.host == "api.resend.com":
            return httpx.Response(200, json={"id": "e1"})
        return respond(request)
    return httpx.MockTransport(handler), seen


def started(respond, caplog):
    transport, seen = logo_server(respond)
    sender = ResendSender(RESEND_ENV["RESEND_API_KEY"], RESEND_ENV["TRIAGE_EMAIL_FROM"], transport=transport,
                          logo_url=LOGO)
    with caplog.at_level(logging.WARNING, logger="triage.api"):
        set_up_email(SimpleNamespace(email_sender=sender))
    return sender, seen


def test_a_logo_that_serves_an_image_is_kept(caplog):
    sender, seen = started(lambda r: httpx.Response(200, headers={"content-type": "image/png"}), caplog)
    assert sender.logo_url == LOGO and sender.logo_problem is None
    assert [(r.method, str(r.url)) for r in seen] == [("HEAD", LOGO)]
    assert "logo" not in caplog.text


def raise_(error):
    def respond(request):
        raise error
    return respond


@pytest.mark.parametrize("respond, reason", [
    (lambda r: httpx.Response(404), "TRIAGE_EMAIL_LOGO_URL returned 404"),
    (lambda r: httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}),
     "TRIAGE_EMAIL_LOGO_URL is not an image (content-type text/html)"),
    (lambda r: httpx.Response(200), "TRIAGE_EMAIL_LOGO_URL is not an image (content-type missing)"),
    (raise_(httpx.ReadTimeout("slow")), "TRIAGE_EMAIL_LOGO_URL is unreachable (timed out)"),
    (raise_(httpx.ConnectError("no such host")), "TRIAGE_EMAIL_LOGO_URL is unreachable"),
    (raise_(RuntimeError("anything else")), "TRIAGE_EMAIL_LOGO_URL could not be checked (RuntimeError)"),
])
def test_a_broken_logo_falls_back_to_the_square_with_one_warning(respond, reason, caplog):
    sender, _ = started(respond, caplog)
    assert sender.logo_url is None and sender.logo_problem == reason
    warnings = [m for m in caplog.messages if "logo" in m]
    assert warnings == [f"reset emails will show the plain logo mark: {reason}"]
    assert LOGO not in caplog.text and "example" not in caplog.text
    _, _, html = reset_email(CODE, "Ada", sender.logo_url)
    assert "<img" not in html and 'bgcolor="#2563EB"' in html


def test_the_logo_is_checked_once_not_per_email(caplog):
    sender, seen = started(lambda r: httpx.Response(200, headers={"content-type": "image/png"}), caplog)
    for n in range(3):
        sender.send_reset_code(EMAIL, USER_ID, CODE, f"idem-{n}", display_name="Ada")
    assert [r.method for r in seen] == ["HEAD", "POST", "POST", "POST"]


def test_no_logo_url_means_no_request(caplog):
    transport, seen = logo_server(lambda r: httpx.Response(500))
    sender = ResendSender(RESEND_ENV["RESEND_API_KEY"], RESEND_ENV["TRIAGE_EMAIL_FROM"], transport=transport)
    set_up_email(SimpleNamespace(email_sender=sender))
    assert seen == [] and sender.logo_problem is None


def test_redirects_to_an_image_are_followed():
    def respond(request):
        if request.url.path == "/triage-logo.png":
            return httpx.Response(301, headers={"location": "https://cdn.example.dev/v2/logo.png"})
        return httpx.Response(200, headers={"content-type": "image/png"})
    transport, _ = logo_server(respond)
    assert logo_url_problem(LOGO, transport) is None

