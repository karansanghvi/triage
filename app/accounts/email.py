"""Sending password reset codes. Blocking: call from a worker thread in async code.

Every sender has send_reset_code(to, user_id, code, idempotency_key) and `available`. Chosen at startup
(sender_from_env), first match wins:

    1. TRIAGE_DEV_EMAIL_LOG=on   ConsoleDevSender: logs "reset code for user <id>: 123456" to the server console.
                                 Local development only: refused (the server won't start) with a non-SQLite
                                 DATABASE_URL, TRIAGE_PUBLIC=on, a non-localhost TRIAGE_CORS_ORIGINS entry or
                                 TRIAGE_TRUSTED_PROXY_HOPS above 0.
    2. TRIAGE_FAKE_MODELS=1      FakeEmailSender: keeps messages in memory, sends nothing.
    3. RESEND_API_KEY and TRIAGE_EMAIL_FROM
                                 ResendSender: POST https://api.resend.com/emails
                                 (https://resend.com/docs/api-reference/emails/send-email).
    4. otherwise                 UnavailableSender: the reset endpoints answer 503 email_unavailable.

Never logged or put in an error: the address, the code, the API key, or Resend's response body. A failed send
raises EmailSendError with a short kind ("resend_403", "timeout", ...) and nothing else.
"""
import logging
import os
from dataclasses import dataclass, field
from typing import Mapping, Optional
from urllib.parse import urlsplit

import httpx
from sqlalchemy.engine import make_url

RESEND_KEY_ENV, EMAIL_FROM_ENV, DEV_EMAIL_LOG_ENV = "RESEND_API_KEY", "TRIAGE_EMAIL_FROM", "TRIAGE_DEV_EMAIL_LOG"
RESEND_URL = "https://api.resend.com/emails"
USER_AGENT = "triage/0.5"            # Resend refuses requests without a User-Agent (403)
TIMEOUT = httpx.Timeout(10.0)
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")

dev_logger = logging.getLogger("triage.email")


class EmailSendError(Exception):
    """A send that failed. kind is safe to log: it never contains the address, code, key or response body."""

    def __init__(self, kind: str):
        super().__init__(kind)
        self.kind = kind


# ---------- the message ----------

SUBJECT = "Your Triage password reset code"


def reset_email(code: str) -> tuple:
    """(subject, text, html). The HTML contains only the code's digits, nothing a user typed."""
    text = (f"Your Triage password reset code is: {code}\n\n"
            "It expires in 10 minutes.\n\n"
            "If you didn't ask for this, ignore this email: your password won't change.\n")
    html = ('<div style="font-family: sans-serif; font-size: 15px; line-height: 1.5; color: #222;">'
            "<p>Your Triage password reset code is:</p>"
            f'<p style="font-size: 28px; font-weight: bold; letter-spacing: 4px;">{code}</p>'
            "<p>It expires in 10 minutes.</p>"
            '<p style="color: #666;">If you didn\'t ask for this, ignore this email: your password won\'t '
            "change.</p></div>")
    return SUBJECT, text, html


# ---------- senders ----------

class ResendSender:
    available = True
    name = "resend"

    def __init__(self, api_key: str, sender: str, transport: Optional[httpx.BaseTransport] = None):
        self._api_key, self.sender, self._transport = api_key, sender, transport

    def __repr__(self) -> str:
        return "ResendSender()"

    def send_reset_code(self, to: str, user_id: str, code: str, idempotency_key: str) -> None:
        subject, text, html = reset_email(code)
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json",
                   "User-Agent": USER_AGENT, "Idempotency-Key": idempotency_key}
        body = {"from": self.sender, "to": [to], "subject": subject, "text": text, "html": html}
        try:
            with httpx.Client(transport=self._transport, timeout=TIMEOUT) as client:
                response = client.post(RESEND_URL, headers=headers, json=body)
        except httpx.TimeoutException:
            raise EmailSendError("timeout") from None
        except httpx.HTTPError:
            raise EmailSendError("network") from None
        if response.status_code != 200:
            raise EmailSendError(f"resend_{response.status_code}")


@dataclass
class FakeEmailSender:
    """Tests and fake mode: remembers each message, sends nothing."""
    available = True
    name = "fake"
    sent: list = field(default_factory=list)

    def send_reset_code(self, to: str, user_id: str, code: str, idempotency_key: str) -> None:
        self.sent.append({"to": to, "user_id": user_id, "code": code, "idempotency_key": idempotency_key})


class ConsoleDevSender:
    """Local development: the code goes to the server console, never anywhere else (no address)."""
    available = True
    name = "console"

    def __init__(self):
        if not dev_logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            dev_logger.addHandler(handler)
            dev_logger.setLevel(logging.INFO)
            dev_logger.propagate = False         # not into any log aggregation configured on the root logger

    def send_reset_code(self, to: str, user_id: str, code: str, idempotency_key: str) -> None:
        dev_logger.info("reset code for user %s: %s", user_id, code)


@dataclass(frozen=True)
class UnavailableSender:
    """Email isn't configured: reset endpoints answer 503. missing: variable names, never values."""
    missing: tuple = ()
    available = False
    name = "unavailable"

    def send_reset_code(self, to: str, user_id: str, code: str, idempotency_key: str) -> None:
        raise EmailSendError("unavailable")


# ---------- choosing one ----------

def _on(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in ("on", "1", "true", "yes")


def dev_log_refusals(environ: Mapping[str, str]) -> list:
    """Why this configuration isn't local development (names and reasons, never values). Empty: it is."""
    reasons = []
    url = (environ.get("DATABASE_URL") or "").strip()
    if url:
        try:
            sqlite = make_url(url).get_backend_name() == "sqlite"
        except Exception:
            sqlite = False
        if not sqlite:
            reasons.append("DATABASE_URL isn't a SQLite file")
    if _on(environ.get("TRIAGE_PUBLIC")):
        reasons.append("TRIAGE_PUBLIC is on")
    origins = [o.strip() for o in (environ.get("TRIAGE_CORS_ORIGINS") or "").split(",") if o.strip()]
    if any(urlsplit(o).hostname not in LOCAL_HOSTS for o in origins):
        reasons.append("TRIAGE_CORS_ORIGINS lists an origin that isn't localhost")
    hops = (environ.get("TRIAGE_TRUSTED_PROXY_HOPS") or "0").strip() or "0"
    if not hops.isdigit() or int(hops) > 0:
        reasons.append("TRIAGE_TRUSTED_PROXY_HOPS is above 0 (the server is behind a proxy)")
    return reasons


def sender_from_env(fake: bool, environ: Optional[Mapping[str, str]] = None):
    """The sender for this configuration (see the module docstring). Raises RuntimeError when
    TRIAGE_DEV_EMAIL_LOG=on outside local development."""
    environ = os.environ if environ is None else environ
    if (environ.get(DEV_EMAIL_LOG_ENV) or "").strip().lower() == "on":
        reasons = dev_log_refusals(environ)
        if reasons:
            raise RuntimeError(f"refusing to start: {DEV_EMAIL_LOG_ENV}=on prints reset codes to the console and "
                               f"is only for local development, but {'; '.join(reasons)}. Turn it off and set "
                               f"{RESEND_KEY_ENV} and {EMAIL_FROM_ENV} instead.")
        return ConsoleDevSender()
    if fake:
        return FakeEmailSender()
    api_key, sender = (environ.get(RESEND_KEY_ENV) or "").strip(), (environ.get(EMAIL_FROM_ENV) or "").strip()
    if api_key and sender:
        return ResendSender(api_key, sender)
    return UnavailableSender(tuple(n for n, v in ((RESEND_KEY_ENV, api_key), (EMAIL_FROM_ENV, sender)) if not v))
