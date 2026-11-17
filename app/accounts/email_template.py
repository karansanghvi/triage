"""The password reset email: subject, plain text and HTML.

Built for email clients: a table layout with every style inline (no <style> block, external CSS, web font or
SVG), a hidden preheader, 600px max width. No tracking pixel and no tracked links: the only link is a plain
https://gettriage.dev.

Inputs are only the code, the user's display name (user input: whitespace collapsed, HTML-escaped) and the
optional logo URL (TRIAGE_EMAIL_LOGO_URL, checked by parse_logo_url). Never the email address, the user id or
any token.
"""
import html
import re
from typing import Optional
from urllib.parse import urlsplit

SUBJECT = "Reset your Triage password"
PREHEADER = "Your code expires in 10 minutes."
SITE_URL = "https://gettriage.dev"

HEADING = "Reset your password"
INTRO = ("We received a request to reset the password for your Triage account. Enter this code to continue:")
EXPIRY = "This code expires in 10 minutes and can be used only once."
IGNORE = "Didn't ask for this? You can safely ignore this email; your password won't change."
NEVER_ASK = "Triage will never ask you for this code by email, phone or chat."
TAGLINE = "Triage · Every prompt, routed to the right model."
WHY = "You're receiving this because a password reset was requested for your account."

FONT = "-apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
MONO = "'SFMono-Regular', Menlo, Consolas, 'Liberation Mono', 'Courier New', monospace"


class LogoUrlError(ValueError):
    """TRIAGE_EMAIL_LOGO_URL isn't usable. The message never contains the value."""


def parse_logo_url(value: Optional[str]) -> Optional[str]:
    """None when unset; otherwise an https URL with a host and nothing that could break out of an attribute."""
    text = (value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    if not text:
        return None
    parts = urlsplit(text)
    if parts.scheme != "https" or not parts.hostname or re.search(r"[\s\"'<>`\\]", text):
        raise LogoUrlError("TRIAGE_EMAIL_LOGO_URL must be an https:// URL of a hosted PNG")
    return text


def greeting(display_name: Optional[str]) -> str:
    """ "Hi <name>," or "Hi," (plain text: the HTML escapes it separately)."""
    name = " ".join((display_name or "").split())
    return f"Hi {name}," if name else "Hi,"


def reset_email(code: str, display_name: Optional[str] = None, logo_url: Optional[str] = None) -> tuple:
    """(subject, text, html)."""
    hello = greeting(display_name)
    text = (f"{HEADING}\n\n"
            f"{hello}\n\n"
            f"{INTRO}\n\n"
            f"    {code}\n\n"
            f"{EXPIRY}\n\n"
            "----\n\n"
            f"{IGNORE} {NEVER_ASK}\n\n"
            "--\n"
            f"{TAGLINE}\n"
            f"{SITE_URL}\n"
            f"{WHY}\n")
    return SUBJECT, text, _html(code, html.escape(hello, quote=True), logo_url)


def _logo(logo_url: Optional[str]) -> str:
    if logo_url:
        inner = (f'<img src="{html.escape(logo_url, quote=True)}" width="36" height="36" alt="" '
                 'style="display: block; width: 36px; height: 36px; border: 0; outline: none; '
                 'text-decoration: none; border-radius: 8px;">')
    else:
        inner = "&nbsp;"                         # the blue square alone still reads as the mark
    return ('<table role="presentation" cellpadding="0" cellspacing="0" border="0"><tr>'
            '<td width="36" height="36" bgcolor="#2563EB" style="width: 36px; height: 36px; background-color: '
            f'#2563EB; border-radius: 8px; font-size: 1px; line-height: 1px;">{inner}</td>'
            f'<td style="padding-left: 10px; font-family: {FONT}; font-size: 20px; font-weight: 700; '
            'color: #111827; letter-spacing: -0.2px;">Triage</td>'
            "</tr></table>")


def _p(content: str, style: str = "") -> str:
    return (f'<p style="margin: 0 0 16px 0; font-family: {FONT}; font-size: 15px; line-height: 24px; '
            f'color: #374151;{style}">{content}</p>')


def _html(code: str, hello: str, logo_url: Optional[str]) -> str:
    e = html.escape
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<meta name="supported-color-schemes" content="light">
<title>{e(SUBJECT)}</title>
</head>
<body style="margin: 0; padding: 0; width: 100%; background-color: #F4F5F8;">
<div style="display: none; max-height: 0; max-width: 0; overflow: hidden; opacity: 0; mso-hide: all; font-size: 1px; line-height: 1px; color: #F4F5F8;">{e(PREHEADER)}</div>
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" bgcolor="#F4F5F8" style="background-color: #F4F5F8;">
<tr><td align="center" style="padding: 32px 16px;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="max-width: 600px; width: 100%;">
<tr><td style="padding: 0 4px 24px 4px;">{_logo(logo_url)}</td></tr>
<tr><td bgcolor="#FFFFFF" style="background-color: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 16px; padding: 40px;">
<h1 style="margin: 0 0 24px 0; font-family: {FONT}; font-size: 24px; line-height: 32px; font-weight: 700; color: #111827;">{e(HEADING)}</h1>
{_p(hello)}
{_p(e(INTRO))}
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="margin: 8px 0 24px 0;">
<tr><td align="center" bgcolor="#EFF4FF" style="background-color: #EFF4FF; border: 1px solid #C7D7FE; border-radius: 12px; padding: 20px 12px 20px 22px; font-family: {MONO}; font-size: 34px; line-height: 40px; font-weight: 700; letter-spacing: 10px; color: #1D4ED8;">{e(code)}</td></tr>
</table>
{_p(e(EXPIRY), " margin-bottom: 24px;")}
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"><tr><td style="border-top: 1px solid #E5E7EB; font-size: 1px; line-height: 1px; height: 1px;">&nbsp;</td></tr></table>
<p style="margin: 24px 0 0 0; font-family: {FONT}; font-size: 13px; line-height: 20px; color: #6B7280;">{e(IGNORE)} {e(NEVER_ASK)}</p>
</td></tr>
<tr><td align="center" style="padding: 24px 16px 0 16px; font-family: {FONT}; font-size: 12px; line-height: 18px; color: #6B7280;">
<p style="margin: 0 0 4px 0;">{e(TAGLINE)}</p>
<p style="margin: 0 0 4px 0;"><a href="{SITE_URL}" style="color: #6B7280; text-decoration: underline;">gettriage.dev</a></p>
<p style="margin: 0;">{e(WHY)}</p>
</td></tr>
</table>
</td></tr>
</table>
</body>
</html>
"""
