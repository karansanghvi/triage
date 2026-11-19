"""The emails Triage sends: the password reset code and the welcome email. Each is (subject, text, html).

Built for email clients: a table layout with every style inline (no <style> block, external CSS, web font or
SVG), a hidden preheader, 600px max width. No tracking pixel and no tracked links: the only links are plain links
to TRIAGE_APP_URL pages, left out entirely when it isn't set. The shared pieces (the page wrapper, the logo header
and the footer) are used by both emails.

Inputs are only the code, the user's display name (user input: whitespace collapsed, HTML-escaped), the optional
logo URL (TRIAGE_EMAIL_LOGO_URL) and the app and API URLs (TRIAGE_APP_URL, TRIAGE_PUBLIC_API_URL), each checked
at startup. Never the email address, the user id or any token.
"""
import html
import re
from typing import Optional
from urllib.parse import urlsplit

LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")
DEFAULT_API_URL = "https://api.gettriage.dev"
_UNSAFE = re.compile(r"[\s\"'<>`\\]")

TAGLINE = "Triage · Every prompt, routed to the right model."
FONT = "-apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
MONO = "'SFMono-Regular', Menlo, Consolas, 'Liberation Mono', 'Courier New', monospace"

# ----- the reset email -----
SUBJECT = "Reset your Triage password"
PREHEADER = "Your code expires in 10 minutes."
HEADING = "Reset your password"
INTRO = "We received a request to reset the password for your Triage account. Enter this code to continue:"
EXPIRY = "This code expires in 10 minutes and can be used only once."
IGNORE = "Didn't ask for this? You can safely ignore this email; your password won't change."
NEVER_ASK = "Triage will never ask you for this code by email, phone or chat."
WHY = "You're receiving this because a password reset was requested for your account."

# ----- the welcome email -----
WELCOME_SUBJECT = "Welcome to Triage"
WELCOME_PREHEADER = "Your first routed request is three steps away."
WELCOME_HEADING = "Welcome to Triage"
WELCOME_INTRO = ("You now have an OpenAI-compatible router that sends every prompt to the cheapest model that can "
                 "answer it well, so you pay less without your answers getting worse.")
STEPS_INTRO = "Get your first request through in three steps:"
STEPS = (
    ("Add a provider key",
     "Save your OpenAI, Anthropic, Groq or Gemini key. It's encrypted the moment you save it, and only the last 4 "
     "characters are ever shown.",
     "Add a provider key →", "/settings/provider-keys"),
    ("Create a Triage API key",
     "Your app sends this key to Triage instead of your provider keys. It's shown once, so save it somewhere safe.",
     "Create an API key →", "/settings/api-keys"),
    ("Point your app at Triage", "Keep your OpenAI code and change two lines:", None, None),
)
NEXT_LABEL = "What happens next:"
NEXT = ("for each request, Triage predicts how hard the prompt is, picks the cheapest model that clears your quality "
        "bar, and falls back automatically if a provider has a bad day. Every response tells you which model "
        "answered, why, and how much you saved, and your dashboard adds it all up.")
BUTTON, BUTTON_PATH = "Open your settings", "/settings/get-started"
WELCOME_WHY = ("You're receiving this because you created a Triage account with this email. If that wasn't you, "
               "use Forgot password on the log in page to take over the account, then delete it in "
               "Settings → Account.")
API_KEY_PLACEHOLDER = "trk_..."


# ---------- startup checks of the URLs ----------

class LogoUrlError(ValueError):
    """TRIAGE_EMAIL_LOGO_URL isn't usable. The message never contains the value."""


class PublicUrlError(ValueError):
    """TRIAGE_APP_URL or TRIAGE_PUBLIC_API_URL isn't usable. The message never contains the value."""


def _unquote(value: Optional[str]) -> str:
    text = (value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


def parse_logo_url(value: Optional[str]) -> Optional[str]:
    """None when unset; otherwise an https URL with a host and nothing that could break out of an attribute."""
    text = _unquote(value)
    if not text:
        return None
    parts = urlsplit(text)
    if parts.scheme != "https" or not parts.hostname or _UNSAFE.search(text):
        raise LogoUrlError("TRIAGE_EMAIL_LOGO_URL must be an https:// URL of a hosted PNG")
    return text


def parse_public_url(value: Optional[str], name: str, strip_v1: bool = False) -> Optional[str]:
    """None when unset; otherwise the URL without a trailing slash (and, for the API, without a trailing /v1).
    https is required, except http for localhost. No query, fragment, credentials or attribute-breaking characters."""
    text = _unquote(value)
    if not text:
        return None
    try:
        parts = urlsplit(text)
        host = parts.hostname
        parts.port                                  # raises on a malformed port
    except ValueError:
        raise PublicUrlError(f"{name} isn't a valid URL") from None
    local = host in LOCAL_HOSTS
    if not host or parts.scheme not in ("https", "http") or (parts.scheme == "http" and not local):
        raise PublicUrlError(f"{name} must be an https:// URL (http:// only for localhost)")
    if parts.query or parts.fragment or "@" in parts.netloc or _UNSAFE.search(text):
        raise PublicUrlError(f"{name} must be a plain URL, without a query, fragment or credentials")
    path = parts.path.rstrip("/")
    if strip_v1 and path.endswith("/v1"):
        path = path[:-3]
    return f"{parts.scheme}://{parts.netloc}{path}"


def greeting(display_name: Optional[str]) -> str:
    """ "Hi <name>," or "Hi," (plain text: the HTML escapes it separately)."""
    name = " ".join((display_name or "").split())
    return f"Hi {name}," if name else "Hi,"


def _site_label(app_url: str) -> str:
    """The app URL as shown in the footer: "gettriage.dev", "localhost:5173"."""
    parts = urlsplit(app_url)
    return parts.netloc + parts.path


# ---------- the reset email ----------

def reset_email(code: str, display_name: Optional[str] = None, logo_url: Optional[str] = None,
                app_url: Optional[str] = None) -> tuple:
    """(subject, text, html)."""
    hello = greeting(display_name)
    text = (f"{HEADING}\n\n"
            f"{hello}\n\n"
            f"{INTRO}\n\n"
            f"    {code}\n\n"
            f"{EXPIRY}\n\n"
            "----\n\n"
            f"{IGNORE} {NEVER_ASK}\n\n"
            + _text_footer(WHY, app_url))
    e = html.escape
    card = (_h1(HEADING)
            + _p(e(hello, quote=True))
            + _p(e(INTRO))
            + '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
              'style="margin: 8px 0 24px 0;">\n'
              '<tr><td align="center" bgcolor="#EFF4FF" style="background-color: #EFF4FF; border: 1px solid #C7D7FE; '
              'border-radius: 12px; padding: 20px 12px 20px 22px; '
              f'font-family: {MONO}; font-size: 34px; line-height: 40px; font-weight: 700; letter-spacing: 10px; '
              f'color: #1D4ED8;">{e(code)}</td></tr>\n</table>\n'
            + _p(e(EXPIRY), " margin-bottom: 24px;")
            + _divider()
            + f'<p style="margin: 24px 0 0 0; font-family: {FONT}; font-size: 13px; line-height: 20px; '
              f'color: #6B7280;">{e(IGNORE)} {e(NEVER_ASK)}</p>\n')
    return SUBJECT, text, _page(SUBJECT, PREHEADER, logo_url, card, WHY, app_url)


# ---------- the welcome email ----------

def _code_lines(api_url: str) -> list:
    """The two-line change, as (plain text, html) per line. The string values are highlighted."""
    def s(value: str) -> str:
        return f'<span style="color: #86EFAC;">"{html.escape(value)}"</span>'
    base = f"{api_url}/v1"
    return [
        ("client = OpenAI(", "client = OpenAI("),
        (f'    base_url="{base}",', f"&nbsp;&nbsp;&nbsp;&nbsp;base_url={s(base)},"),
        (f'    api_key="{API_KEY_PLACEHOLDER}",', f"&nbsp;&nbsp;&nbsp;&nbsp;api_key={s(API_KEY_PLACEHOLDER)},"),
        (")", ")"),
        ('# then use model="auto"', '<span style="color: #94A3B8;"># then use model="auto"</span>'),
    ]


def welcome_email(display_name: Optional[str] = None, logo_url: Optional[str] = None,
                  app_url: Optional[str] = None, api_url: Optional[str] = None) -> tuple:
    """(subject, text, html). Without app_url: the same email without the step links and the button."""
    api_url = api_url or DEFAULT_API_URL
    hello, code = greeting(display_name), _code_lines(api_url)

    lines = [WELCOME_HEADING, "", hello, "", WELCOME_INTRO, "", STEPS_INTRO, ""]
    for number, (title, body, link, path) in enumerate(STEPS, 1):
        lines += [f"{number}. {title}", f"   {body}"]
        if number == 3:
            lines += [""] + [f"       {plain}" for plain, _ in code]
        if link and app_url:
            lines.append(f"   {link} {app_url}{path}")
        lines.append("")
    lines += [f"{NEXT_LABEL} {NEXT}", ""]
    if app_url:
        lines += [f"{BUTTON}: {app_url}{BUTTON_PATH}", ""]
    text = "\n".join(lines) + "\n" + _text_footer(WELCOME_WHY, app_url)

    e = html.escape
    steps = "".join(_step(number, title, body, link, path, app_url, code if number == 3 else None)
                    for number, (title, body, link, path) in enumerate(STEPS, 1))
    card = (_h1(WELCOME_HEADING)
            + _p(e(hello, quote=True))
            + _p(e(WELCOME_INTRO))
            + _p(f'<strong style="color: #111827;">{e(STEPS_INTRO)}</strong>', " margin-bottom: 20px;")
            + '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">\n'
            + steps + "</table>\n"
            + _divider()
            + _p(f'<strong style="color: #111827;">{e(NEXT_LABEL)}</strong> {e(NEXT)}', " margin: 24px 0 0 0;")
            + (_button(BUTTON, app_url + BUTTON_PATH) if app_url else ""))
    return WELCOME_SUBJECT, text, _page(WELCOME_SUBJECT, WELCOME_PREHEADER, logo_url, card, WELCOME_WHY, app_url)


def _step(number: int, title: str, body: str, link: Optional[str], path: Optional[str], app_url: Optional[str],
          code: Optional[list]) -> str:
    e = html.escape
    content = (f'<p style="margin: 2px 0 4px 0; font-family: {FONT}; font-size: 15px; line-height: 22px; '
               f'font-weight: 700; color: #111827;">{e(title)}</p>'
               f'<p style="margin: 0; font-family: {FONT}; font-size: 14px; line-height: 22px; color: #374151;">'
               f'{e(body)}</p>')
    if code:
        content += ('<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
                    'style="margin: 10px 0 0 0;"><tr><td bgcolor="#0F172A" style="background-color: #0F172A; '
                    'border-radius: 10px; padding: 14px 16px; '
                    f'font-family: {MONO}; font-size: 13px; line-height: 20px; color: #E2E8F0; '
                    'word-break: break-all; overflow-wrap: anywhere;">'   # long URLs wrap on phones
                    + "<br>".join(markup for _, markup in code) + "</td></tr></table>")
    if link and app_url:
        content += (f'<p style="margin: 6px 0 0 0; font-family: {FONT}; font-size: 14px; line-height: 22px;">'
                    f'<a href="{e(app_url + path, quote=True)}" style="color: #2563EB; font-weight: 600; '
                    f'text-decoration: none;">{e(link)}</a></p>')
    return ('<tr><td width="40" valign="top" style="width: 26px; min-width: 26px; padding: 0 14px 22px 0;">'
            '<table role="presentation" cellpadding="0" cellspacing="0" border="0"><tr>'
            '<td width="26" height="26" align="center" valign="middle" bgcolor="#EFF4FF" style="width: 26px; '
            'min-width: 26px; height: 26px; background-color: #EFF4FF; border: 1px solid #C7D7FE; border-radius: 13px; '
            f'font-family: {FONT}; font-size: 13px; line-height: 24px; font-weight: 700; color: #1D4ED8; '
            f'text-align: center;">{number}</td></tr></table></td>'
            f'<td valign="top" style="padding: 0 0 22px 0;">{content}</td></tr>\n')


def _button(label: str, url: str) -> str:
    e = html.escape
    return ('<table role="presentation" cellpadding="0" cellspacing="0" border="0" style="margin: 28px 0 0 0;">'
            '<tr><td bgcolor="#2563EB" style="background-color: #2563EB; border-radius: 10px;">'
            f'<a href="{e(url, quote=True)}" style="display: inline-block; padding: 13px 24px; font-family: {FONT}; '
            'font-size: 15px; line-height: 20px; font-weight: 700; color: #FFFFFF; text-decoration: none; '
            f'border-radius: 10px;">{e(label)}</a></td></tr></table>\n')


# ---------- shared pieces ----------

def _text_footer(why: str, app_url: Optional[str]) -> str:
    return "--\n" + TAGLINE + "\n" + (f"{app_url}\n" if app_url else "") + why + "\n"


def _logo(logo_url: Optional[str]) -> str:
    if logo_url:
        # Decorative (alt=""): the "Triage" wordmark sits right beside it, also when images are blocked
        inner = (f'<img src="{html.escape(logo_url, quote=True)}" width="36" height="36" border="0" alt="" '
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


def _h1(text: str) -> str:
    return (f'<h1 style="margin: 0 0 24px 0; font-family: {FONT}; font-size: 24px; line-height: 32px; '
            f'font-weight: 700; color: #111827;">{html.escape(text)}</h1>\n')


def _p(content: str, style: str = "") -> str:
    return (f'<p style="margin: 0 0 16px 0; font-family: {FONT}; font-size: 15px; line-height: 24px; '
            f'color: #374151;{style}">{content}</p>\n')


def _divider() -> str:
    return ('<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"><tr><td '
            'style="border-top: 1px solid #E5E7EB; font-size: 1px; line-height: 1px; height: 1px;">&nbsp;</td>'
            '</tr></table>\n')


def _footer(why: str, app_url: Optional[str]) -> str:
    e = html.escape
    link = (f'<p style="margin: 0 0 4px 0;"><a href="{e(app_url, quote=True)}" style="color: #6B7280; '
            f'text-decoration: underline;">{e(_site_label(app_url))}</a></p>\n') if app_url else ""
    return (f'<tr><td align="center" style="padding: 24px 16px 0 16px; font-family: {FONT}; font-size: 12px; '
            'line-height: 18px; color: #6B7280;">\n'
            f'<p style="margin: 0 0 4px 0;">{e(TAGLINE)}</p>\n'
            f'{link}'
            f'<p style="margin: 0;">{e(why)}</p>\n'
            '</td></tr>\n')


def _page(subject: str, preheader: str, logo_url: Optional[str], card: str, why: str,
          app_url: Optional[str]) -> str:
    e = html.escape
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<meta name="supported-color-schemes" content="light">
<title>{e(subject)}</title>
</head>
<body style="margin: 0; padding: 0; width: 100%; background-color: #F4F5F8;">
<div style="display: none; max-height: 0; max-width: 0; overflow: hidden; opacity: 0; mso-hide: all; font-size: 1px; line-height: 1px; color: #F4F5F8;">{e(preheader)}</div>
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" bgcolor="#F4F5F8" style="background-color: #F4F5F8;">
<tr><td align="center" style="padding: 32px 16px;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="max-width: 600px; width: 100%;">
<tr><td style="padding: 0 4px 24px 4px;">{_logo(logo_url)}</td></tr>
<tr><td bgcolor="#FFFFFF" style="background-color: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 16px; padding: 40px;">
{card}</td></tr>
{_footer(why, app_url)}</table>
</td></tr>
</table>
</body>
</html>
"""
