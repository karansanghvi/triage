"""The free demo (Phase 5): visitor identity and usage limits for requests without any key.

Demo requests (no Triage key, no provider keys; TRIAGE_PUBLIC on) run on the server's own free-tier keys, so they
are limited per visitor and in total, every UTC day:

    per visitor     visitor_daily_requests (10)
    all visitors    daily_requests (300), daily_tokens (150,000), daily_model_calls (600)
    all visitors    per_minute_requests (4) in any rolling 60 s (Groq's gpt-oss allows 8,000 tokens/minute)

Public BYOK requests (provider keys, no Triage key) only get a generous per-visitor daily cap
(public_byok_visitor_daily_requests, 200): the user pays the provider, so this only protects the server.

Visitors are identified by client IP (client_ip, honouring TRIAGE_TRUSTED_PROXY_HOPS), and only ever stored as a
keyed hash (visitor_id). Raw IPs are never stored or logged.

The DemoLimiter keeps its counters in SQLite (TRIAGE_DEMO_DB) so they survive restarts, or in memory for tests.
On a host with an ephemeral disk the file is lost on every redeploy or restart, and every limit starts over; the
providers' own free-tier quotas are then the only backstop.
"""
import hashlib
import hmac
import ipaddress
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Literal, Optional

from app.registry import PROJECT_ROOT

TRUSTED_PROXY_HOPS_ENV = "TRIAGE_TRUSTED_PROXY_HOPS"
DEMO_DB_ENV = "TRIAGE_DEMO_DB"
DEFAULT_DEMO_DB = PROJECT_ROOT / "data" / "demo" / "usage.sqlite3"
FAKE_DEMO_DB = PROJECT_ROOT / "data" / "demo" / "fake_usage.sqlite3"     # TRIAGE_FAKE_MODELS=1 runs
VISITOR_ID_HEX_CHARS = 16
UNKNOWN_CLIENT = "unknown"


# ---------- visitor identity ----------

def client_ip(request, trusted_hops: int = 0) -> str:
    """The visitor's IP address as seen by the closest proxy we trust.

    trusted_hops = 0 (the default, safe locally): the direct connection's address. X-Forwarded-For is ignored,
    because any client can send one.
    trusted_hops = N: our N proxies each append the address they received the request from, so the real client
    is the N-th entry from the RIGHT of X-Forwarded-For. Anything a client adds itself sits further left, so a
    faked header can't choose the address used. Too few entries (the request skipped a proxy): the direct
    connection's address.
    `request` is a Starlette Request (or anything with .headers and .client).
    """
    direct = request.client.host if getattr(request, "client", None) and request.client.host else UNKNOWN_CLIENT
    if trusted_hops <= 0:
        return direct
    forwarded = [part.strip() for part in ",".join(request.headers.getlist("x-forwarded-for")).split(",")
                 if part.strip()]
    if len(forwarded) < trusted_hops:
        return direct
    return forwarded[-trusted_hops]


def _visitor_group(ip: str) -> str:
    """IPv4: the address. IPv6: its /64 network (one household or device usually owns a whole /64, so rotating
    addresses inside it doesn't buy a new allowance). An IPv4-mapped IPv6 address counts as the IPv4 address.
    Anything that isn't an IP (e.g. a test client's name) is used as it is."""
    try:
        address = ipaddress.ip_address(ip.strip().strip("[]"))
    except ValueError:
        return ip.strip()
    if address.version == 6 and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    if address.version == 6:
        return str(ipaddress.ip_network(f"{address}/64", strict=False))
    return str(address)


def visitor_id(ip: str, secret: bytes) -> str:
    """A visitor's id: HMAC-SHA256(TRIAGE_HASH_SECRET, "visitor|" + group), first 16 hex characters. Without the
    secret, nobody can check a guessed IP against it, so the stored ids reveal no addresses."""
    group = _visitor_group(ip)
    digest = hmac.new(secret, f"visitor|{group}".encode("utf-8"), hashlib.sha256).hexdigest()
    return digest[:VISITOR_ID_HEX_CHARS]


def trusted_hops_from_env() -> int:
    raw = os.environ.get(TRUSTED_PROXY_HOPS_ENV, "0").strip() or "0"
    hops = int(raw)
    if hops < 0:
        raise ValueError(f"{TRUSTED_PROXY_HOPS_ENV} must be 0 or more, got {raw!r}")
    return hops


# ---------- limits ----------

@dataclass(frozen=True)
class DemoSettings:
    """Demo limits (config/demo.yaml, loaded in a later step). Days are UTC days."""
    visitor_daily_requests: int = 10
    daily_requests: int = 300
    daily_tokens: int = 150_000
    daily_model_calls: int = 600
    per_minute_requests: int = 4
    max_prompt_chars: int = 8_000
    max_tokens: int = 2_048
    public_byok_visitor_daily_requests: int = 200
    retention_days: int = 7


ErrorCode = Literal["demo_limit_reached", "demo_paused", "demo_busy", "public_limit_reached"]
PausedReason = Literal["daily_requests", "daily_tokens", "daily_model_calls"]


class DemoLimitError(Exception):
    """A request the limiter refused. The API turns it into an OpenAI-style 429 with Retry-After.

    code         demo_limit_reached    this visitor used today's demo requests
                 demo_paused           a demo-wide daily budget (requests, tokens or model calls) is used up
                 demo_busy             the demo-wide per-minute limit is reached
                 public_limit_reached  this visitor used today's public BYOK requests
    retry_after  whole seconds until it can succeed: until 00:00 UTC for the daily limits, until the oldest
                 request leaves the 60 s window for demo_busy (always at least 1)
    reason       demo_paused only: which budget ran out (daily_requests, daily_tokens or daily_model_calls)
    """

    def __init__(self, code: ErrorCode, retry_after: int, reason: Optional[PausedReason] = None):
        super().__init__(code if reason is None else f"{code}:{reason}")
        self.code, self.retry_after, self.reason = code, retry_after, reason


@dataclass(frozen=True)
class Admission:
    """A request the limiter let through. Pass it back to settle() when the request has finished.

    kind             "demo" or "byok" (public BYOK)
    visitor          the visitor id it was counted for
    day              the UTC day it was counted on, "YYYY-MM-DD" (settle() applies to this day, even after midnight)
    reserved_tokens  tokens held against daily_tokens until settle() (0 for byok)
    remaining        the visitor's requests left today of this kind, after this one
    limit            the visitor's daily limit of this kind
    """
    kind: Literal["demo", "byok"]
    visitor: str
    day: str
    reserved_tokens: int
    remaining: int
    limit: int


@dataclass(frozen=True)
class DemoStatus:
    """Today's demo state for one visitor (GET /v1/demo/status). Reading it never counts as a request.

    paused         True if any demo-wide daily budget is used up (requests >= daily_requests,
                   tokens >= daily_tokens or calls >= daily_model_calls)
    paused_reason  which one, or None
    used           the visitor's demo requests today
    remaining      max(0, visitor_daily_requests - used)
    limit          visitor_daily_requests
    resets_at      the next 00:00 UTC (an aware datetime)
    """
    paused: bool
    paused_reason: Optional[PausedReason]
    used: int
    remaining: int
    limit: int
    resets_at: datetime


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DemoLimiter:
    """Demo and public-BYOK usage counters, shared by every request (and every worker process, through SQLite).

    path   a SQLite file, so counters survive restarts; None = in memory (tests, TRIAGE_LOG=memory).
    clock  returns the current time as an aware UTC datetime; days, the per-minute window and Retry-After all
           use it (tests pass a fake clock).

    Counted per UTC day ("YYYY-MM-DD" of clock()):
        per visitor and kind ("demo" / "byok"): requests
        demo-wide: requests, tokens (settled actuals plus open reservations), model calls
    The per-minute window holds the times of demo requests admitted in the last 60 s. It lives in memory in each
    process (not in SQLite), so with several workers each enforces it separately.

    Every check-and-count is atomic: with threads (a lock) and with several processes on one SQLite file (one
    BEGIN IMMEDIATE transaction per admission or settlement). A refused request changes nothing.
    Rows for days older than retention_days are deleted when the limiter is created, and by cleanup().
    """

    def __init__(self, settings: DemoSettings, path: Optional[Path] = None, clock: Callable[[], datetime] = utc_now):
        raise NotImplementedError

    def admit_demo(self, visitor: str, estimated_tokens: int) -> Admission:
        """Count one demo request for `visitor`, or raise DemoLimitError without counting anything.

        Checked in this order:
          1. demo_paused         demo-wide: requests + 1 > daily_requests, or tokens + estimated_tokens >
                                 daily_tokens (tokens = settled actuals plus open reservations), or
                                 calls >= daily_model_calls. retry_after: seconds to 00:00 UTC.
          2. demo_limit_reached  the visitor's demo requests today >= visitor_daily_requests.
                                 retry_after: seconds to 00:00 UTC.
          3. demo_busy           per_minute_requests demo requests were admitted in the last 60 s.
                                 retry_after: seconds until the oldest of them is 60 s old (at least 1).
        On success: the visitor's and the demo-wide request counts go up by 1, estimated_tokens is reserved
        against daily_tokens, the request joins the per-minute window, and an Admission is returned
        (remaining = visitor_daily_requests - the visitor's count, reserved_tokens = estimated_tokens).
        """
        raise NotImplementedError

    def admit_public_byok(self, visitor: str) -> Admission:
        """Count one public BYOK request (provider keys, no Triage key), or raise DemoLimitError
        ("public_limit_reached", retry_after to 00:00 UTC) without counting.

        Only the visitor's own daily count of this kind is checked (public_byok_visitor_daily_requests). It is
        separate from the demo count, and BYOK never touches the demo-wide budgets or the per-minute window.
        Returns an Admission with kind "byok" and reserved_tokens 0.
        """
        raise NotImplementedError

    def settle(self, admission: Admission, tokens: int, calls: int) -> None:
        """Record what an admitted request actually used, once, when it has finished (or failed).

        demo: the reservation is replaced by `tokens` (the day's tokens change by tokens - reserved_tokens), and
        `calls` (every model call it made: retries, fallbacks, truncated attempts) is added to the day's calls.
        Applied to admission.day, so a request that ends after midnight doesn't count against the new day.
        byok: nothing to record (a no-op).
        """
        raise NotImplementedError

    def status(self, visitor: str) -> DemoStatus:
        """Today's demo state for `visitor` (see DemoStatus). Counts nothing and needs no window slot."""
        raise NotImplementedError

    def requests_in_last_minute(self) -> int:
        """How many demo requests were admitted in the rolling 60 s before clock() (the per-minute window).
        Entries older than 60 s are dropped as a side effect."""
        raise NotImplementedError

    def cleanup(self) -> int:
        """Delete stored counters for days more than retention_days before today (UTC). Returns how many day
        records (visitor rows and demo-wide rows) were removed. Also run once by __init__."""
        raise NotImplementedError

    def stored_days(self) -> list:
        """The UTC days ("YYYY-MM-DD", oldest first) that have any stored counters, for cleanup checks."""
        raise NotImplementedError

    def close(self) -> None:
        """Close the SQLite connection (no-op in memory)."""
        raise NotImplementedError


def seconds_until_midnight(now: datetime) -> int:
    """Whole seconds from `now` (aware, UTC) to the next 00:00 UTC, at least 1 (for Retry-After)."""
    midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, int((midnight - now).total_seconds() + 0.999999))


def next_midnight(now: datetime) -> datetime:
    return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

