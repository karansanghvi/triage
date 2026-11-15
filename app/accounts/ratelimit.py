"""Sign-in rate limits in the database (auth_attempts), shared by every instance. Blocking: worker threads.

    signup_ip             5 signups per IP per hour
    login_ip              10 logins per IP per 15 minutes
    login_failure_email   5 failed logins per email per 15 minutes

Sliding windows: one row per counted attempt, counted over the last window. The subject is always a keyed hash:
an IP's visitor_id (app/api/demo.py), or email_subject() of a normalized email. Never an IP or an email.
Check-and-count runs alone (app.db.exclusive_transaction), so concurrent attempts can't overshoot a limit.
"""
import hashlib
import hmac
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable

from sqlalchemy import and_, delete, select

from app import db

AUTH_LOCK_KEY = 0x7472_6961_6765_0002          # pg_advisory_xact_lock key (see app/api/demo.py DEMO_LOCK_KEY)
KEEP = timedelta(days=1)                       # rows older than this are deleted as attempts come in


@dataclass(frozen=True)
class Limit:
    attempts: int
    window: timedelta


LIMITS = {
    "signup_ip": Limit(5, timedelta(hours=1)),
    "login_ip": Limit(10, timedelta(minutes=15)),
    "login_failure_email": Limit(5, timedelta(minutes=15)),
}


class RateLimited(Exception):
    def __init__(self, kind: str, retry_after: int):
        super().__init__(kind)
        self.kind, self.retry_after = kind, retry_after


def email_subject(email: str, secret: bytes) -> str:
    """The auth_attempts subject for a normalized email: an HMAC, so the table holds no email addresses."""
    return hmac.new(secret, b"login-email:" + email.encode("utf-8"), hashlib.sha256).hexdigest()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AuthRateLimiter:
    def __init__(self, engine, clock: Callable[[], datetime] = _utc_now):
        self.engine, self.clock = engine, clock
        self._table = db.auth_attempts_table

    def _window(self, connection, kind: str, subject: str, now: datetime) -> list:
        t, limit = self._table, LIMITS[kind]
        rows = connection.execute(select(t.c.created_at).where(and_(
            t.c.kind == kind, t.c.subject == subject, t.c.created_at > now - limit.window))
            .order_by(t.c.created_at)).scalars().all()
        return [db.aware(r) for r in rows]

    def _refuse_if_full(self, kind: str, recent: list, now: datetime) -> None:
        limit = LIMITS[kind]
        if len(recent) >= limit.attempts:
            oldest = recent[-limit.attempts]
            raise RateLimited(kind, max(1, math.ceil((oldest + limit.window - now).total_seconds())))

    def hit(self, kind: str, subject: str) -> None:
        """Count one attempt, or raise RateLimited (counting nothing) if the window is already full."""
        now = self.clock()
        with db.exclusive_transaction(self.engine, AUTH_LOCK_KEY) as connection:
            self._refuse_if_full(kind, self._window(connection, kind, subject, now), now)
            connection.execute(self._table.insert(), dict(kind=kind, subject=subject, created_at=now))
            connection.execute(delete(self._table).where(self._table.c.created_at < now - KEEP))

    def check(self, kind: str, subject: str) -> None:
        """Raise RateLimited if the window is full; count nothing (login: the per-email failure limit)."""
        now = self.clock()
        with self.engine.connect() as connection:
            recent = self._window(connection, kind, subject, now)
        self._refuse_if_full(kind, recent, now)

    def record(self, kind: str, subject: str) -> None:
        """Count one attempt without checking (login: a failure for that email)."""
        with self.engine.begin() as connection:
            connection.execute(self._table.insert(), dict(kind=kind, subject=subject, created_at=self.clock()))

    def count(self, kind: str, subject: str) -> int:
        now = self.clock()
        with self.engine.connect() as connection:
            return len(self._window(connection, kind, subject, now))
