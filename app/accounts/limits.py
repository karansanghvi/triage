"""Per-account chat limits (config/accounts.yaml). Blocking: call from a worker thread in async code.

    requests_per_minute   rolling 60 s, in memory per server instance (like the demo's per-minute window)
    requests_per_day      per UTC day, in the database (account_day), shared by every instance

admit() counts one request or raises AccountLimitError (counting nothing). The daily check-and-count runs alone
(app.db.exclusive_transaction), so concurrent requests can't overshoot it. Accounts get no demo or public-BYOK
limits: they use their own provider keys, so these only protect the server.
"""
import math
import threading
from collections import deque
from dataclasses import dataclass, fields
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from sqlalchemy import delete, select

from app import db
from app.registry import PROJECT_ROOT

ACCOUNT_LIMITS_PATH = PROJECT_ROOT / "config" / "accounts.yaml"
ACCOUNT_LOCK_KEY = 0x7472_6961_6765_0004       # pg_advisory_xact_lock key (see app/api/demo.py DEMO_LOCK_KEY)
WINDOW_SECONDS = 60


@dataclass(frozen=True)
class AccountLimits:
    requests_per_minute: int = 60
    requests_per_day: int = 10_000
    retention_days: int = 7


def load_account_limits(path=ACCOUNT_LIMITS_PATH) -> AccountLimits:
    """Every field required, each a whole number above 0, so a typo can't silently turn a limit off."""
    import yaml
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{Path(path).name} must be a mapping")
    names = [f.name for f in fields(AccountLimits)]
    unknown, missing = set(data) - set(names), [n for n in names if n not in data]
    if unknown:
        raise ValueError(f"{Path(path).name}: unknown setting(s): {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"{Path(path).name}: missing setting(s): {', '.join(missing)}")
    for name in names:
        value = data[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{Path(path).name}: '{name}' must be a whole number above 0, got {value!r}")
    return AccountLimits(**{n: data[n] for n in names})


class AccountLimitError(Exception):
    def __init__(self, which: str, retry_after: int):
        super().__init__(which)
        self.which, self.retry_after = which, retry_after          # which: "minute" | "day"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _seconds_until_midnight(now: datetime) -> int:
    midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, math.ceil((midnight - now).total_seconds()))


class AccountLimiter:
    def __init__(self, limits: AccountLimits, engine, clock: Callable[[], datetime] = _utc_now):
        self.limits, self.engine, self.clock = limits, engine, clock
        self._table = db.account_day_table
        self._lock = threading.Lock()
        self._windows: dict = {}                 # user_id -> deque of admission times (this instance only)
        self.cleanup()

    def _prune(self, window: deque, now: datetime) -> None:
        while window and (now - window[0]).total_seconds() >= WINDOW_SECONDS:
            window.popleft()

    def admit(self, user_id: str) -> int:
        """Count one chat request for the user; returns their requests left today. Raises AccountLimitError."""
        now, t = self.clock(), self._table
        day = now.strftime("%Y-%m-%d")
        with self._lock:
            window = self._windows.setdefault(user_id, deque())
            self._prune(window, now)
            if len(window) >= self.limits.requests_per_minute:
                oldest = (now - window[0]).total_seconds()
                raise AccountLimitError("minute", max(1, math.ceil(WINDOW_SECONDS - oldest)))
            with db.exclusive_transaction(self.engine, ACCOUNT_LOCK_KEY) as connection:
                used = connection.execute(select(t.c.requests).where(t.c.day == day, t.c.user_id == user_id)) \
                    .scalar() or 0
                if used >= self.limits.requests_per_day:
                    raise AccountLimitError("day", _seconds_until_midnight(now))
                statement = db.upsert(self.engine, t).values(day=day, user_id=user_id, requests=1)
                connection.execute(statement.on_conflict_do_update(index_elements=[t.c.day, t.c.user_id],
                                                                   set_={"requests": t.c.requests + 1}))
            window.append(now)                   # only once the day's count is committed
            if len(self._windows) > 10_000:      # forget users idle for a minute, so memory stays small
                for w in self._windows.values():
                    self._prune(w, now)
                self._windows = {u: w for u, w in self._windows.items() if w}
        return self.limits.requests_per_day - (used + 1)

    def used_today(self, user_id: str) -> int:
        t, day = self._table, self.clock().strftime("%Y-%m-%d")
        with self.engine.connect() as connection:
            return connection.execute(select(t.c.requests).where(t.c.day == day, t.c.user_id == user_id)) \
                .scalar() or 0

    def cleanup(self) -> int:
        cutoff = (self.clock() - timedelta(days=self.limits.retention_days)).strftime("%Y-%m-%d")
        with self.engine.begin() as connection:
            return connection.execute(delete(self._table).where(self._table.c.day < cutoff)).rowcount
