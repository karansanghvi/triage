"""Account API keys in the database (api_keys). Blocking: call from a worker thread in async code.

A key ("trk_..."; app/accounts/credentials.py) is shown once, when it's created. Stored: its public id ("trk_" +
8 characters, shown in lists) and an HMAC-SHA256 of the full key with TRIAGE_HASH_SECRET. A request's key is
found by its public id and then compared with hmac.compare_digest. Revoking keeps the row (revoked_at).
"""
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from sqlalchemy import and_, func, select, update
from sqlalchemy.exc import IntegrityError

from app import db
from app.accounts.credentials import (api_key_hash, api_key_public_id, hashes_match, looks_like_api_key,
                                      new_api_key)

MAX_ACTIVE_KEYS = 20
NAME_MAX = 60
TOUCH_EVERY = timedelta(minutes=1)          # last_used_at is written at most this often per key


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Optional[datetime]) -> Optional[str]:
    return None if value is None else db.aware(value).isoformat().replace("+00:00", "Z")


class TooManyKeys(Exception):
    """The user already has MAX_ACTIVE_KEYS keys that aren't revoked."""


@dataclass(frozen=True)
class ApiKey:
    """A key as lists show it: never the secret."""
    id: str
    user_id: str
    name: str
    public_id: str
    created_at: datetime
    last_used_at: Optional[datetime]
    revoked_at: Optional[datetime]

    def public(self) -> dict:
        return {"id": self.id, "name": self.name, "public_id": self.public_id, "created_at": _iso(self.created_at),
                "last_used_at": _iso(self.last_used_at), "revoked_at": _iso(self.revoked_at)}


def _row(r) -> ApiKey:
    return ApiKey(r.id, r.user_id, r.name, r.public_id, db.aware(r.created_at), db.aware(r.last_used_at),
                  db.aware(r.revoked_at))


class ApiKeyStore:
    def __init__(self, engine, secret: bytes, clock: Callable[[], datetime] = _utc_now):
        self.engine, self._secret, self.clock = engine, secret, clock
        self._table = db.api_keys_table

    def create(self, user_id: str, name: str) -> tuple:
        """(the full key, ApiKey). The full key exists only in this return value. Raises TooManyKeys.
        The count and the insert run alone (app.db.exclusive_transaction), so the cap holds under concurrency."""
        t = self._table
        for _ in range(5):                       # a public id clash (8 base62 characters) is rare: try again
            key, now = new_api_key(), self.clock()
            row = ApiKey(str(uuid.uuid4()), user_id, name, api_key_public_id(key), now, None, None)
            try:
                with db.exclusive_transaction(self.engine, API_KEYS_LOCK_KEY) as connection:
                    active = connection.execute(select(func.count()).select_from(t).where(
                        and_(t.c.user_id == user_id, t.c.revoked_at.is_(None)))).scalar()
                    if active >= MAX_ACTIVE_KEYS:
                        raise TooManyKeys()
                    connection.execute(t.insert(), dict(id=row.id, user_id=user_id, name=name,
                                                        public_id=row.public_id,
                                                        secret_hash=api_key_hash(key, self._secret),
                                                        created_at=now, last_used_at=None, revoked_at=None))
                return key, row
            except IntegrityError:
                continue
        raise RuntimeError("couldn't create a unique API key id")

    def list(self, user_id: str) -> list:
        """Newest first, revoked ones included."""
        t = self._table
        with self.engine.connect() as connection:
            rows = connection.execute(select(t).where(t.c.user_id == user_id)
                                      .order_by(t.c.created_at.desc(), t.c.id)).fetchall()
        return [_row(r) for r in rows]

    def revoke(self, user_id: str, key_id: str) -> bool:
        """True if the key exists and belongs to the user (revoking again keeps the first revoked_at).
        False for an unknown id or another user's key: callers answer both the same way."""
        t = self._table
        with self.engine.begin() as connection:
            owned = connection.execute(select(t.c.id).where(and_(t.c.id == key_id, t.c.user_id == user_id))) \
                .fetchone()
            if owned is None:
                return False
            connection.execute(update(t).where(and_(t.c.id == key_id, t.c.revoked_at.is_(None)))
                               .values(revoked_at=self.clock()))
        return True

    def lookup(self, key: str) -> Optional[ApiKey]:
        """The active ApiKey for a full key, else None (malformed, unknown, wrong secret or revoked).
        Found by public id, then compared in constant time."""
        if not looks_like_api_key(key):
            return None
        t = self._table
        with self.engine.connect() as connection:
            r = connection.execute(select(t).where(t.c.public_id == api_key_public_id(key))).fetchone()
        if r is None or not hashes_match(r.secret_hash, api_key_hash(key, self._secret)) or r.revoked_at is not None:
            return None
        return _row(r)

    def touch(self, api_key: ApiKey) -> None:
        """Record a use: written only if the last one is more than a minute old."""
        now, t = self.clock(), self._table
        if api_key.last_used_at is not None and now - api_key.last_used_at < TOUCH_EVERY:
            return
        with self.engine.begin() as connection:
            connection.execute(update(t).where(and_(t.c.id == api_key.id, (t.c.last_used_at.is_(None))
                                                    | (t.c.last_used_at < now - TOUCH_EVERY)))
                               .values(last_used_at=now))


API_KEYS_LOCK_KEY = 0x7472_6961_6765_0003      # pg_advisory_xact_lock key (see app/api/demo.py DEMO_LOCK_KEY)
