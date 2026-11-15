"""Saved provider keys in the database (provider_keys). Blocking: call from a worker thread in async code.

A key is encrypted before it reaches the database and decrypted only by decrypt(), on demand: nothing here keeps
a decrypted key. Rows hold the ciphertext, nonce, encryption key id, last 4 characters, and the status of the
last check (valid / invalid / unverified).
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

from sqlalchemy import and_, delete, select, update

from app import db
from app.accounts.encryption import KeyRing, Sealed

PROVIDERS = ("openai", "anthropic", "groq", "gemini")
STATUSES = ("valid", "invalid", "unverified")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Optional[datetime]) -> Optional[str]:
    return None if value is None else db.aware(value).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class SavedKey:
    """A saved key as the API shows it: never the key."""
    provider: str
    last4: str
    status: str
    status_detail: Optional[str]
    verified_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime

    def public(self) -> dict:
        return {"provider": self.provider, "last4": self.last4, "status": self.status,
                "verified_at": _iso(self.verified_at), "created_at": _iso(self.created_at),
                "updated_at": _iso(self.updated_at)}


class ProviderKeyStore:
    def __init__(self, engine, keyring: Optional[KeyRing], clock: Callable[[], datetime] = _utc_now):
        self.engine, self.keyring, self.clock = engine, keyring, clock
        self._table = db.provider_keys_table

    def _ring(self) -> KeyRing:
        if self.keyring is None:
            raise RuntimeError("no encryption keys configured (TRIAGE_ENCRYPTION_KEYS)")
        return self.keyring

    @staticmethod
    def _summary(row) -> SavedKey:
        return SavedKey(row.provider, row.last4, row.status, row.status_detail, db.aware(row.verified_at),
                        db.aware(row.created_at), db.aware(row.updated_at))

    def save(self, user_id: str, provider: str, key: str, status: str, status_detail: Optional[str],
             verified: bool) -> SavedKey:
        """Encrypt and store `key`, replacing any saved key for this provider (a new key: new created_at).
        verified: the check got an answer from the provider (verified_at = now)."""
        if provider not in PROVIDERS or status not in STATUSES:
            raise ValueError("unknown provider or status")
        sealed = self._ring().encrypt(key, user_id, provider)
        now, t = self.clock(), self._table
        values = dict(user_id=user_id, provider=provider, ciphertext=sealed.ciphertext, nonce=sealed.nonce,
                      enc_key_id=sealed.key_id, last4=key[-4:], status=status, status_detail=status_detail,
                      verified_at=now if verified else None, created_at=now, updated_at=now)
        with self.engine.begin() as connection:
            connection.execute(delete(t).where(and_(t.c.user_id == user_id, t.c.provider == provider)))
            connection.execute(t.insert(), values)
        return SavedKey(provider, values["last4"], status, status_detail, values["verified_at"], now, now)

    def list(self, user_id: str) -> list:
        t = self._table
        with self.engine.connect() as connection:
            rows = connection.execute(select(t).where(t.c.user_id == user_id)).fetchall()
        order = {p: i for i, p in enumerate(PROVIDERS)}
        return sorted((self._summary(r) for r in rows), key=lambda k: order.get(k.provider, 99))

    def get(self, user_id: str, provider: str) -> Optional[SavedKey]:
        t = self._table
        with self.engine.connect() as connection:
            row = connection.execute(select(t).where(and_(t.c.user_id == user_id, t.c.provider == provider))) \
                .fetchone()
        return self._summary(row) if row else None

    def decrypt(self, user_id: str, provider: str) -> Optional[str]:
        """The saved key in plain text, or None if there is none. Use it and drop it: never store the result."""
        t = self._table
        with self.engine.connect() as connection:
            row = connection.execute(select(t.c.ciphertext, t.c.nonce, t.c.enc_key_id).where(
                and_(t.c.user_id == user_id, t.c.provider == provider))).fetchone()
        if row is None:
            return None
        return self._ring().decrypt(Sealed(bytes(row.ciphertext), bytes(row.nonce), row.enc_key_id),
                                    user_id, provider)

    def set_status(self, user_id: str, provider: str, status: str, status_detail: Optional[str],
                   verified: bool) -> Optional[SavedKey]:
        if status not in STATUSES:
            raise ValueError("unknown status")
        now, t = self.clock(), self._table
        values = dict(status=status, status_detail=status_detail, updated_at=now)
        if verified:
            values["verified_at"] = now
        with self.engine.begin() as connection:
            connection.execute(update(t).where(and_(t.c.user_id == user_id, t.c.provider == provider))
                               .values(**values))
        return self.get(user_id, provider)

    def delete(self, user_id: str, provider: str) -> bool:
        t = self._table
        with self.engine.begin() as connection:
            return connection.execute(delete(t).where(and_(t.c.user_id == user_id, t.c.provider == provider))) \
                .rowcount > 0
