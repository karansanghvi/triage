"""Password reset codes and tokens in the database. Blocking: call from a worker thread in async code.

    issue_code(user_id)            -> (code, code id), or None inside the 60-second resend cooldown
    verify_code(user_id, code)     -> (reset token, expires_at), or None (wrong, expired, used or dead)
    complete(token, password_hash) -> the user id, or None (unknown, expired or used token)

A code is 6 digits, lives 10 minutes and allows 5 tries. Each try is counted BEFORE the code is compared, with a
conditional UPDATE, so parallel guesses can't get more than 5 comparisons. A new code replaces any earlier one; a
request inside the cooldown changes nothing (the code already emailed stays valid).

A reset token ("rst_...") lives 15 minutes and is used once. Verifying a code cancels any earlier token.
complete() uses the token, sets the password, revokes every session and ends any outstanding code, in one
transaction. Only hashes are stored: an HMAC of the code (with the user id), the SHA-256 of the token.
"""
import uuid
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from sqlalchemy import and_, delete, select, update
from sqlalchemy.exc import IntegrityError

from app import db
from app.accounts.credentials import (hashes_match, new_reset_code, new_reset_token, normalize_reset_code,
                                      reset_code_hash, token_hash)

CODE_TTL = timedelta(minutes=10)
TOKEN_TTL = timedelta(minutes=15)
MAX_ATTEMPTS = 5
COOLDOWN = timedelta(seconds=60)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ResetStore:
    def __init__(self, engine, secret: bytes, clock: Callable[[], datetime] = _utc_now):
        self.engine, self._secret, self.clock = engine, secret, clock
        self._codes, self._tokens = db.password_reset_codes_table, db.password_reset_tokens_table

    def issue_code(self, user_id: str) -> Optional[tuple]:
        """(code, code id) for a new code, replacing any earlier one. None inside the cooldown (nothing changes).
        The code exists only in this return value."""
        now, c = self.clock(), self._codes
        code, code_id = new_reset_code(), str(uuid.uuid4())
        values = dict(id=code_id, code_hash=reset_code_hash(code, user_id, self._secret), created_at=now,
                      expires_at=now + CODE_TTL, attempts=0, used_at=None, last_sent_at=now)
        with self.engine.begin() as connection:           # replace, only if the last email is old enough
            replaced = connection.execute(update(c).where(and_(c.c.user_id == user_id,
                                                               c.c.last_sent_at <= now - COOLDOWN))
                                          .values(**values)).rowcount
            if replaced:
                return code, code_id
            if connection.execute(select(c.c.user_id).where(c.c.user_id == user_id)).fetchone() is not None:
                return None                                  # inside the cooldown
        try:
            with self.engine.begin() as connection:
                connection.execute(c.insert(), dict(user_id=user_id, **values))
        except IntegrityError:                               # another request inserted one just now
            return None
        return code, code_id

    def verify_code(self, user_id: Optional[str], code) -> Optional[tuple]:
        """(reset token, expires_at) if `code` is this user's live code; else None. user_id None (no account):
        the same work, and None."""
        entered, now, c = normalize_reset_code(code), self.clock(), self._codes
        # Computed whatever happens, so an unknown email takes as long as a known one
        entered_hash = reset_code_hash(entered or "", user_id or "", self._secret)
        if user_id is None or entered is None:
            return None
        with self.engine.begin() as connection:
            live = and_(c.c.user_id == user_id, c.c.used_at.is_(None), c.c.attempts < MAX_ATTEMPTS,
                        c.c.expires_at > now)
            # Count the try first: only MAX_ATTEMPTS comparisons can ever happen for one code
            if not connection.execute(update(c).where(live).values(attempts=c.c.attempts + 1)).rowcount:
                return None
            row = connection.execute(select(c.c.id, c.c.code_hash).where(c.c.user_id == user_id)).fetchone()
            if not hashes_match(row.code_hash, entered_hash):
                return None
            if not connection.execute(update(c).where(and_(c.c.id == row.id, c.c.used_at.is_(None)))
                                      .values(used_at=now)).rowcount:
                return None
            token, t = new_reset_token(), self._tokens
            connection.execute(delete(t).where(t.c.user_id == user_id))       # one live token per user
            connection.execute(t.insert(), dict(token_hash=token_hash(token), user_id=user_id, created_at=now,
                                                expires_at=now + TOKEN_TTL, used_at=None))
        return token, now + TOKEN_TTL

    def complete(self, token: str, password_hash: str) -> Optional[str]:
        """Use the token (once), set the password, revoke every session and end any outstanding code, in one
        transaction. Returns the user id, or None if the token is unknown, expired or already used."""
        now, t, c = self.clock(), self._tokens, self._codes
        u, s = db.users_table, db.sessions_table
        wanted = token_hash(token)
        with self.engine.begin() as connection:
            row = connection.execute(select(t.c.user_id, t.c.token_hash).where(t.c.token_hash == wanted)).fetchone()
            if row is None or not hashes_match(row.token_hash, wanted):
                return None
            used = connection.execute(update(t).where(and_(t.c.token_hash == wanted, t.c.used_at.is_(None),
                                                           t.c.expires_at > now)).values(used_at=now)).rowcount
            if not used:
                return None
            user_id = row.user_id
            connection.execute(update(u).where(u.c.id == user_id).values(password_hash=password_hash, updated_at=now))
            connection.execute(update(s).where(and_(s.c.user_id == user_id, s.c.revoked_at.is_(None)))
                               .values(revoked_at=now))
            connection.execute(update(c).where(and_(c.c.user_id == user_id, c.c.used_at.is_(None)))
                               .values(used_at=now))
        return user_id
