"""Users and sessions in the database (app/db.py). Blocking: call from a worker thread in async code.

Sessions are bearer tokens ("trs_..."; app/accounts/credentials.py). Only their SHA-256 is stored. A session is
valid until the first of: logout or a revocation (revoked_at), its absolute expiry (expires_at, 30 days after
login), or 7 days without use (last_used_at, updated at most once a minute so each request isn't a write).
"""
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from sqlalchemy import and_, delete, select, update
from sqlalchemy.exc import IntegrityError

from app import db
from app.accounts.credentials import hashes_match, token_hash, new_session_token

SESSION_IDLE = timedelta(days=7)
SESSION_ABSOLUTE = timedelta(days=30)
TOUCH_EVERY = timedelta(minutes=1)
PROVIDERS = ("openai", "anthropic", "groq", "gemini")      # what preferred_provider may be (or None)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class EmailTaken(Exception):
    """Signup with an email that already has an account."""


@dataclass(frozen=True)
class User:
    id: str
    email: str
    display_name: Optional[str]
    preferred_provider: Optional[str]
    created_at: datetime
    password_hash: str = field(repr=False)

    def public(self) -> dict:
        """What GET /v1/auth/me returns: never the password hash."""
        return {"id": self.id, "email": self.email, "display_name": self.display_name,
                "preferred_provider": self.preferred_provider,
                "created_at": self.created_at.isoformat().replace("+00:00", "Z")}


@dataclass(frozen=True)
class Session:
    id: str
    user_id: str
    created_at: datetime
    last_used_at: datetime
    expires_at: datetime                     # absolute expiry


def _user(row) -> User:
    return User(row.id, row.email, row.display_name, row.preferred_provider, db.aware(row.created_at),
                row.password_hash)


class AccountStore:
    def __init__(self, engine, clock: Callable[[], datetime] = utc_now):
        self.engine, self.clock = engine, clock
        self._users, self._sessions = db.users_table, db.sessions_table

    # ----- users -----

    def create_user(self, email: str, password_hash: str, display_name: Optional[str]) -> User:
        """email: already normalized. Raises EmailTaken (the unique index decides, so two signups at once with
        one email can't both succeed)."""
        now = self.clock()
        values = dict(id=str(uuid.uuid4()), email=email, password_hash=password_hash, display_name=display_name,
                      preferred_provider=None, created_at=now, updated_at=now)
        try:
            with self.engine.begin() as connection:
                connection.execute(self._users.insert(), values)
        except IntegrityError:
            raise EmailTaken() from None
        return User(values["id"], email, display_name, None, now, password_hash)

    def user_by_email(self, email: str) -> Optional[User]:
        with self.engine.connect() as connection:
            row = connection.execute(select(self._users).where(self._users.c.email == email)).fetchone()
        return _user(row) if row else None

    def user_by_id(self, user_id: str) -> Optional[User]:
        with self.engine.connect() as connection:
            row = connection.execute(select(self._users).where(self._users.c.id == user_id)).fetchone()
        return _user(row) if row else None

    def update_user(self, user_id: str, **values) -> Optional[User]:
        """Set display_name, preferred_provider and/or password_hash; updated_at follows."""
        allowed = {"display_name", "preferred_provider", "password_hash"}
        if not set(values) <= allowed:
            raise ValueError(f"can't update {sorted(set(values) - allowed)}")
        with self.engine.begin() as connection:
            connection.execute(update(self._users).where(self._users.c.id == user_id)
                               .values(**values, updated_at=self.clock()))
        return self.user_by_id(user_id)

    def delete_user(self, user_id: str) -> None:
        """The user and every session, in one transaction. Their request rows stay, with user_id set to NULL
        (the foreign key's ON DELETE SET NULL)."""
        with self.engine.begin() as connection:
            connection.execute(delete(self._sessions).where(self._sessions.c.user_id == user_id))
            connection.execute(delete(self._users).where(self._users.c.id == user_id))

    # ----- sessions -----

    def create_session(self, user_id: str) -> tuple:
        """(token, Session). The token is returned only here; only its hash is stored."""
        now, token = self.clock(), new_session_token()
        session = Session(str(uuid.uuid4()), user_id, now, now, now + SESSION_ABSOLUTE)
        with self.engine.begin() as connection:
            connection.execute(self._sessions.insert(), dict(
                id=session.id, user_id=user_id, token_hash=token_hash(token), created_at=now, last_used_at=now,
                expires_at=session.expires_at, revoked_at=None))
        return token, session

    def session_for_token(self, token: str) -> Optional[tuple]:
        """(Session, User) for a valid token, else None (unknown, revoked, expired or idle too long).
        Refreshes last_used_at when it is more than a minute old."""
        wanted = token_hash(token)
        s, u = self._sessions, self._users
        with self.engine.connect() as connection:
            row = connection.execute(
                select(s.c.id, s.c.user_id, s.c.token_hash, s.c.created_at, s.c.last_used_at, s.c.expires_at,
                       s.c.revoked_at, u.c.id.label("uid"), u.c.email, u.c.display_name, u.c.preferred_provider,
                       u.c.created_at.label("user_created_at"), u.c.password_hash)
                .join(u, u.c.id == s.c.user_id).where(s.c.token_hash == wanted)).fetchone()
        if row is None or not hashes_match(row.token_hash, wanted):
            return None
        now = self.clock()
        last_used, expires = db.aware(row.last_used_at), db.aware(row.expires_at)
        if row.revoked_at is not None or now >= expires or now - last_used >= SESSION_IDLE:
            return None
        if now - last_used >= TOUCH_EVERY:
            with self.engine.begin() as connection:
                connection.execute(update(s).where(s.c.id == row.id).values(last_used_at=now))
            last_used = now
        session = Session(row.id, row.user_id, db.aware(row.created_at), last_used, expires)
        user = User(row.uid, row.email, row.display_name, row.preferred_provider, db.aware(row.user_created_at),
                    row.password_hash)
        return session, user

    def revoke_session(self, session_id: str) -> None:
        with self.engine.begin() as connection:
            connection.execute(update(self._sessions).where(and_(self._sessions.c.id == session_id,
                                                                 self._sessions.c.revoked_at.is_(None)))
                               .values(revoked_at=self.clock()))

    def revoke_other_sessions(self, user_id: str, keep_session_id: str) -> int:
        """Revoke every session of the user except one (after a password change). Returns how many."""
        s = self._sessions
        with self.engine.begin() as connection:
            return connection.execute(update(s).where(and_(s.c.user_id == user_id, s.c.id != keep_session_id,
                                                           s.c.revoked_at.is_(None)))
                                      .values(revoked_at=self.clock())).rowcount
