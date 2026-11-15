"""Passwords, emails and session tokens: the pure parts of accounts (no database).

Passwords are hashed with argon2id at OWASP's parameters (19 MiB, t=2, p=1): the library default (64 MiB) could
exhaust a small instance during a burst of logins. Hashing is slow on purpose: call these from a worker thread.
A password is never stored, logged or put in an error message; only its hash is kept.

Session tokens are "trs_" + 32 random bytes in base62. Only their SHA-256 is stored; the token itself is shown
once, at signup or login.
"""
import hashlib
import hmac
import re
import secrets
import unicodedata
from typing import Optional

from argon2 import PasswordHasher, Type
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

PASSWORD_MIN, PASSWORD_MAX = 10, 128
EMAIL_MAX = 254
DISPLAY_NAME_MAX = 80
SESSION_PREFIX = "trs_"
TOKEN_BYTES = 32

_hasher = PasswordHasher(time_cost=2, memory_cost=19 * 1024, parallelism=1, hash_len=32, salt_len=16,
                         type=Type.ID)
# Verified against when the email is unknown, so a login takes as long whether or not the account exists
_DUMMY_HASH = _hasher.hash(secrets.token_urlsafe(16))


class CredentialError(ValueError):
    """A password, email or display name that doesn't meet the rules. The message never contains the value."""


# ---------- passwords ----------

def check_password_rules(password: str) -> None:
    if not isinstance(password, str) or not PASSWORD_MIN <= len(password) <= PASSWORD_MAX:
        raise CredentialError(f"Passwords must be {PASSWORD_MIN} to {PASSWORD_MAX} characters long.")


def hash_password(password: str) -> str:
    check_password_rules(password)
    return _hasher.hash(password)


def verify_password(password_hash: Optional[str], password: str) -> bool:
    """True if `password` matches. password_hash None (no such user): checks against a dummy hash and returns
    False, taking the same time as a real check."""
    try:
        return _hasher.verify(password_hash or _DUMMY_HASH, password) and password_hash is not None
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def needs_rehash(password_hash: str) -> bool:
    """True if the hash was made with other parameters (rehash it at the next successful login)."""
    return _hasher.check_needs_rehash(password_hash)


# ---------- emails and names ----------

# Deliberately loose: one @, something before it, a dot in the domain, no spaces. Real checking would need a
# verification email, which is out of scope.
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s.]+$")


def normalize_email(email: str) -> str:
    """Trimmed, NFKC, lowercase; dots and +tags are kept (they may be different mailboxes)."""
    if not isinstance(email, str):
        raise CredentialError("Enter a valid email address.")
    normalized = unicodedata.normalize("NFKC", email).strip().lower()
    if len(normalized) > EMAIL_MAX or not _EMAIL.match(normalized):
        raise CredentialError("Enter a valid email address.")
    return normalized


def normalize_display_name(name: Optional[str]) -> Optional[str]:
    """Trimmed; None or empty = no name; otherwise 1 to 80 characters."""
    if name is None:
        return None
    name = unicodedata.normalize("NFKC", name).strip()
    if not name:
        return None
    if len(name) > DISPLAY_NAME_MAX:
        raise CredentialError(f"Display names can be at most {DISPLAY_NAME_MAX} characters.")
    return name


# ---------- session tokens ----------

_BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_TOKEN_CHARS = 43                       # 62**43 > 2**256: every 32-byte value fits


def _base62(data: bytes) -> str:
    n, out = int.from_bytes(data, "big"), []
    for _ in range(_TOKEN_CHARS):
        n, r = divmod(n, 62)
        out.append(_BASE62[r])
    return "".join(reversed(out))


def new_session_token() -> str:
    return SESSION_PREFIX + _base62(secrets.token_bytes(TOKEN_BYTES))


def token_hash(token: str) -> str:
    """The stored form of a token: SHA-256, hex. A token has 256 random bits, so a fast hash is safe."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def hashes_match(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("ascii"), b.encode("ascii"))


def looks_like_session_token(token: str) -> bool:
    return (token.startswith(SESSION_PREFIX) and len(token) == len(SESSION_PREFIX) + _TOKEN_CHARS
            and all(c in _BASE62 for c in token[len(SESSION_PREFIX):]))
