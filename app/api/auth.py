"""Bearer-key auth. Keys come from TRIAGE_API_KEYS / TRIAGE_ADMIN_KEYS (comma-separated).

Keys are never logged, echoed or put in an error message; logs only ever see key_id().
"""
import hashlib
import hmac
import os
from dataclasses import dataclass
from typing import Optional

from app.api.errors import APIError

API_KEYS_ENV, ADMIN_KEYS_ENV = "TRIAGE_API_KEYS", "TRIAGE_ADMIN_KEYS"


def key_id(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]


def keys_from_env(name: str) -> tuple:
    return tuple(k.strip() for k in os.environ.get(name, "").split(",") if k.strip())


@dataclass(frozen=True)
class Caller:
    key_id: str
    admin: bool


class Auth:
    def __init__(self, api_keys, admin_keys=()):
        self._api = tuple(api_keys)
        self._admin = tuple(admin_keys)
        if not self._api and not self._admin:
            raise RuntimeError(f"no API keys configured: set {API_KEYS_ENV} (and {ADMIN_KEYS_ENV} for admin calls)")

    @staticmethod
    def _matches(key: str, keys: tuple) -> bool:
        found = False
        for k in keys:                                  # compare against every key: constant time overall
            found |= hmac.compare_digest(key.encode("utf-8"), k.encode("utf-8"))
        return found

    def check(self, authorization: Optional[str], admin: bool = False) -> Caller:
        if not authorization or not authorization.lower().startswith("bearer ") or not authorization[7:].strip():
            raise APIError(401, "Missing API key. Send it as: Authorization: Bearer <key>.", code="invalid_api_key")
        key = authorization[7:].strip()
        is_admin = self._matches(key, self._admin)
        if not is_admin and not self._matches(key, self._api):
            raise APIError(401, "Incorrect API key provided.", code="invalid_api_key")
        if admin and not is_admin:
            raise APIError(403, "This endpoint needs an admin key.", "permission_error", code="admin_key_required")
        return Caller(key_id(key), is_admin)
