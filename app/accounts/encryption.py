"""Encryption of saved provider keys at rest (Phase 5, accounts).

AES-256-GCM (cryptography), a fresh random 96-bit nonce for every encryption, and associated data
"<user_id>:<provider>": the ciphertext only decrypts for the row it was written for, so a ciphertext copied to
another user or provider fails to decrypt instead of handing over someone else's key.

The keys come from TRIAGE_ENCRYPTION_KEYS, never from the database or the repo:

    TRIAGE_ENCRYPTION_KEYS="k2:<base64 of 32 bytes>,k1:<base64 of 32 bytes>"

The first key encrypts; every listed key decrypts (by the key id stored with each row). To rotate: put a new
key first, deploy, re-encrypt the saved keys, then drop the old one. Generate a key with:

    python3 -c "import base64, secrets; print(base64.b64encode(secrets.token_bytes(32)).decode())"

No error message or repr ever contains a key's value.
"""
import base64
import binascii
import os
import re
import secrets
from dataclasses import dataclass
from typing import Optional

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

ENCRYPTION_KEYS_ENV = "TRIAGE_ENCRYPTION_KEYS"
KEY_BYTES, NONCE_BYTES = 32, 12
_KEY_ID = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
GENERATE_COMMAND = ('python3 -c "import base64, secrets; '
                    'print(base64.b64encode(secrets.token_bytes(32)).decode())"')


class EncryptionConfigError(RuntimeError):
    """TRIAGE_ENCRYPTION_KEYS is missing or malformed. The message names the problem, never a value."""


class DecryptionError(Exception):
    """A saved key that can't be decrypted: its key id isn't configured, or the data or associated data is wrong."""


@dataclass(frozen=True, repr=False)
class Sealed:
    """What is stored for one saved key."""
    ciphertext: bytes          # includes GCM's 16-byte tag
    nonce: bytes
    key_id: str

    def __repr__(self) -> str:
        return f"Sealed(key_id={self.key_id!r}, {len(self.ciphertext)} bytes)"


def _decode_key(key_id: str, text: str) -> bytes:
    try:
        raw = base64.b64decode(text.strip(), validate=True)
    except (binascii.Error, ValueError):
        try:
            raw = base64.urlsafe_b64decode(text.strip())
        except (binascii.Error, ValueError):
            raise EncryptionConfigError(f"{ENCRYPTION_KEYS_ENV}: key {key_id!r} isn't valid base64") from None
    if len(raw) != KEY_BYTES:
        raise EncryptionConfigError(f"{ENCRYPTION_KEYS_ENV}: key {key_id!r} must be exactly {KEY_BYTES} bytes "
                                    f"after base64 decoding (it is {len(raw)}). Generate one with: "
                                    f"{GENERATE_COMMAND}")
    return raw


def parse_keys(value: Optional[str]) -> list:
    """[(key_id, 32 bytes)] from "id:base64,id:base64", first = the one that encrypts."""
    if not value or not value.strip():
        raise EncryptionConfigError(f"{ENCRYPTION_KEYS_ENV} is not set. Accounts need it to encrypt saved "
                                    f'provider keys: {ENCRYPTION_KEYS_ENV}="k1:<key>", where <key> comes from: '
                                    f"{GENERATE_COMMAND}")
    keys, seen = [], set()
    for position, entry in enumerate(e.strip() for e in value.split(",") if e.strip()):
        key_id, sep, text = entry.partition(":")
        key_id = key_id.strip()
        if not sep or not _KEY_ID.match(key_id):
            raise EncryptionConfigError(f"{ENCRYPTION_KEYS_ENV}: entry {position + 1} must look like "
                                        "id:<base64>, with an id of letters, digits, '-' or '_'")
        if key_id in seen:
            raise EncryptionConfigError(f"{ENCRYPTION_KEYS_ENV}: key id {key_id!r} is listed twice")
        seen.add(key_id)
        keys.append((key_id, _decode_key(key_id, text)))
    if not keys:
        raise EncryptionConfigError(f"{ENCRYPTION_KEYS_ENV} lists no keys")
    return keys


def _associated_data(user_id: str, provider: str) -> bytes:
    return f"{user_id}:{provider}".encode("utf-8")


class KeyRing:
    """The configured encryption keys. Holds no user data; never prints a key."""

    def __init__(self, keys: list):
        if not keys:
            raise EncryptionConfigError("no encryption keys")
        self._keys = {key_id: AESGCM(raw) for key_id, raw in keys}
        self.primary_id = keys[0][0]

    @classmethod
    def from_env(cls, value: Optional[str] = None) -> "KeyRing":
        return cls(parse_keys(os.environ.get(ENCRYPTION_KEYS_ENV) if value is None else value))

    @property
    def key_ids(self) -> tuple:
        return tuple(self._keys)

    def encrypt(self, plaintext: str, user_id: str, provider: str) -> Sealed:
        """With the first (primary) key and a fresh random nonce."""
        nonce = secrets.token_bytes(NONCE_BYTES)
        ciphertext = self._keys[self.primary_id].encrypt(nonce, plaintext.encode("utf-8"),
                                                          _associated_data(user_id, provider))
        return Sealed(ciphertext, nonce, self.primary_id)

    def decrypt(self, sealed: Sealed, user_id: str, provider: str) -> str:
        cipher = self._keys.get(sealed.key_id)
        if cipher is None:
            raise DecryptionError(f"encryption key {sealed.key_id!r} isn't in {ENCRYPTION_KEYS_ENV}")
        try:
            return cipher.decrypt(sealed.nonce, sealed.ciphertext, _associated_data(user_id, provider)).decode("utf-8")
        except InvalidTag:
            raise DecryptionError("a saved key failed to decrypt (wrong row or damaged data)") from None

    def __repr__(self) -> str:
        return f"KeyRing(primary={self.primary_id!r}, keys={list(self._keys)})"
