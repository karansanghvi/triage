"""Encryption of saved provider keys (app/accounts/encryption.py). No network, no database."""
import base64
import secrets

import pytest

from app.accounts.encryption import (ENCRYPTION_KEYS_ENV, DecryptionError, EncryptionConfigError, KeyRing, Sealed,
                                     parse_keys)

SECRET = "sk-test-SECRET123"


def b64(raw):
    return base64.b64encode(raw).decode()


K1, K2 = secrets.token_bytes(32), secrets.token_bytes(32)
RING = KeyRing([("k1", K1)])


def test_round_trip_with_a_fresh_nonce_each_time():
    a, b = RING.encrypt(SECRET, "u1", "openai"), RING.encrypt(SECRET, "u1", "openai")
    assert a.nonce != b.nonce and a.ciphertext != b.ciphertext and len(a.nonce) == 12
    assert SECRET.encode() not in a.ciphertext and a.key_id == "k1"
    assert RING.decrypt(a, "u1", "openai") == RING.decrypt(b, "u1", "openai") == SECRET


@pytest.mark.parametrize("user, provider", [("u2", "openai"), ("u1", "anthropic"), ("u1:openai", ""),
                                            ("u", "1:openai")])
def test_a_ciphertext_only_decrypts_for_its_own_row(user, provider):
    sealed = RING.encrypt(SECRET, "u1", "openai")
    with pytest.raises(DecryptionError):
        RING.decrypt(sealed, user, provider)


def test_damaged_data_fails_to_decrypt():
    sealed = RING.encrypt(SECRET, "u1", "openai")
    flipped = bytes([sealed.ciphertext[0] ^ 1]) + sealed.ciphertext[1:]
    with pytest.raises(DecryptionError):
        RING.decrypt(Sealed(flipped, sealed.nonce, "k1"), "u1", "openai")


def test_rotation_new_saves_use_the_first_key_and_old_ones_still_decrypt():
    old = KeyRing([("k1", K1)]).encrypt(SECRET, "u1", "groq")
    rotated = KeyRing.from_env(f"k2:{b64(K2)},k1:{b64(K1)}")
    assert rotated.primary_id == "k2" and rotated.key_ids == ("k2", "k1")
    assert rotated.decrypt(old, "u1", "groq") == SECRET
    new = rotated.encrypt(SECRET, "u1", "groq")
    assert new.key_id == "k2" and rotated.decrypt(new, "u1", "groq") == SECRET
    with pytest.raises(DecryptionError, match="'k2' isn't in"):
        KeyRing([("k1", K1)]).decrypt(new, "u1", "groq")          # the old key alone can't read new rows


def test_the_same_bytes_under_another_id_still_decrypt_nothing_else():
    sealed = RING.encrypt(SECRET, "u1", "openai")
    with pytest.raises(DecryptionError):
        KeyRing([("k1", K2)]).decrypt(sealed, "u1", "openai")   # right id, wrong key material


def test_urlsafe_base64_is_accepted():
    raw = b"\xfb" * 32                                              # encodes with - and _ in urlsafe form
    assert KeyRing.from_env(f"k1:{base64.urlsafe_b64encode(raw).decode()}").primary_id == "k1"


@pytest.mark.parametrize("value, message", [
    (None, "is not set"), ("", "is not set"), ("  ", "is not set"), (",", "lists no keys"),
    (b64(K1), "must look like id:<base64>"),                         # no id
    (f"k 1:{b64(K1)}", "must look like id:<base64>"),
    (f"k1:{b64(secrets.token_bytes(16))}", "exactly 32 bytes after base64 decoding \\(it is 16\\)"),
    (f"k1:{b64(secrets.token_bytes(33))}", "exactly 32 bytes"),
    ("k1:not base64 at all!!", "isn't valid base64"),
    (f"k1:{b64(K1)},k1:{b64(K2)}", "listed twice"),
])
def test_bad_configuration_is_refused_without_printing_any_key(value, message):
    with pytest.raises(EncryptionConfigError, match=message) as error:
        parse_keys(value)
    for raw in (K1, K2):
        assert b64(raw) not in str(error.value)
    if value:
        for part in value.replace(",", ":").split(":"):
            if len(part) > 8:
                assert part not in str(error.value)


def test_from_env_reads_the_variable(monkeypatch):
    monkeypatch.setenv(ENCRYPTION_KEYS_ENV, f"main:{b64(K1)}")
    assert KeyRing.from_env().primary_id == "main"
    monkeypatch.delenv(ENCRYPTION_KEYS_ENV)
    with pytest.raises(EncryptionConfigError):
        KeyRing.from_env()


def test_reprs_hide_the_keys_and_the_ciphertext():
    ring = KeyRing.from_env(f"k1:{b64(K1)}")
    sealed = ring.encrypt(SECRET, "u1", "openai")
    assert b64(K1) not in repr(ring) and repr(ring) == "KeyRing(primary='k1', keys=['k1'])"
    assert repr(sealed).startswith("Sealed(key_id='k1'") and "\\x" not in repr(sealed)
