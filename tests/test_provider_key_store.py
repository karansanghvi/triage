"""Saved provider keys in the database (app/accounts/provider_keys.py), on SQLite. No network."""
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from app import db
from app.accounts.encryption import DecryptionError, KeyRing
from app.accounts.provider_keys import ProviderKeyStore
from app.accounts.store import AccountStore

SECRET = "sk-test-SECRET123"
K1, K2 = ("k1", secrets.token_bytes(32)), ("k2", secrets.token_bytes(32))


class Clock:
    def __init__(self):
        self.now = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now


@pytest.fixture
def setup(tmp_path):
    engine = db.open_engine(db.sqlite_url(tmp_path / "t.sqlite3"))
    clock = Clock()
    accounts = AccountStore(engine, clock)
    ada = accounts.create_user("ada@example.com", "$argon2id$fake", None)
    bob = accounts.create_user("bob@example.com", "$argon2id$fake", None)
    yield engine, ProviderKeyStore(engine, KeyRing([K1]), clock), accounts, ada, bob, clock, tmp_path
    engine.dispose()


def test_save_list_and_decrypt(setup):
    engine, keys, _, ada, _, _, _ = setup
    saved = keys.save(ada.id, "openai", SECRET, "valid", None, verified=True)
    assert saved.public() == {"provider": "openai", "last4": "T123", "status": "valid",
                              "verified_at": "2026-11-16T12:00:00Z", "created_at": "2026-11-16T12:00:00Z",
                              "updated_at": "2026-11-16T12:00:00Z"}
    assert [k.public() for k in keys.list(ada.id)] == [saved.public()]
    assert keys.decrypt(ada.id, "openai") == SECRET and keys.decrypt(ada.id, "groq") is None


def test_the_database_holds_only_ciphertext(setup):
    engine, keys, _, ada, _, _, tmp_path = setup
    keys.save(ada.id, "openai", SECRET, "valid", None, verified=True)
    engine.dispose()
    raw = b"".join(p.read_bytes() for p in tmp_path.iterdir() if p.name.startswith("t.sqlite3"))
    assert SECRET.encode() not in raw and b"SECRET123" not in raw
    row = sqlite3.connect(tmp_path / "t.sqlite3").execute("SELECT enc_key_id, last4 FROM provider_keys").fetchone()
    assert row == ("k1", "T123")


def test_saving_again_replaces_the_key(setup):
    _, keys, _, ada, _, clock, _ = setup
    keys.save(ada.id, "groq", "gsk_old_key_aaaa", "valid", None, verified=True)
    clock.now += timedelta(hours=1)
    keys.save(ada.id, "groq", "gsk_new_key_bbbb", "unverified", "timeout", verified=False)
    (only,) = keys.list(ada.id)
    assert (only.last4, only.status, only.verified_at) == ("bbbb", "unverified", None)
    assert only.created_at == clock.now and keys.decrypt(ada.id, "groq") == "gsk_new_key_bbbb"


def test_one_key_per_provider_and_users_apart(setup):
    _, keys, _, ada, bob, _, _ = setup
    for provider in ("gemini", "openai", "groq", "anthropic"):
        keys.save(ada.id, provider, f"key-for-{provider}", "valid", None, verified=True)
    keys.save(bob.id, "openai", "bobs-openai-key", "valid", None, verified=True)
    assert [k.provider for k in keys.list(ada.id)] == ["openai", "anthropic", "groq", "gemini"]
    assert keys.decrypt(bob.id, "openai") == "bobs-openai-key" and keys.decrypt(ada.id, "openai") == "key-for-openai"


def test_a_row_copied_to_another_user_doesnt_decrypt(setup):
    engine, keys, _, ada, bob, _, _ = setup
    keys.save(ada.id, "openai", SECRET, "valid", None, verified=True)
    keys.save(bob.id, "openai", "bobs-openai-key", "valid", None, verified=True)
    t = db.provider_keys_table
    with engine.begin() as c:
        stolen = c.execute(t.select().where(t.c.user_id == ada.id)).one()
        c.execute(t.update().where(t.c.user_id == bob.id).values(ciphertext=stolen.ciphertext, nonce=stolen.nonce))
    with pytest.raises(DecryptionError):
        keys.decrypt(bob.id, "openai")


def test_rotation_reads_old_rows_and_writes_with_the_new_key(setup):
    engine, keys, _, ada, _, clock, _ = setup
    keys.save(ada.id, "openai", SECRET, "valid", None, verified=True)
    rotated = ProviderKeyStore(engine, KeyRing([K2, K1]), clock)
    assert rotated.decrypt(ada.id, "openai") == SECRET
    rotated.save(ada.id, "groq", "gsk_after_rotation", "valid", None, verified=True)
    with engine.connect() as c:
        ids = dict(c.execute(db.provider_keys_table.select().with_only_columns(
            db.provider_keys_table.c.provider, db.provider_keys_table.c.enc_key_id)).all())
    assert ids == {"openai": "k1", "groq": "k2"}
    with pytest.raises(DecryptionError):
        ProviderKeyStore(engine, KeyRing([K2]), clock).decrypt(ada.id, "openai")   # k1 dropped too early


def test_set_status_and_delete(setup):
    _, keys, _, ada, _, clock, _ = setup
    keys.save(ada.id, "anthropic", SECRET, "unverified", "timeout", verified=False)
    clock.now += timedelta(minutes=5)
    updated = keys.set_status(ada.id, "anthropic", "invalid", "rejected by the provider (401)", verified=True)
    assert (updated.status, updated.status_detail, updated.verified_at) == ("invalid", "rejected by the provider (401)",
                                                                            clock.now)
    assert keys.delete(ada.id, "anthropic") is True and keys.delete(ada.id, "anthropic") is False
    assert keys.list(ada.id) == []


def test_deleting_the_user_deletes_their_keys(setup):
    engine, keys, accounts, ada, bob, _, _ = setup
    keys.save(ada.id, "openai", SECRET, "valid", None, verified=True)
    keys.save(bob.id, "openai", "bobs-openai-key", "valid", None, verified=True)
    accounts.delete_user(ada.id)
    with engine.connect() as c:
        assert [r.user_id for r in c.execute(db.provider_keys_table.select())] == [bob.id]


def test_without_encryption_keys_nothing_is_saved(setup):
    engine, _, _, ada, _, clock, _ = setup
    with pytest.raises(RuntimeError, match="TRIAGE_ENCRYPTION_KEYS"):
        ProviderKeyStore(engine, None, clock).save(ada.id, "openai", SECRET, "valid", None, verified=True)
