"""scripts/rotate_provider_keys.py: re-encrypt saved provider keys with the first configured key. On SQLite."""
import base64
import json
import secrets

import pytest
from sqlalchemy import select

from app import db
from app.accounts.encryption import KeyRing
from app.accounts.provider_keys import ProviderKeyStore
from app.accounts.store import AccountStore
from scripts import rotate_provider_keys as rotation

K1, K2, K3 = (("k1", secrets.token_bytes(32)), ("k2", secrets.token_bytes(32)), ("k3", secrets.token_bytes(32)))
SECRETS = {"openai": "sk-test-SECRET-openai-1111", "groq": "gsk_test_SECRET_groq_2222"}


def env_value(*keys):
    return ",".join(f"{kid}:{base64.b64encode(raw).decode()}" for kid, raw in keys)


@pytest.fixture
def saved(tmp_path):
    """Five users with two saved keys each, all encrypted with k1."""
    engine = db.open_engine(db.sqlite_url(tmp_path / "t.sqlite3"))
    accounts, store = AccountStore(engine), ProviderKeyStore(engine, KeyRing([K1]))
    users = [accounts.create_user(f"u{i}@example.com", "h", None).id for i in range(5)]
    for user in users:
        for provider, key in SECRETS.items():
            store.save(user, provider, key, "valid", None, verified=True)
    yield engine, users, tmp_path
    engine.dispose()


def key_ids(engine):
    with engine.connect() as c:
        return sorted(r.enc_key_id for r in c.execute(select(db.provider_keys_table.c.enc_key_id)))


def test_every_row_moves_to_the_first_key_and_still_decrypts(saved):
    engine, users, _ = saved
    rotated = KeyRing([K2, K1])
    result = rotation.rotate(engine, rotated, batch_size=3)
    assert (result["rows"], result["re_encrypted"], result["already_current"]) == (10, 10, 0)
    assert result["by_key_before"] == {"k1": 10} and result["by_key_after"] == {"k2": 10}
    after = ProviderKeyStore(engine, KeyRing([K2]))                     # k1 can now be dropped
    for user in users:
        assert after.usable_keys(user) == SECRETS


def test_running_again_changes_nothing(saved):
    engine, _, _ = saved
    rotated = KeyRing([K2, K1])
    rotation.rotate(engine, rotated)
    with engine.connect() as c:
        before = c.execute(select(db.provider_keys_table)).fetchall()
    again = rotation.rotate(engine, rotated)
    assert (again["re_encrypted"], again["already_current"]) == (0, 10)
    with engine.connect() as c:
        assert c.execute(select(db.provider_keys_table)).fetchall() == before


def test_mixed_old_keys_all_move(saved):
    engine, users, _ = saved
    ProviderKeyStore(engine, KeyRing([K3, K1])).save(users[0], "openai", "sk-test-SECRET-newer-3333", "valid",
                                                     None, verified=True)
    result = rotation.rotate(engine, KeyRing([K2, K3, K1]))
    assert result["by_key_before"] == {"k1": 9, "k3": 1} and result["by_key_after"] == {"k2": 10}


def test_refuses_and_changes_nothing_if_any_row_cant_be_decrypted(saved, capsys):
    engine, _, _ = saved
    with engine.connect() as c:
        before = c.execute(select(db.provider_keys_table)).fetchall()
    with pytest.raises(SystemExit, match="refusing to rotate: 10 saved key"):
        rotation.rotate(engine, KeyRing([K2]))                          # k1 missing: nothing readable
    with engine.connect() as c:
        assert c.execute(select(db.provider_keys_table)).fetchall() == before
    assert json.loads(capsys.readouterr().out)["unreadable"] == 10


def test_one_damaged_row_blocks_the_whole_run(saved):
    engine, users, _ = saved
    t = db.provider_keys_table
    with engine.begin() as c:
        row = c.execute(select(t).where(t.c.user_id == users[2], t.c.provider == "groq")).one()
        c.execute(t.update().where(t.c.user_id == users[2], t.c.provider == "groq")
                  .values(ciphertext=bytes([row.ciphertext[0] ^ 1]) + row.ciphertext[1:]))
    with pytest.raises(SystemExit, match="1 saved key"):
        rotation.rotate(engine, KeyRing([K2, K1]))
    assert key_ids(engine) == ["k1"] * 10


def test_dry_run_counts_only(saved):
    engine, _, _ = saved
    result = rotation.rotate(engine, KeyRing([K2, K1]), dry_run=True)
    assert result["would_re_encrypt"] == 10 and result["re_encrypted"] == 0 and key_ids(engine) == ["k1"] * 10


def test_a_row_saved_again_meanwhile_is_left_alone(saved, monkeypatch):
    engine, users, _ = saved
    rotated = KeyRing([K2, K1])
    store = ProviderKeyStore(engine, rotated)
    real_decrypt, calls = rotated.decrypt, []

    def decrypt_then_user_saves(sealed, user_id, provider):
        plain = real_decrypt(sealed, user_id, provider)
        calls.append(1)
        if len(calls) == 11:                    # the first decrypt of the re-encryption pass (10 checks before it)
            store.save(user_id, provider, "sk-test-SECRET-resaved-4444", "valid", None, verified=True)
        return plain

    monkeypatch.setattr(rotated, "decrypt", decrypt_then_user_saves)
    result = rotation.rotate(engine, rotated, batch_size=100)
    assert result["re_encrypted"] + result["changed_meanwhile"] == 10 and result["changed_meanwhile"] == 1
    usable = {u: store.usable_keys(u) for u in users}
    assert sum(1 for keys in usable.values() for k in keys.values() if "resaved" in k) == 1


def test_main_prints_counts_and_never_a_key(saved, monkeypatch, capsys):
    engine, _, tmp_path = saved
    monkeypatch.setenv("DATABASE_URL", db.sqlite_url(tmp_path / "t.sqlite3"))
    monkeypatch.setenv("TRIAGE_ENCRYPTION_KEYS", env_value(K2, K1))
    rotation.main(["--batch-size", "4"])
    out = capsys.readouterr().out
    assert json.loads(out)["re_encrypted"] == 10
    assert "SECRET" not in out and base64.b64encode(K1[1]).decode() not in out


def test_main_refuses_without_valid_encryption_keys(saved, monkeypatch):
    engine, _, tmp_path = saved
    monkeypatch.setenv("DATABASE_URL", db.sqlite_url(tmp_path / "t.sqlite3"))
    with pytest.raises(Exception, match="TRIAGE_ENCRYPTION_KEYS is not set"):
        rotation.main([])
    with pytest.raises(SystemExit, match="batch-size"):
        rotation.main(["--batch-size", "0"])
