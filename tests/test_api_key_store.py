"""Account API keys in the database (app/accounts/api_keys.py) and their format (credentials.py), on SQLite."""
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

import pytest

from app import db
from app.accounts import credentials as c
from app.accounts.api_keys import MAX_ACTIVE_KEYS, ApiKeyStore, TooManyKeys
from app.accounts.store import AccountStore
from app.providers.keys import REDACTED, redact

SECRET = b"test-hash-secret"


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
    yield engine, ApiKeyStore(engine, SECRET, clock), accounts, ada, bob, clock, tmp_path
    engine.dispose()


# ---------- the format ----------

def test_keys_are_trk_plus_256_random_bits_in_base62():
    keys = {c.new_api_key() for _ in range(200)}
    assert len(keys) == 200
    for k in keys:
        assert k.startswith("trk_") and len(k) == 47 and c.looks_like_api_key(k)
        assert c.api_key_public_id(k) == k[:12] and not c.looks_like_session_token(k)


@pytest.mark.parametrize("value", ["trk_short", "trs_" + "a" * 43, "trk_" + "!" * 43, "TRK_" + "a" * 43, "",
                                   "trk_" + "a" * 44])
def test_things_that_are_not_api_keys(value):
    assert not c.looks_like_api_key(value)


def test_full_keys_are_redacted_but_public_ids_stay_readable():
    key = c.new_api_key()
    text = f"bad key {key}, public id {c.api_key_public_id(key)}"
    out = redact(text)
    assert key not in out and out.count(REDACTED) == 1 and c.api_key_public_id(key) in out


def test_the_hash_is_keyed():
    key = c.new_api_key()
    assert c.api_key_hash(key, SECRET) != c.api_key_hash(key, b"other") and key not in c.api_key_hash(key, SECRET)


# ---------- the store ----------

def test_create_returns_the_key_once_and_stores_only_its_hash(setup):
    engine, keys, _, ada, _, _, tmp_path = setup
    key, row = keys.create(ada.id, "laptop")
    assert row.public_id == key[:12] and row.name == "laptop"
    assert row.public() == {"id": row.id, "name": "laptop", "public_id": key[:12],
                            "created_at": "2026-11-16T12:00:00Z", "last_used_at": None, "revoked_at": None}
    engine.dispose()
    raw = b"".join(p.read_bytes() for p in tmp_path.iterdir() if p.name.startswith("t.sqlite3"))
    assert key.encode() not in raw and key[12:].encode() not in raw
    stored = sqlite3.connect(tmp_path / "t.sqlite3").execute("SELECT public_id, secret_hash FROM api_keys").fetchone()
    assert stored == (key[:12], c.api_key_hash(key, SECRET))


def test_lookup_finds_the_key_and_nothing_else(setup):
    _, keys, _, ada, _, _, _ = setup
    key, row = keys.create(ada.id, "laptop")
    assert keys.lookup(key) == row
    altered = key[:-1] + ("A" if key[-1] != "A" else "B")
    assert keys.lookup(altered) is None                            # same public id, different secret
    assert keys.lookup(c.new_api_key()) is None and keys.lookup("trk_short") is None
    assert ApiKeyStore(keys.engine, b"another-secret").lookup(key) is None


def test_lookup_compares_in_constant_time(setup, monkeypatch):
    _, keys, _, ada, _, _, _ = setup
    key, _ = keys.create(ada.id, "laptop")
    calls = []
    import app.accounts.api_keys as module
    monkeypatch.setattr(module, "hashes_match", lambda a, b: calls.append((a, b)) or c.hashes_match(a, b))
    keys.lookup(key)
    assert len(calls) == 1 and calls[0][1] == c.api_key_hash(key, SECRET)


def test_list_newest_first_with_revoked_ones(setup):
    _, keys, _, ada, bob, clock, _ = setup
    ids = []
    for name in ("first", "second", "third"):
        ids.append(keys.create(ada.id, name)[1].id)
        clock.now += timedelta(minutes=1)
    keys.create(bob.id, "bobs")
    keys.revoke(ada.id, ids[1])
    listed = keys.list(ada.id)
    assert [k.name for k in listed] == ["third", "second", "first"] and listed[1].revoked_at is not None


def test_revoke_only_your_own_and_twice_is_harmless(setup):
    _, keys, _, ada, bob, clock, _ = setup
    key, row = keys.create(ada.id, "laptop")
    assert keys.revoke(bob.id, row.id) is False and keys.lookup(key) is not None   # not bob's
    assert keys.revoke(ada.id, "no-such-id") is False
    assert keys.revoke(ada.id, row.id) is True and keys.lookup(key) is None
    first = keys.list(ada.id)[0].revoked_at
    clock.now += timedelta(hours=1)
    assert keys.revoke(ada.id, row.id) is True and keys.list(ada.id)[0].revoked_at == first


def test_at_most_twenty_active_keys(setup):
    _, keys, _, ada, bob, _, _ = setup
    made = [keys.create(ada.id, f"k{i}")[1] for i in range(MAX_ACTIVE_KEYS)]
    with pytest.raises(TooManyKeys):
        keys.create(ada.id, "one too many")
    keys.create(bob.id, "bobs")                                    # per user
    keys.revoke(ada.id, made[0].id)
    keys.create(ada.id, "after revoking one")                      # revoked keys don't count


def test_the_cap_holds_under_concurrency(setup):
    engine, _, _, ada, _, clock, _ = setup
    stores = [ApiKeyStore(engine, SECRET, clock) for _ in range(3)]
    barrier, made, refused, lock = threading.Barrier(30), [], [], threading.Lock()

    def go(i):
        barrier.wait()
        try:
            stores[i % 3].create(ada.id, f"k{i}")
            with lock:
                made.append(i)
        except TooManyKeys:
            with lock:
                refused.append(i)

    threads = [threading.Thread(target=go, args=(i,)) for i in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(made) == 20 and len(refused) == 10


def test_last_used_is_written_at_most_once_a_minute(setup):
    _, keys, _, ada, _, clock, _ = setup
    key, _ = keys.create(ada.id, "laptop")
    keys.touch(keys.lookup(key))
    assert keys.lookup(key).last_used_at == clock.now
    first = clock.now
    clock.now += timedelta(seconds=30)
    keys.touch(keys.lookup(key))
    assert keys.lookup(key).last_used_at == first
    clock.now += timedelta(seconds=31)
    keys.touch(keys.lookup(key))
    assert keys.lookup(key).last_used_at == clock.now


def test_deleting_the_user_deletes_their_keys(setup):
    engine, keys, accounts, ada, bob, _, _ = setup
    keys.create(ada.id, "laptop")
    keys.create(bob.id, "bobs")
    accounts.delete_user(ada.id)
    with engine.connect() as conn:
        assert [r.user_id for r in conn.execute(db.api_keys_table.select())] == [bob.id]
