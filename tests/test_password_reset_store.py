"""Password reset codes and tokens (app/accounts/password_reset.py) on SQLite, with a fake clock. No network."""
import threading
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from app import db
from app.accounts.credentials import (hash_password, looks_like_reset_token, new_reset_code, normalize_reset_code,
                                      reset_code_hash, token_hash, verify_password)
from app.accounts.password_reset import CODE_TTL, COOLDOWN, MAX_ATTEMPTS, TOKEN_TTL, ResetStore
from app.accounts.store import AccountStore
from app.providers.keys import redact

START = datetime(2026, 11, 16, 12, 0, 0, tzinfo=timezone.utc)
SECRET = b"s" * 32


class Clock:
    def __init__(self):
        self.now = START

    def __call__(self):
        return self.now

    def advance(self, **kw):
        self.now += timedelta(**kw)


@pytest.fixture
def engine(tmp_path):
    e = db.open_engine(db.sqlite_url(tmp_path / "t.sqlite3"))
    yield e
    e.dispose()


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def accounts(engine, clock):
    return AccountStore(engine, clock=clock)


@pytest.fixture
def resets(engine, clock):
    return ResetStore(engine, SECRET, clock=clock)


@pytest.fixture
def user(accounts):
    return accounts.create_user("a@example.com", hash_password("old password 1"), None)


def wrong(code: str) -> str:
    return f"{(int(code) + 1) % 1_000_000:06d}"


# ---------- the pure parts ----------

def test_codes_are_six_digits_and_tokens_have_the_rst_prefix():
    codes = {new_reset_code() for _ in range(200)}
    assert all(len(c) == 6 and c.isdigit() for c in codes) and len(codes) > 150
    assert normalize_reset_code(" 012 345 ") == "012345"
    for bad in ("12345", "1234567", "12a456", "１２３４５６", None, 123456):
        assert normalize_reset_code(bad) is None
    assert not looks_like_reset_token("rst_short") and not looks_like_reset_token(None)


def test_the_code_hash_is_keyed_and_bound_to_the_user():
    assert reset_code_hash("123456", "u1", SECRET) != reset_code_hash("123456", "u2", SECRET)
    assert reset_code_hash("123456", "u1", SECRET) != reset_code_hash("123456", "u1", b"t" * 32)


def test_reset_tokens_are_redacted():
    token = "rst_" + "A1b2" * 10 + "xyz"
    assert token not in redact(f"oops {token} leaked")


# ---------- codes ----------

def test_a_correct_code_gives_a_single_use_reset_token(resets, user, clock):
    code, code_id = resets.issue_code(user.id)
    token, expires = resets.verify_code(user.id, code)
    assert looks_like_reset_token(token) and expires == START + TOKEN_TTL
    assert resets.verify_code(user.id, code) is None                    # the code is used up


def test_a_new_code_after_the_cooldown_replaces_the_old_one(resets, user, clock):
    first, first_id = resets.issue_code(user.id)
    clock.advance(seconds=COOLDOWN.total_seconds())
    second, second_id = resets.issue_code(user.id)
    assert first_id != second_id
    if first != second:
        assert resets.verify_code(user.id, first) is None
    assert resets.verify_code(user.id, second) is not None


def test_inside_the_cooldown_nothing_changes(resets, user, clock):
    code, _ = resets.issue_code(user.id)
    clock.advance(seconds=59)
    assert resets.issue_code(user.id) is None
    assert resets.verify_code(user.id, code) is not None                # the emailed code still works


def test_the_cooldown_outlives_a_used_code(resets, user, clock):
    code, _ = resets.issue_code(user.id)
    resets.verify_code(user.id, code)
    assert resets.issue_code(user.id) is None
    clock.advance(seconds=60)
    assert resets.issue_code(user.id) is not None


def test_a_code_expires_after_ten_minutes(resets, user, clock):
    code, _ = resets.issue_code(user.id)
    clock.advance(seconds=CODE_TTL.total_seconds())
    assert resets.verify_code(user.id, code) is None


def test_five_wrong_tries_kill_the_code(resets, user):
    code, _ = resets.issue_code(user.id)
    for _ in range(MAX_ATTEMPTS):
        assert resets.verify_code(user.id, wrong(code)) is None
    assert resets.verify_code(user.id, code) is None                    # dead, even with the right code


def test_four_wrong_tries_still_allow_the_right_code(resets, user):
    code, _ = resets.issue_code(user.id)
    for _ in range(MAX_ATTEMPTS - 1):
        resets.verify_code(user.id, wrong(code))
    assert resets.verify_code(user.id, code) is not None


def test_a_malformed_code_counts_nothing(resets, user, engine):
    code, _ = resets.issue_code(user.id)
    for _ in range(10):
        assert resets.verify_code(user.id, "abc") is None
    assert resets.verify_code(user.id, code) is not None


def test_no_account_and_no_code_are_just_none(resets, user):
    assert resets.verify_code(None, "123456") is None
    assert resets.verify_code(user.id, "123456") is None                # never issued


def test_a_code_hash_copied_to_another_user_doesnt_work(resets, accounts, user, engine):
    other = accounts.create_user("b@example.com", hash_password("other password"), None)
    code, _ = resets.issue_code(user.id)
    resets.issue_code(other.id)
    t = db.password_reset_codes_table
    with engine.begin() as c:
        mine = c.execute(select(t.c.code_hash).where(t.c.user_id == user.id)).scalar()
        c.execute(t.update().where(t.c.user_id == other.id).values(code_hash=mine))
    assert resets.verify_code(other.id, code) is None
    assert resets.verify_code(user.id, code) is not None


def test_parallel_wrong_guesses_get_at_most_five_comparisons(resets, user, engine):
    code, _ = resets.issue_code(user.id)
    barrier = threading.Barrier(12)

    def guess():
        barrier.wait()
        resets.verify_code(user.id, wrong(code))
    threads = [threading.Thread(target=guess) for _ in range(12)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    with engine.connect() as c:
        assert c.execute(select(db.password_reset_codes_table.c.attempts)).scalar() == MAX_ATTEMPTS
    assert resets.verify_code(user.id, code) is None


# ---------- tokens ----------

def test_complete_sets_the_password_and_revokes_every_session(resets, accounts, user, clock):
    tokens = [accounts.create_session(user.id)[0] for _ in range(3)]
    code, _ = resets.issue_code(user.id)
    token, _ = resets.verify_code(user.id, code)
    assert resets.complete(token, hash_password("new password 2")) == user.id
    assert all(accounts.session_for_token(t) is None for t in tokens)
    assert verify_password(accounts.user_by_id(user.id).password_hash, "new password 2")
    assert resets.complete(token, hash_password("third password")) is None   # single use


def test_a_reset_token_expires_after_fifteen_minutes(resets, user, clock):
    code, _ = resets.issue_code(user.id)
    token, _ = resets.verify_code(user.id, code)
    clock.advance(seconds=TOKEN_TTL.total_seconds())
    assert resets.complete(token, hash_password("new password 2")) is None


def test_a_newer_token_cancels_the_older_one(resets, user, clock):
    code, _ = resets.issue_code(user.id)
    old, _ = resets.verify_code(user.id, code)
    clock.advance(seconds=60)
    code, _ = resets.issue_code(user.id)
    new, _ = resets.verify_code(user.id, code)
    assert resets.complete(old, hash_password("new password 2")) is None
    assert resets.complete(new, hash_password("new password 2")) == user.id


def test_complete_ends_an_outstanding_code(resets, user, clock):
    code, _ = resets.issue_code(user.id)
    token, _ = resets.verify_code(user.id, code)
    clock.advance(seconds=60)
    code2, _ = resets.issue_code(user.id)                               # e.g. requested again meanwhile
    resets.complete(token, hash_password("new password 2"))
    assert resets.verify_code(user.id, code2) is None


def test_unknown_tokens_are_none(resets):
    assert resets.complete("rst_" + "0" * 43, hash_password("new password 2")) is None


def test_parallel_completes_with_one_token_only_one_wins(resets, user):
    code, _ = resets.issue_code(user.id)
    token, _ = resets.verify_code(user.id, code)
    hashed = hash_password("new password 2")
    barrier, results = threading.Barrier(6), []

    def go():
        barrier.wait()
        results.append(resets.complete(token, hashed))
    threads = [threading.Thread(target=go) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results.count(user.id) == 1 and results.count(None) == 5


# ---------- storage ----------

def test_only_hashes_are_stored_and_rows_go_with_the_user(resets, accounts, user, engine):
    code, _ = resets.issue_code(user.id)
    token, _ = resets.verify_code(user.id, code)
    with engine.connect() as c:
        code_row = c.execute(select(db.password_reset_codes_table)).one()
        token_row = c.execute(select(db.password_reset_tokens_table)).one()
    stored = " ".join(str(v) for v in (*code_row, *token_row))
    assert code not in stored and token not in stored and "example.com" not in stored
    assert code_row.code_hash == reset_code_hash(code, user.id, SECRET)
    assert token_row.token_hash == token_hash(token)
    accounts.delete_user(user.id)
    with engine.connect() as c:
        assert c.execute(select(db.password_reset_codes_table)).all() == []
        assert c.execute(select(db.password_reset_tokens_table)).all() == []
