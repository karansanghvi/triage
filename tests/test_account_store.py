"""Users, sessions (app/accounts/store.py) and sign-in rate limits (app/accounts/ratelimit.py), on SQLite.
No network. Time comes from a fake clock."""
import threading
from datetime import datetime, timedelta, timezone

import pytest

from app import db
from app.accounts.credentials import token_hash
from app.accounts.ratelimit import LIMITS, AuthRateLimiter, RateLimited, email_subject
from app.accounts.store import AccountStore, EmailTaken

START = datetime(2026, 11, 15, 12, 0, 0, tzinfo=timezone.utc)


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
def store(engine, clock):
    return AccountStore(engine, clock)


def new_user(store, email="ada@example.com"):
    return store.create_user(email, "$argon2id$fake", "Ada")


# ---------- users ----------

def test_create_and_find_a_user(store):
    user = new_user(store)
    assert store.user_by_email("ada@example.com") == user == store.user_by_id(user.id)
    assert user.public() == {"id": user.id, "email": "ada@example.com", "display_name": "Ada",
                             "preferred_provider": None, "created_at": "2026-11-15T12:00:00Z"}
    assert "password_hash" not in user.public() and "fake" not in repr(user)


def test_a_taken_email_is_refused(store):
    new_user(store)
    with pytest.raises(EmailTaken):
        new_user(store)


def test_update_user_sets_updated_at(store, engine, clock):
    user = new_user(store)
    clock.advance(minutes=5)
    updated = store.update_user(user.id, display_name=None, preferred_provider="anthropic")
    assert (updated.display_name, updated.preferred_provider) == (None, "anthropic")
    with engine.connect() as c:
        assert db.aware(c.execute(db.users_table.select()).one().updated_at) == clock.now


# ---------- sessions ----------

def test_a_session_token_is_returned_once_and_only_its_hash_is_stored(store, engine):
    user = new_user(store)
    token, session = store.create_session(user.id)
    assert token.startswith("trs_") and session.expires_at == START + timedelta(days=30)
    with engine.connect() as c:
        row = c.execute(db.sessions_table.select()).one()
    assert row.token_hash == token_hash(token) and token not in str(tuple(row))
    found_session, found_user = store.session_for_token(token)
    assert found_session.id == session.id and found_user.id == user.id


def test_unknown_and_altered_tokens_find_nothing(store):
    token, _ = store.create_session(new_user(store).id)
    assert store.session_for_token(token[:-1] + ("A" if token[-1] != "A" else "B")) is None
    assert store.session_for_token("trs_" + "0" * 43) is None


def test_idle_expiry_after_seven_days_without_use(store, clock):
    token, _ = store.create_session(new_user(store).id)
    clock.advance(days=6, hours=23)
    assert store.session_for_token(token) is not None             # used: the idle clock restarts
    clock.advance(days=6, hours=23)
    assert store.session_for_token(token) is not None
    clock.advance(days=7)
    assert store.session_for_token(token) is None


def test_absolute_expiry_after_thirty_days_even_when_used_daily(store, clock):
    token, _ = store.create_session(new_user(store).id)
    for _ in range(29):
        clock.advance(days=1)
        assert store.session_for_token(token) is not None
    clock.advance(days=1)
    assert store.session_for_token(token) is None


def test_last_used_is_written_at_most_once_a_minute(store, engine, clock):
    token, _ = store.create_session(new_user(store).id)
    from sqlalchemy import event
    updates = []

    def record(conn, cursor, statement, *args):
        if statement.lstrip().upper().startswith("UPDATE"):
            updates.append(statement)
    event.listen(engine, "before_cursor_execute", record)
    for _ in range(5):
        clock.advance(seconds=10)
        store.session_for_token(token)
    assert len(updates) == 0
    clock.advance(seconds=20)
    store.session_for_token(token)
    store.session_for_token(token)
    assert len(updates) == 1
    event.remove(engine, "before_cursor_execute", record)


def test_revoking_one_session_and_all_others(store):
    user = new_user(store)
    (a, sa), (b, _), (c, _) = (store.create_session(user.id) for _ in range(3))
    store.revoke_session(sa.id)
    assert store.session_for_token(a) is None and store.session_for_token(b) is not None
    sb = store.session_for_token(b)[0]
    assert store.revoke_other_sessions(user.id, keep_session_id=sb.id) == 1      # only c was still open
    assert store.session_for_token(b) is not None and store.session_for_token(c) is None


def test_other_users_sessions_are_untouched(store):
    ada, bob = new_user(store), new_user(store, "bob@example.com")
    ta, sa = store.create_session(ada.id)
    tb, _ = store.create_session(bob.id)
    store.revoke_other_sessions(ada.id, keep_session_id=sa.id)
    assert store.session_for_token(tb) is not None


def test_deleting_a_user_removes_them_and_their_sessions(store, engine):
    user = new_user(store)
    token, _ = store.create_session(user.id)
    store.delete_user(user.id)
    assert store.user_by_id(user.id) is None and store.session_for_token(token) is None
    with engine.connect() as c:
        assert c.execute(db.sessions_table.select()).all() == []


# ---------- rate limits ----------

@pytest.fixture
def limiter(engine, clock):
    return AuthRateLimiter(engine, clock)


@pytest.mark.parametrize("kind", list(LIMITS))
def test_each_limit_then_retry_after_until_the_oldest_leaves_the_window(limiter, clock, kind):
    limit = LIMITS[kind]
    for _ in range(limit.attempts):
        limiter.hit(kind, "visitor-a")
        clock.advance(seconds=1)
    with pytest.raises(RateLimited) as error:
        limiter.hit(kind, "visitor-a")
    assert error.value.retry_after == int(limit.window.total_seconds()) - limit.attempts
    limiter.hit(kind, "visitor-b")                                 # someone else is unaffected
    clock.advance(seconds=error.value.retry_after)
    limiter.hit(kind, "visitor-a")                                 # the oldest has left the window


def test_a_refused_attempt_counts_nothing(limiter):
    for _ in range(5):
        limiter.hit("signup_ip", "v")
    for _ in range(3):
        with pytest.raises(RateLimited):
            limiter.hit("signup_ip", "v")
    assert limiter.count("signup_ip", "v") == 5


def test_check_and_record_for_login_failures(limiter):
    subject = email_subject("ada@example.com", b"secret")
    for _ in range(5):
        limiter.check("login_failure_email", subject)
        limiter.record("login_failure_email", subject)
    with pytest.raises(RateLimited):
        limiter.check("login_failure_email", subject)


def test_email_subjects_are_keyed_hashes(engine, limiter):
    subject = email_subject("ada@example.com", b"secret")
    assert "ada" not in subject and subject != email_subject("ada@example.com", b"other-secret")
    limiter.record("login_failure_email", subject)
    with engine.connect() as c:
        assert "ada" not in str(c.execute(db.auth_attempts_table.select()).all())


def test_old_attempts_are_deleted(limiter, engine, clock):
    limiter.hit("login_ip", "v")
    clock.advance(days=2)
    limiter.hit("login_ip", "w")
    with engine.connect() as c:
        assert [r.subject for r in c.execute(db.auth_attempts_table.select())] == ["w"]


def test_concurrent_attempts_never_overshoot(engine, clock):
    limiters = [AuthRateLimiter(engine, clock) for _ in range(3)]
    barrier, allowed, refused, lock = threading.Barrier(30), [], [], threading.Lock()

    def go(i):
        barrier.wait()
        try:
            limiters[i % 3].hit("login_ip", "v")
            with lock:
                allowed.append(i)
        except RateLimited:
            with lock:
                refused.append(i)

    threads = [threading.Thread(target=go, args=(i,)) for i in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(allowed) == 10 and len(refused) == 20
