"""Per-account chat limits (app/accounts/limits.py, config/accounts.yaml), on SQLite. No network."""
import threading
from datetime import datetime, timedelta, timezone

import pytest

from app import db
from app.accounts.limits import AccountLimiter, AccountLimitError, AccountLimits, load_account_limits
from app.accounts.store import AccountStore

START = datetime(2026, 11, 16, 23, 59, 0, tzinfo=timezone.utc)


class Clock:
    def __init__(self):
        self.now = START

    def __call__(self):
        return self.now


@pytest.fixture
def setup(tmp_path):
    engine = db.open_engine(db.sqlite_url(tmp_path / "t.sqlite3"))
    accounts = AccountStore(engine)
    ada = accounts.create_user("ada@example.com", "h", None)
    bob = accounts.create_user("bob@example.com", "h", None)
    yield engine, accounts, ada, bob, Clock()
    engine.dispose()


def test_the_config_file_has_the_approved_defaults():
    assert load_account_limits() == AccountLimits(requests_per_minute=60, requests_per_day=10_000, retention_days=7)


@pytest.mark.parametrize("text, message", [("requests_per_minute: 60\nrequests_per_day: 10\n", "missing"),
                                           ("requests_per_minute: 0\nrequests_per_day: 1\nretention_days: 7\n",
                                            "above 0"),
                                           ("requests_per_minute: 1\nrequests_per_day: 1\nretention_days: 7\nx: 1\n",
                                            "unknown")])
def test_bad_config_is_refused(tmp_path, text, message):
    path = tmp_path / "accounts.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match=message):
        load_account_limits(path)


def test_per_minute_limit_then_retry_after(setup):
    engine, _, ada, bob, clock = setup
    limiter = AccountLimiter(AccountLimits(requests_per_minute=3, requests_per_day=100), engine, clock)
    for _ in range(3):
        limiter.admit(ada.id)
        clock.now += timedelta(seconds=5)
    with pytest.raises(AccountLimitError) as error:
        limiter.admit(ada.id)
    assert (error.value.which, error.value.retry_after) == ("minute", 45)
    limiter.admit(bob.id)                                          # per user
    assert limiter.used_today(ada.id) == 3                         # a refusal counts nothing
    clock.now += timedelta(seconds=45)
    limiter.admit(ada.id)


def test_per_day_limit_until_midnight_utc_and_shared_by_instances(setup):
    engine, _, ada, _, clock = setup
    limits = AccountLimits(requests_per_minute=100, requests_per_day=3)
    a, b = AccountLimiter(limits, engine, clock), AccountLimiter(limits, engine, clock)
    assert [a.admit(ada.id), b.admit(ada.id), a.admit(ada.id)] == [2, 1, 0]
    with pytest.raises(AccountLimitError) as error:
        b.admit(ada.id)
    assert (error.value.which, error.value.retry_after) == ("day", 60)          # 23:59 -> 00:00
    clock.now += timedelta(minutes=1)
    assert a.admit(ada.id) == 2                                   # a new UTC day


def test_concurrent_requests_never_overshoot_the_day(setup):
    engine, _, ada, _, clock = setup
    limiters = [AccountLimiter(AccountLimits(requests_per_minute=1_000, requests_per_day=10), engine, clock)
                for _ in range(3)]
    barrier, admitted, refused, lock = threading.Barrier(30), [], [], threading.Lock()

    def go(i):
        barrier.wait()
        try:
            limiters[i % 3].admit(ada.id)
            with lock:
                admitted.append(i)
        except AccountLimitError:
            with lock:
                refused.append(i)

    threads = [threading.Thread(target=go, args=(i,)) for i in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(admitted) == 10 and len(refused) == 20


def test_old_days_are_cleaned_and_rows_go_with_the_user(setup):
    engine, accounts, ada, bob, clock = setup
    limiter = AccountLimiter(AccountLimits(), engine, clock)
    limiter.admit(ada.id)
    limiter.admit(bob.id)
    clock.now += timedelta(days=9)
    assert limiter.cleanup() == 2
    limiter.admit(ada.id)
    accounts.delete_user(ada.id)
    with engine.connect() as c:
        assert c.execute(db.account_day_table.select()).all() == []
