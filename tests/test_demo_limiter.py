"""DemoLimiter: the expected behaviour (app/api/demo.py). No network.

Every test runs twice: in memory (path=None) and on a SQLite file. Time comes from a fake clock (aware UTC
datetimes) so day rollover and the per-minute window can be tested without waiting.

These tests describe the contract in DemoLimiter's docstrings; they fail until DemoLimiter is implemented.
"""
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

import pytest

from app.api.demo import Admission, DemoLimiter, DemoLimitError, DemoSettings, DemoStatus, visitor_id

START = datetime(2026, 11, 13, 12, 0, 0, tzinfo=timezone.utc)
SMALL = DemoSettings(visitor_daily_requests=3, daily_requests=5, daily_tokens=1_000, daily_model_calls=10,
                     per_minute_requests=100, public_byok_visitor_daily_requests=2, retention_days=7)


class Clock:
    def __init__(self, now=START):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, **kw):
        self.now += timedelta(**kw)


@pytest.fixture(params=["memory", "sqlite"])
def storage(request, tmp_path):
    return None if request.param == "memory" else tmp_path / "usage.sqlite3"


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def make(storage, clock):
    made = []

    def build(settings=SMALL, path=storage, clk=clock):
        limiter = DemoLimiter(settings, path=path, clock=clk)
        made.append(limiter)
        return limiter

    yield build
    for limiter in made:
        try:
            limiter.close()
        except Exception:
            pass


def code_of(call):
    with pytest.raises(DemoLimitError) as error:
        call()
    return error.value


SECONDS_TO_MIDNIGHT = 12 * 3600           # from START (12:00 UTC)


# ---------- per-visitor daily cap ----------

def test_admission_counts_down_the_visitors_requests(make):
    limiter = make()
    admissions = [limiter.admit_demo("v1", 10) for _ in range(3)]
    assert all(isinstance(a, Admission) for a in admissions)
    assert [a.remaining for a in admissions] == [2, 1, 0]
    a = admissions[0]
    assert (a.kind, a.visitor, a.day, a.reserved_tokens, a.limit) == ("demo", "v1", "2026-11-13", 10, 3)


def test_the_visitor_cap_is_a_demo_limit_reached_until_midnight(make):
    limiter = make()
    for _ in range(3):
        limiter.admit_demo("v1", 10)
    error = code_of(lambda: limiter.admit_demo("v1", 10))
    assert error.code == "demo_limit_reached" and error.retry_after == SECONDS_TO_MIDNIGHT and error.reason is None


def test_visitors_are_counted_separately(make):
    limiter = make()
    for _ in range(3):
        limiter.admit_demo("v1", 10)
    assert limiter.admit_demo("v2", 10).remaining == 2


def test_a_refused_request_counts_nothing(make):
    limiter = make()
    for _ in range(3):
        limiter.admit_demo("v1", 10)
    for _ in range(5):
        code_of(lambda: limiter.admit_demo("v1", 10))
    status = limiter.status("v1")
    assert status.used == 3 and not status.paused          # refusals didn't use the demo-wide request budget
    assert limiter.requests_in_last_minute() == 3


# ---------- demo-wide daily caps ----------

def test_daily_request_cap_pauses_the_demo_for_everyone(make):
    limiter = make()
    for i in range(5):
        limiter.admit_demo(f"v{i}", 10)
    error = code_of(lambda: limiter.admit_demo("new-visitor", 10))
    assert (error.code, error.reason, error.retry_after) == ("demo_paused", "daily_requests", SECONDS_TO_MIDNIGHT)
    status = limiter.status("new-visitor")
    assert (status.paused, status.paused_reason, status.used, status.remaining) == (True, "daily_requests", 0, 3)


def test_paused_wins_over_the_visitor_cap(make):
    limiter = make(DemoSettings(visitor_daily_requests=2, daily_requests=2, daily_tokens=10_000,
                                daily_model_calls=100, per_minute_requests=100))
    limiter.admit_demo("v1", 1)
    limiter.admit_demo("v1", 1)
    assert code_of(lambda: limiter.admit_demo("v1", 1)).code == "demo_paused"


def test_tokens_are_reserved_at_admission(make):
    limiter = make()
    limiter.admit_demo("v1", 600)
    error = code_of(lambda: limiter.admit_demo("v2", 600))        # 600 reserved + 600 > 1,000
    assert (error.code, error.reason) == ("demo_paused", "daily_tokens")
    assert limiter.admit_demo("v2", 400).reserved_tokens == 400   # 600 + 400 = 1,000 still fits


def test_settling_replaces_the_reservation_with_the_actual_tokens(make):
    limiter = make()
    first = limiter.admit_demo("v1", 600)
    limiter.settle(first, tokens=100, calls=1)                   # it only used 100
    limiter.admit_demo("v2", 600)                                 # 100 + 600 <= 1,000
    error = code_of(lambda: limiter.admit_demo("v3", 400))       # 100 + 600 + 400 > 1,000
    assert error.reason == "daily_tokens"


def test_settling_more_than_reserved_counts_the_actual(make):
    limiter = make()
    first = limiter.admit_demo("v1", 100)
    limiter.settle(first, tokens=950, calls=1)
    error = code_of(lambda: limiter.admit_demo("v2", 100))       # 950 + 100 > 1,000
    assert error.reason == "daily_tokens"
    assert limiter.admit_demo("v2", 50).reserved_tokens == 50


def test_used_up_tokens_show_as_paused(make):
    limiter = make()
    limiter.settle(limiter.admit_demo("v1", 10), tokens=1_000, calls=1)
    status = limiter.status("v2")
    assert (status.paused, status.paused_reason) == (True, "daily_tokens")


def test_model_calls_are_counted_at_settlement(make):
    limiter = make()
    limiter.settle(limiter.admit_demo("v1", 10), tokens=10, calls=6)    # e.g. retries and fallbacks
    limiter.settle(limiter.admit_demo("v2", 10), tokens=10, calls=4)    # 10 calls: the budget is used up
    error = code_of(lambda: limiter.admit_demo("v3", 10))
    assert (error.code, error.reason) == ("demo_paused", "daily_model_calls")
    assert limiter.status("v3").paused_reason == "daily_model_calls"


def test_an_unsettled_request_keeps_its_reservation(make):
    limiter = make()
    limiter.admit_demo("v1", 900)                                  # still running (or crashed before settle)
    assert code_of(lambda: limiter.admit_demo("v2", 200)).reason == "daily_tokens"


# ---------- per-minute window ----------

def test_per_minute_window_says_busy_and_when_to_retry(make, clock):
    limiter = make(DemoSettings(visitor_daily_requests=10, daily_requests=100, daily_tokens=100_000,
                                daily_model_calls=100, per_minute_requests=2))
    limiter.admit_demo("v1", 1)
    clock.advance(seconds=20)
    limiter.admit_demo("v2", 1)
    clock.advance(seconds=10)                                       # the oldest is 30 s old
    error = code_of(lambda: limiter.admit_demo("v3", 1))
    assert error.code == "demo_busy" and 1 <= error.retry_after <= 30
    assert limiter.requests_in_last_minute() == 2
    assert limiter.status("v3").used == 0                           # busy didn't use the visitor's quota
    clock.advance(seconds=31)                                       # the oldest has left the window
    assert limiter.requests_in_last_minute() == 1
    assert limiter.admit_demo("v3", 1).remaining == 9


def test_busy_is_checked_after_the_daily_caps(make):
    limiter = make(DemoSettings(visitor_daily_requests=1, daily_requests=100, daily_tokens=100_000,
                                daily_model_calls=100, per_minute_requests=1))
    limiter.admit_demo("v1", 1)
    assert code_of(lambda: limiter.admit_demo("v1", 1)).code == "demo_limit_reached"
    assert code_of(lambda: limiter.admit_demo("v2", 1)).code == "demo_busy"


# ---------- day rollover ----------

def test_a_new_utc_day_resets_every_daily_count(make, clock):
    clock.now = datetime(2026, 11, 13, 23, 59, 30, tzinfo=timezone.utc)
    limiter = make()
    for i in range(3):
        limiter.admit_demo("v1", 10)
    limiter.admit_demo("v2", 10)
    limiter.admit_demo("v3", 10)                                    # 5: demo-wide cap reached
    assert code_of(lambda: limiter.admit_demo("v4", 10)).retry_after == 30
    clock.advance(seconds=31)                                       # 00:00:01 on the 14th
    a = limiter.admit_demo("v1", 10)
    assert (a.day, a.remaining) == ("2026-11-14", 2)
    status = limiter.status("v4")
    assert (status.paused, status.used) == (False, 0)
    assert status.resets_at == datetime(2026, 11, 15, tzinfo=timezone.utc)


def test_settling_after_midnight_counts_against_the_admission_day(make, clock):
    clock.now = datetime(2026, 11, 13, 23, 59, 50, tzinfo=timezone.utc)
    limiter = make()
    yesterday = limiter.admit_demo("v1", 10)
    clock.advance(seconds=20)
    limiter.settle(yesterday, tokens=1_000, calls=10)                # would use up a whole day's budget
    assert limiter.status("v2").paused is False                     # ...but it belongs to the 13th
    assert limiter.admit_demo("v2", 900).day == "2026-11-14"


# ---------- status ----------

def test_status_for_a_new_visitor(make):
    status = make().status("v1")
    assert isinstance(status, DemoStatus)
    assert (status.paused, status.paused_reason, status.used, status.remaining, status.limit) == \
        (False, None, 0, 3, 3)
    assert status.resets_at == datetime(2026, 11, 14, tzinfo=timezone.utc)


def test_status_counts_nothing(make):
    limiter = make()
    limiter.admit_demo("v1", 10)
    for _ in range(10):
        limiter.status("v1")
    status = limiter.status("v1")
    assert (status.used, status.remaining) == (1, 2)
    assert limiter.requests_in_last_minute() == 1


# ---------- public BYOK ----------

def test_public_byok_has_its_own_visitor_cap(make):
    limiter = make()
    a = limiter.admit_public_byok("v1")
    assert (a.kind, a.reserved_tokens, a.remaining, a.limit) == ("byok", 0, 1, 2)
    limiter.admit_public_byok("v1")
    error = code_of(lambda: limiter.admit_public_byok("v1"))
    assert error.code == "public_limit_reached" and error.retry_after == SECONDS_TO_MIDNIGHT


def test_public_byok_and_demo_are_counted_apart(make):
    limiter = make()
    for _ in range(3):
        limiter.admit_demo("v1", 10)
    assert limiter.admit_public_byok("v1").remaining == 1           # demo used up, BYOK still available
    limiter.admit_public_byok("v1")
    assert limiter.status("v1").used == 3


def test_public_byok_touches_no_demo_budget_or_window(make):
    limiter = make(DemoSettings(visitor_daily_requests=10, daily_requests=1, daily_tokens=10,
                                daily_model_calls=1, per_minute_requests=1, public_byok_visitor_daily_requests=50))
    for i in range(20):
        limiter.settle(limiter.admit_public_byok(f"v{i}"), tokens=10_000, calls=99)   # settle is a no-op
    assert limiter.requests_in_last_minute() == 0
    assert limiter.status("v0").paused is False
    assert limiter.admit_demo("v0", 5).remaining == 9


# ---------- concurrency ----------

def test_concurrent_admissions_never_exceed_the_visitor_cap(make):
    limiter = make(DemoSettings(visitor_daily_requests=10, daily_requests=1_000, daily_tokens=1_000_000,
                                daily_model_calls=1_000, per_minute_requests=1_000))
    admitted, refused = [], []

    def go():
        try:
            admitted.append(limiter.admit_demo("v1", 1))
        except DemoLimitError as error:
            refused.append(error.code)

    threads = [threading.Thread(target=go) for _ in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(admitted) == 10 and refused == ["demo_limit_reached"] * 20
    assert sorted(a.remaining for a in admitted) == list(range(10))


def test_concurrent_admissions_never_exceed_the_token_budget(make):
    limiter = make(DemoSettings(visitor_daily_requests=1_000, daily_requests=1_000, daily_tokens=1_000,
                                daily_model_calls=1_000, per_minute_requests=1_000))
    admitted = []

    def go(i):
        try:
            admitted.append(limiter.admit_demo(f"v{i}", 100))
        except DemoLimitError:
            pass

    threads = [threading.Thread(target=go, args=(i,)) for i in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(admitted) == 10


# ---------- SQLite: restarts and several workers ----------

def test_sqlite_counters_survive_a_restart(tmp_path, clock):
    path = tmp_path / "usage.sqlite3"
    first = DemoLimiter(SMALL, path=path, clock=clock)
    first.admit_demo("v1", 10)
    first.admit_demo("v1", 10)
    first.settle(first.admit_demo("v2", 500), tokens=700, calls=3)
    first.close()
    again = DemoLimiter(SMALL, path=path, clock=clock)
    assert again.status("v1").used == 2
    assert code_of(lambda: again.admit_demo("v3", 400)).reason == "daily_tokens"    # 10 + 10 + 700 + 400 > 1,000
    again.close()


def test_two_workers_on_one_file_share_the_daily_counts(tmp_path, clock):
    path = tmp_path / "usage.sqlite3"
    a = DemoLimiter(SMALL, path=path, clock=clock)
    b = DemoLimiter(SMALL, path=path, clock=clock)
    a.admit_demo("v1", 10)
    b.admit_demo("v1", 10)
    a.admit_demo("v1", 10)
    assert code_of(lambda: b.admit_demo("v1", 10)).code == "demo_limit_reached"
    a.close()
    b.close()


def test_the_sqlite_file_holds_no_raw_ip(tmp_path, clock):
    path = tmp_path / "usage.sqlite3"
    limiter = DemoLimiter(SMALL, path=path, clock=clock)
    vid = visitor_id("203.0.113.77", b"secret")
    limiter.settle(limiter.admit_demo(vid, 10), tokens=10, calls=1)
    limiter.admit_public_byok(vid)
    limiter.close()
    data = b"".join(f.read_bytes() for f in tmp_path.iterdir())
    assert b"203.0.113.77" not in data


# ---------- 7-day cleanup ----------

def test_cleanup_keeps_seven_days_and_deletes_older(make, clock):
    limiter = make()
    for days_ago in (10, 8, 7, 1, 0):
        clock.now = START - timedelta(days=days_ago)
        limiter.admit_demo("v1", 10)
    clock.now = START
    assert limiter.stored_days() == ["2026-11-03", "2026-11-05", "2026-11-06", "2026-11-12", "2026-11-13"]
    removed = limiter.cleanup()
    assert removed >= 2                                              # the 3rd and the 5th (visitor + demo-wide rows)
    assert limiter.stored_days() == ["2026-11-06", "2026-11-12", "2026-11-13"]
    assert limiter.cleanup() == 0


def test_cleanup_runs_when_the_limiter_starts(tmp_path, clock):
    path = tmp_path / "usage.sqlite3"
    clock.now = START - timedelta(days=30)
    old = DemoLimiter(SMALL, path=path, clock=clock)
    old.admit_demo("v1", 10)
    old.close()
    clock.now = START
    fresh = DemoLimiter(SMALL, path=path, clock=clock)
    assert fresh.stored_days() == []
    fresh.close()
    old_day = (START - timedelta(days=30)).date().isoformat()
    with sqlite3.connect(path) as db:                               # really gone from the file, not just hidden
        tables = [t for (t,) in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        for table in tables:                                        # (a metadata table, e.g. a schema version, is fine)
            for row in db.execute(f"SELECT * FROM {table}"):
                assert old_day not in str(row), (table, row)
