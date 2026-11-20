"""Housekeeping for the privacy policy (app/api/maintenance.py): expired cache answers are deleted, request log rows
older than the retention period are deleted in batches (with their labels), and nothing in the job can crash the
server. Plus the documented start command has --no-access-log (no raw IPs in our logs)."""
import asyncio
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import db
from app.api import maintenance
from app.api.auth import Auth
from app.api.main import create_app
from app.api.request_log import InMemoryRequestLog, LabelRow, RequestRow, SQLRequestLog
from app.api.stats import summarize
from app.cache import ResponseCache
from app.providers.base import Response
from app.registry import PROJECT_ROOT
from tests.api_helpers import API_KEY, make_deps

NOW = datetime(2026, 11, 19, 12, 0, tzinfo=timezone.utc)


def at(days_ago: float) -> str:
    return (NOW - timedelta(days=days_ago)).isoformat()


def row(request_id: str, days_ago: float) -> RequestRow:
    return RequestRow(request_id=request_id, timestamp=at(days_ago), key_id="k", requested_model="auto",
                      stream=False, status="ok")


def label(request_id: str) -> LabelRow:
    return LabelRow(request_id=request_id, created_at=NOW.isoformat(), status="labeled")


def answer(text="ok") -> Response:
    return Response(text=text, model_name="groq-small", model_id="m", provider="groq", input_tokens=1,
                    output_tokens=1, latency_ms=1.0, cost=0.0, reported_cost=0.0)


# ---------- 1. the cache: expired answers only ----------

class Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


def test_purge_removes_expired_answers_only(tmp_path):
    clock = Clock(1_000_000.0)
    cache = ResponseCache(tmp_path / "c.sqlite3", clock=clock)
    cache.set("old", answer("old"))
    clock.now += 23 * 3600
    cache.set("new", answer("new"))
    clock.now += 1 * 3600 + 1                       # "old" is now 24 h + 1 s old, "new" is 1 h old
    assert cache.purge_expired() == 1
    assert cache.count() == 1 and cache.get("new").text == "new" and cache.get("old") is None
    assert cache.purge_expired() == 0               # nothing left to remove


def test_an_expired_answer_is_gone_from_the_file_not_just_hidden(tmp_path):
    import sqlite3
    clock = Clock(1_000_000.0)
    cache = ResponseCache(tmp_path / "c.sqlite3", clock=clock)
    cache.set("k", answer("secret answer text"))
    clock.now += 24 * 3600 + 1
    maintenance.run_once(SimpleNamespace(cached_sender=SimpleNamespace(cache=cache), log=InMemoryRequestLog()), 365,
                         clock=lambda: NOW)
    raw = sqlite3.connect(tmp_path / "c.sqlite3").execute("SELECT response_json FROM responses").fetchall()
    assert raw == []


# ---------- 3. request log retention ----------

def test_retention_deletes_only_rows_older_than_the_cutoff_in_memory():
    log = InMemoryRequestLog()
    for r in (row("ancient", 800), row("old", 366), row("edge", 365), row("new", 10)):
        log.write(r)
    for request_id in ("old", "new"):
        log.write_label(label(request_id))
    result = maintenance.run_once(SimpleNamespace(log=log), 365, clock=lambda: NOW)
    assert result["request_rows_deleted"] == 2
    assert [r.request_id for r in log.rows()] == ["edge", "new"]        # exactly 365 days old is kept
    assert [l.request_id for l in log.labels()] == ["new"]              # the old row's label went with it
    assert log.get("old") is None and log.get("new") is not None


@pytest.fixture
def sql_log(tmp_path):
    engine = db.open_engine(db.sqlite_url(tmp_path / "t.sqlite3"))
    yield SQLRequestLog(engine)
    engine.dispose()


def test_retention_deletes_old_rows_in_batches_in_the_database(sql_log, monkeypatch):
    for i in range(2500):
        sql_log.write(row(f"old-{i}", 400 + i / 1000))
    for i in range(30):
        sql_log.write(row(f"new-{i}", 30))
    for request_id in ("old-1", "old-2499", "new-0"):
        sql_log.write_label(label(request_id))
    transactions = []
    real_begin = sql_log.engine.begin

    def counting_begin(*args, **kwargs):
        transactions.append(1)
        return real_begin(*args, **kwargs)

    monkeypatch.setattr(sql_log.engine, "begin", counting_begin)
    deleted = sql_log.delete_older_than(maintenance.retention_cutoff(365, NOW), batch_size=1000)
    assert deleted == 2500
    assert len(transactions) == 3                     # 1000 + 1000 + 500: three short transactions, not one
    left = sql_log.rows(include_vectors=False)
    assert {r.request_id for r in left} == {f"new-{i}" for i in range(30)}
    assert [l.request_id for l in sql_log.labels()] == ["new-0"]


def test_retention_with_nothing_old_changes_nothing(sql_log):
    sql_log.write(row("new", 1))
    assert sql_log.delete_older_than(maintenance.retention_cutoff(365, NOW)) == 0
    assert [r.request_id for r in sql_log.rows()] == ["new"]


def test_the_retention_period_is_configurable(monkeypatch):
    log = InMemoryRequestLog()
    for r in (row("a", 40), row("b", 20)):
        log.write(r)
    monkeypatch.setenv(maintenance.RETENTION_ENV, "30")
    days = maintenance.retention_days_from_env()
    maintenance.run_once(SimpleNamespace(log=log), days, clock=lambda: NOW)
    assert days == 30 and [r.request_id for r in log.rows()] == ["b"]


@pytest.mark.parametrize("value, days", [(None, 365), ("", 365), ("  ", 365), ("30", 30), ("730", 730)])
def test_retention_days_from_env(value, days):
    environ = {} if value is None else {maintenance.RETENTION_ENV: value}
    assert maintenance.retention_days_from_env(environ) == days


@pytest.mark.parametrize("value", ["0", "-5", "abc", "1.5", "365d"])
def test_a_bad_retention_value_refuses_to_start(value, monkeypatch):
    with pytest.raises(RuntimeError, match="TRIAGE_REQUEST_LOG_RETENTION_DAYS"):
        maintenance.retention_days_from_env({maintenance.RETENTION_ENV: value})
    monkeypatch.setenv(maintenance.RETENTION_ENV, value)
    with pytest.raises(RuntimeError):
        with TestClient(create_app(deps=make_deps(), auth=Auth([API_KEY]))):
            pass


def test_stats_work_with_no_rows_or_only_recent_ones():
    empty = summarize([])
    assert empty["requests"] == 0 and empty["saved_share"] is None and empty["tier_mix"] == {}
    assert summarize([row("new", 1)])["requests"] == 1


# ---------- the job never crashes the server ----------

class Broken:
    def purge_expired(self):
        raise OSError("disk I/O error")

    def delete_older_than(self, cutoff, batch_size=1000):
        raise ConnectionError("database unreachable")


def test_a_failing_cache_purge_doesnt_stop_retention(caplog):
    log = InMemoryRequestLog()
    log.write(row("old", 400))
    result = maintenance.run_once(SimpleNamespace(cached_sender=SimpleNamespace(cache=Broken()), log=log), 365,
                                  clock=lambda: NOW)
    assert result == {"cache_expired_deleted": "error:OSError", "request_rows_deleted": 1}
    assert "cache purge failed (OSError)" in caplog.text and "disk I/O error" not in caplog.text


def test_a_failing_retention_step_is_reported_not_raised():
    result = maintenance.run_once(SimpleNamespace(cached_sender=None, log=Broken()), 365, clock=lambda: NOW)
    assert result == {"request_rows_deleted": "error:ConnectionError"}


def test_without_a_cache_only_retention_runs():
    result = maintenance.run_once(SimpleNamespace(cached_sender=None, log=InMemoryRequestLog()), 365,
                                  clock=lambda: NOW)
    assert result == {"request_rows_deleted": 0}


def test_the_hourly_loop_survives_a_pass_that_raises(monkeypatch):
    calls, sleeps = [], []

    def flaky(deps, days):
        calls.append(days)
        if len(calls) == 2:
            raise RuntimeError("unexpected")
        return {}

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(maintenance, "run_once", flaky)
    asyncio.run(maintenance.run_forever(object(), 365, sleep=fake_sleep, passes=3))
    assert calls == [365, 365, 365] and sleeps == [3600, 3600]        # kept going after the failure


def test_the_server_runs_it_at_startup_and_stops_it_at_shutdown(tmp_path):
    deps = make_deps(cache=ResponseCache(tmp_path / "c.sqlite3", clock=lambda: 0.0))
    deps.log.write(row("very-old", 9999))
    deps.log.write(RequestRow(request_id="today", timestamp=datetime.now(timezone.utc).isoformat(), key_id="k",
                              requested_model="auto", stream=False, status="ok"))
    app = create_app(deps=deps, auth=Auth([API_KEY]))
    with TestClient(app) as http:
        for _ in range(100):                        # the startup pass runs in a worker thread
            if deps.log.get("very-old") is None:
                break
            asyncio.run(asyncio.sleep(0.02))
        assert deps.log.get("very-old") is None and deps.log.get("today") is not None
        assert http.get("/v1/models", headers={"Authorization": f"Bearer {API_KEY}"}).status_code == 200
    # leaving the block ran shutdown: the task was cancelled without an error


def test_a_broken_log_at_startup_doesnt_stop_the_server():
    deps = make_deps()
    deps.log.delete_older_than = Broken().delete_older_than
    with TestClient(create_app(deps=deps, auth=Auth([API_KEY]))) as http:
        assert http.get("/v1/models", headers={"Authorization": f"Bearer {API_KEY}"}).status_code == 200


# ---------- 2. no raw IPs in our logs ----------

def test_every_documented_start_command_turns_off_the_access_log():
    readme = (PROJECT_ROOT / "README.md").read_text()
    commands = re.findall(r"uvicorn (?:app\.api\.main:app|\.\.\.)[^`\n]*", readme)
    starts = [c for c in commands if "--port" in c or "..." in c]
    assert len(starts) >= 3 and all("--no-access-log" in c for c in starts), starts
    for name in ("Procfile", "render.yaml"):           # any deploy config added later must do the same
        path = Path(PROJECT_ROOT, name)
        if path.exists():
            assert "--no-access-log" in path.read_text(), name
