"""The database layer (app/db.py, migrations/): engines, schema and Alembic. SQLite only, no network.
Postgres runs the same checks in test_db_postgres.py when TRIAGE_TEST_DATABASE_URL is set."""
import sqlite3
from dataclasses import fields

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext

from app import db
from app.api.request_log import LabelRow, RequestRow


def engine_at(tmp_path, name="t.sqlite3"):
    return db.open_engine(db.sqlite_url(tmp_path / name))


def test_opening_a_new_sqlite_file_creates_the_schema_at_the_latest_revision(tmp_path):
    engine = engine_at(tmp_path)
    assert db.current_revision(engine) == db.head_revision() == "0010"
    raw = sqlite3.connect(tmp_path / "t.sqlite3")
    tables = {r[0] for r in raw.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"requests", "labels", "visitor_day", "demo_day", "users", "sessions", "auth_attempts",
            "provider_keys", "api_keys", "account_day", "password_reset_codes",
            "password_reset_tokens", "alembic_version"} <= tables
    engine.dispose()


def test_the_columns_are_the_dataclass_fields(tmp_path):
    engine = engine_at(tmp_path)
    raw = sqlite3.connect(tmp_path / "t.sqlite3")
    for table, dataclass_type in (("requests", RequestRow), ("labels", LabelRow)):
        cols = [c[1] for c in raw.execute(f"PRAGMA table_info({table})")]
        assert cols == ["id"] + [f.name for f in fields(dataclass_type)]
    assert "user_id" in [f.name for f in fields(RequestRow)]
    engine.dispose()


def test_the_migrations_match_the_tables_the_code_uses(tmp_path):
    """A column added to RequestRow without a migration (or the reverse) fails here."""
    engine = engine_at(tmp_path)
    with engine.connect() as connection:
        assert compare_metadata(MigrationContext.configure(connection), db.metadata) == []
    engine.dispose()


def test_indexes(tmp_path):
    engine = engine_at(tmp_path)
    raw = sqlite3.connect(tmp_path / "t.sqlite3")
    indexes = {r[1]: r[2] for r in raw.execute("PRAGMA index_list(requests)")}
    assert indexes["idx_requests_request_id"] == 1                 # unique
    assert {"idx_requests_key_id", "idx_requests_timestamp", "idx_requests_user_id"} <= set(indexes)
    columns = {name: [c[2] for c in raw.execute(f"PRAGMA index_info({name})")]
               for name in ("idx_requests_model_ts", "idx_requests_user_ts", "idx_requests_user_id_id")}
    assert columns == {"idx_requests_model_ts": ["requested_model", "timestamp"],
                       "idx_requests_user_ts": ["user_id", "timestamp"], "idx_requests_user_id_id": ["user_id", "id"]}
    engine.dispose()


def test_sqlite_connections_use_wal_and_foreign_keys(tmp_path):
    engine = engine_at(tmp_path)
    with engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA journal_mode").scalar() == "wal"
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
        assert connection.exec_driver_sql("PRAGMA busy_timeout").scalar() == 30000
    engine.dispose()


def test_reopening_is_a_no_op(tmp_path):
    db.open_engine(db.sqlite_url(tmp_path / "t.sqlite3")).dispose()
    engine = engine_at(tmp_path)
    assert db.current_revision(engine) == db.head_revision()
    engine.dispose()


def test_a_pre_alembic_log_is_refused_with_the_import_command(tmp_path):
    path = tmp_path / "old.sqlite3"
    raw = sqlite3.connect(path)
    raw.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
    raw.execute("INSERT INTO schema_version VALUES (4)")
    raw.execute("CREATE TABLE requests (id INTEGER PRIMARY KEY, request_id TEXT)")
    raw.commit()
    with pytest.raises(RuntimeError, match=r"schema version 4.*import_sqlite_log\.py"):
        db.open_engine(db.sqlite_url(path))
    assert [r[0] for r in raw.execute("SELECT name FROM sqlite_master WHERE type = 'table'")] == \
        ["schema_version", "requests"]                              # nothing was changed


def test_a_database_newer_than_the_code_is_refused(tmp_path, monkeypatch):
    engine = engine_at(tmp_path)
    with engine.begin() as connection:
        connection.exec_driver_sql("UPDATE alembic_version SET version_num = '9999'")
    with pytest.raises(RuntimeError, match="revision 9999, newer than this code.*alembic downgrade 0010"):
        db.migrate_or_check(engine)
    engine.dispose()


def test_the_head_comes_from_the_recorded_file_when_the_image_wrote_one(tmp_path, monkeypatch):
    recorded = tmp_path / "HEAD"
    monkeypatch.setattr(db, "SCHEMA_HEAD_FILE", recorded)
    assert db.expected_head() == db.head_revision()                  # no file (development): Alembic answers
    recorded.write_text("0010\n")
    monkeypatch.setattr(db, "head_revision", lambda: pytest.fail("Alembic was asked despite the recorded head"))
    assert db.expected_head() == "0010"


def test_stored_revision_reads_alembic_version_with_plain_sql(tmp_path):
    engine = engine_at(tmp_path)
    assert db.stored_revision(engine) == "0010"
    engine.dispose()
    empty = db.create_engine(db.sqlite_url(tmp_path / "empty.sqlite3"))
    assert db.stored_revision(empty) is None
    empty.dispose()


def _run(code: str):
    import subprocess
    import sys
    from app.registry import PROJECT_ROOT
    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip()


def test_at_head_startup_never_imports_alembic(tmp_path):
    """The fast path (every start once deployed): one SELECT, no Alembic import, nothing written."""
    path = tmp_path / "at_head.sqlite3"
    engine_at(tmp_path, "at_head.sqlite3").dispose()                 # created and upgraded to head
    (tmp_path / "HEAD").write_text(db.head_revision() + "\n")
    out = _run(f"""
import sys
from app import db
db.SCHEMA_HEAD_FILE = __import__("pathlib").Path({str(tmp_path / "HEAD")!r})
engine = db.open_engine(db.sqlite_url({str(path)!r}))
print("alembic" in sys.modules, db.stored_revision(engine))
""")
    assert out == "False 0010"


def test_a_database_behind_is_upgraded_in_process(tmp_path):
    """Behind (a release with a new migration): Alembic is imported and the database upgraded before serving."""
    path = tmp_path / "behind.sqlite3"
    engine = db.create_engine(db.sqlite_url(path))
    db.upgrade(engine, "0009")
    assert db.stored_revision(engine) == "0009"
    engine.dispose()
    (tmp_path / "HEAD").write_text("0010\n")
    out = _run(f"""
import sys
from app import db
db.SCHEMA_HEAD_FILE = __import__("pathlib").Path({str(tmp_path / "HEAD")!r})
engine = db.open_engine(db.sqlite_url({str(path)!r}))
print("alembic" in sys.modules, db.stored_revision(engine))
""")
    assert out == "True 0010"


def test_postgres_migrates_on_the_direct_url_when_the_engine_is_database_urls(monkeypatch):
    pooled = "postgresql+psycopg://u:p@ep-x-pooler.us-west-2.aws.neon.tech/neondb?sslmode=require"
    direct = "postgresql+psycopg://u:p@ep-x.us-west-2.aws.neon.tech/neondb?sslmode=require"
    other = "postgresql+psycopg://u:p@localhost/test"
    monkeypatch.setenv(db.DB_URL_ENV, pooled)
    monkeypatch.setenv(db.UNPOOLED_URL_ENV, direct)
    assert db._migration_url_for(db.create_engine(pooled)) == direct
    assert db._migration_url_for(db.create_engine(other)) == other   # a test database: its own URL


# ---------- which database ----------

def test_the_default_is_a_local_sqlite_file():
    assert db.database_url() == db.sqlite_url(db.DEFAULT_DB)


def test_database_url_is_used_in_real_mode(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@db.example/triage")
    assert db.database_url() == "postgresql+psycopg://u:p@db.example/triage"


def test_fake_mode_never_uses_database_url(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@db.example/triage")
    assert db.database_url(fake=True) == db.sqlite_url(db.FAKE_DB)
    monkeypatch.setenv("TRIAGE_FAKE_DB", str(tmp_path / "f.sqlite3"))
    assert db.database_url(fake=True) == db.sqlite_url(tmp_path / "f.sqlite3")


def test_real_mode_refuses_the_fake_database(monkeypatch, tmp_path):
    monkeypatch.setenv("TRIAGE_FAKE_DB", str(tmp_path / "f.sqlite3"))
    monkeypatch.setenv("DATABASE_URL", db.sqlite_url(tmp_path / "f.sqlite3"))
    with pytest.raises(RuntimeError, match="fake-mode database"):
        db.database_url()


def test_an_in_memory_database_is_one_database_for_every_thread():
    import threading
    engine = db.open_engine("sqlite://")
    seen = []

    def look():
        with engine.connect() as connection:
            seen.append(connection.exec_driver_sql("SELECT count(*) FROM demo_day").scalar())

    thread = threading.Thread(target=look)
    thread.start()
    thread.join()
    assert seen == [0]                                              # the migrated tables, seen from another thread
    engine.dispose()


def test_a_revision_0001_database_is_upgraded_to_add_the_demo_tables(tmp_path):
    engine = db.open_engine(db.sqlite_url(tmp_path / "t.sqlite3"))
    with engine.begin() as connection:
        from alembic import command
        command.downgrade(db.alembic_config(connection), "0001")
    assert db.current_revision(engine) == "0001"
    engine.dispose()
    again = db.open_engine(db.sqlite_url(tmp_path / "t.sqlite3"))
    assert db.current_revision(again) == db.head_revision()
    raw = sqlite3.connect(tmp_path / "t.sqlite3")
    assert {"visitor_day", "demo_day"} <= {r[0] for r in raw.execute("SELECT name FROM sqlite_master")}
    again.dispose()


# ---------- users, sessions and the request log's user_id (0003) ----------

def _user(connection, user_id="u1", email="a@example.com"):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    connection.execute(db.users_table.insert(), dict(id=user_id, email=email, password_hash="h", created_at=now,
                                                     updated_at=now))
    connection.execute(db.sessions_table.insert(), dict(id=f"s-{user_id}", user_id=user_id, token_hash=f"t-{user_id}",
                                                        created_at=now, last_used_at=now, expires_at=now))


def test_deleting_a_user_keeps_their_request_rows_unlinked_and_drops_their_sessions(tmp_path):
    from app.api.request_log import RequestRow, SQLRequestLog
    engine = engine_at(tmp_path)
    with engine.begin() as connection:
        _user(connection, "u1", "a@example.com")
        _user(connection, "u2", "b@example.com")
    log = SQLRequestLog(engine)
    for i, user in enumerate(("u1", "u2", None)):
        log.write(RequestRow(request_id=f"r{i}", timestamp="t", key_id="k", requested_model="auto", stream=False,
                             user_id=user))
    with engine.begin() as connection:
        connection.execute(db.users_table.delete().where(db.users_table.c.id == "u1"))
    assert [(r.request_id, r.user_id) for r in log.rows()] == [("r0", None), ("r1", "u2"), ("r2", None)]
    with engine.connect() as connection:
        assert [r.user_id for r in connection.execute(db.sessions_table.select())] == ["u2"]
    engine.dispose()


def test_a_request_row_cant_name_a_user_that_doesnt_exist(tmp_path):
    from sqlalchemy.exc import IntegrityError
    from app.api.request_log import RequestRow, SQLRequestLog
    engine = engine_at(tmp_path)
    with pytest.raises(IntegrityError):
        SQLRequestLog(engine).write(RequestRow(request_id="r0", timestamp="t", key_id="k", requested_model="auto",
                                               stream=False, user_id="nobody"))
    engine.dispose()


def test_emails_and_token_hashes_are_unique(tmp_path):
    from sqlalchemy.exc import IntegrityError
    engine = engine_at(tmp_path)
    with engine.begin() as connection:
        _user(connection, "u1", "a@example.com")
    with pytest.raises(IntegrityError), engine.begin() as connection:
        _user(connection, "u2", "a@example.com")
    engine.dispose()


def test_upgrading_from_0002_keeps_every_request_row(tmp_path):
    from alembic import command
    from app.api.request_log import RequestRow, SQLRequestLog
    engine = engine_at(tmp_path)
    with engine.begin() as connection:
        command.downgrade(db.alembic_config(connection), "0002")
    rows = [RequestRow(request_id=f"r{i}", timestamp="t", key_id="k", requested_model="auto", stream=False,
                       p_low=0.1 * i, attempts=[{"model": "m"}]) for i in range(5)]
    # Written as at 0002: only the columns that table had (routed_by came in 0008), as the code of that time did
    from sqlalchemy import MetaData, Table
    from app.api.request_log import _to_sql
    old = Table("requests", MetaData(), autoload_with=engine)
    with engine.begin() as connection:
        for r in rows:
            connection.execute(old.insert(), {c.name: _to_sql(c.name, getattr(r, c.name)) for c in old.columns
                                              if c.name != "id"})
    engine.dispose()
    again = engine_at(tmp_path)                                    # upgraded on open: the table is rebuilt
    assert db.current_revision(again) == db.head_revision() and SQLRequestLog(again).rows() == rows
    raw = sqlite3.connect(tmp_path / "t.sqlite3")
    indexes = {r[1] for r in raw.execute("PRAGMA index_list(requests)")}
    assert {"idx_requests_request_id", "idx_requests_key_id", "idx_requests_timestamp", "idx_requests_user_id"} <= indexes
    again.dispose()


def test_0010_turns_old_client_disconnected_errors_into_stopped(tmp_path):
    from alembic import command
    from app.api.request_log import SQLRequestLog
    engine = engine_at(tmp_path)
    with engine.begin() as connection:
        command.downgrade(db.alembic_config(connection), "0009")
    log = SQLRequestLog(engine)
    for rid, status, kind in (("stop", "error", "client_disconnected"), ("fail", "error", "provider_timeout"),
                              ("fine", "ok", None)):
        log.write(RequestRow(request_id=rid, timestamp="2026-11-01T00:00:00+00:00", key_id="k", requested_model="auto",
                             stream=True, status=status, error_kind=kind))
    db.upgrade(engine)
    assert {r.request_id: r.status for r in log.rows()} == {"stop": "stopped", "fail": "error", "fine": "ok"}
    with engine.begin() as connection:
        command.downgrade(db.alembic_config(connection), "0009")
    assert {r.request_id: r.status for r in log.rows()}["stop"] == "error"
    engine.dispose()


def test_alembic_upgrade_head_still_works_by_hand(tmp_path):
    import os
    import subprocess
    import sys
    from app.registry import PROJECT_ROOT
    path = tmp_path / "by_hand.sqlite3"
    env = {**os.environ, db.DB_URL_ENV: db.sqlite_url(path)}
    env.pop(db.UNPOOLED_URL_ENV, None)
    result = subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=PROJECT_ROOT, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    engine = db.create_engine(db.sqlite_url(path))
    assert db.stored_revision(engine) == db.head_revision()
    engine.dispose()
