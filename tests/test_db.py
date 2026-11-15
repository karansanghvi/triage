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
    assert db.current_revision(engine) == db.head_revision() == "0002"
    raw = sqlite3.connect(tmp_path / "t.sqlite3")
    tables = {r[0] for r in raw.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"requests", "labels", "visitor_day", "demo_day", "alembic_version"} <= tables
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


def test_a_database_behind_the_code_is_refused_unless_sqlite(tmp_path, monkeypatch):
    engine = engine_at(tmp_path)
    monkeypatch.setattr(db, "head_revision", lambda: "9999")
    monkeypatch.setattr(engine.dialect, "name", "postgresql")
    with pytest.raises(RuntimeError, match="revision 0002, this code needs 9999.*alembic upgrade head"):
        db.migrate_or_check(engine)
    engine.dispose()


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
    assert db.current_revision(again) == "0002"
    raw = sqlite3.connect(tmp_path / "t.sqlite3")
    assert {"visitor_day", "demo_day"} <= {r[0] for r in raw.execute("SELECT name FROM sqlite_master")}
    again.dispose()
