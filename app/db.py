"""Triage's database (Phase 5): SQLAlchemy Core on SQLite (local dev, tests) or Postgres (production, Neon).

    DATABASE_URL      e.g. postgresql+psycopg://user:pass@host/db?sslmode=require, or sqlite:///path/to/file
                      (default: sqlite:///data/triage.sqlite3)
    TRIAGE_FAKE_DB    fake-model mode's own SQLite file (default data/fake.sqlite3). Fake mode never uses
                      DATABASE_URL, so fake answers can't reach the real log, even when .env sets it.

The schema is owned by Alembic (migrations/). A SQLite database is upgraded to the latest revision when it is
opened (local files and tests need no extra step); any other database must already be at the latest revision
(run `alembic upgrade head` as a deploy step) or Triage refuses to start.

Tables here are what the code reads and writes; migrations/versions/ is what creates them. A test checks that
the two match.
"""
import os
from dataclasses import fields
from pathlib import Path
from typing import Optional, Union, get_args

from sqlalchemy import (Boolean, Column, Double, Engine, Index, Integer, LargeBinary, MetaData, Table, Text,
                        create_engine, event, inspect)
from sqlalchemy.engine import make_url

from app.registry import PROJECT_ROOT

DB_URL_ENV, FAKE_DB_ENV = "DATABASE_URL", "TRIAGE_FAKE_DB"
DEFAULT_DB = PROJECT_ROOT / "data" / "triage.sqlite3"
FAKE_DB = PROJECT_ROOT / "data" / "fake.sqlite3"
MIGRATIONS_DIR = PROJECT_ROOT / "migrations"

metadata = MetaData()


def _base_type(t):
    """Optional[X] -> X; X -> X."""
    args = [a for a in get_args(t) if a is not type(None)]
    return args[0] if args else t


_SQL_TYPE = {str: Text, int: Integer, float: Double, bool: Boolean, bytes: LargeBinary, list: Text}   # list: JSON


def _columns(dataclass_type, required=()) -> list:
    return [Column(f.name, _SQL_TYPE[_base_type(f.type)](), nullable=f.name not in required)
            for f in fields(dataclass_type)]


def _tables():
    from app.api.request_log import LabelRow, RequestRow        # the dataclasses are the column lists
    requests = Table("requests", metadata, Column("id", Integer, primary_key=True, autoincrement=True),
                     *_columns(RequestRow, required=("request_id",)))
    Index("idx_requests_request_id", requests.c.request_id, unique=True)
    Index("idx_requests_key_id", requests.c.key_id)
    Index("idx_requests_timestamp", requests.c.timestamp)
    Index("idx_requests_user_id", requests.c.user_id)
    labels = Table("labels", metadata, Column("id", Integer, primary_key=True, autoincrement=True),
                   *_columns(LabelRow))
    Index("idx_labels_request_id", labels.c.request_id)
    Index("idx_labels_created_at", labels.c.created_at)
    return requests, labels


requests_table, labels_table = _tables()


# ---------- engines ----------

def database_url(fake: bool = False) -> str:
    """Fake mode: TRIAGE_FAKE_DB or data/fake.sqlite3, never DATABASE_URL. Real mode: DATABASE_URL or
    data/triage.sqlite3, refusing the fake file."""
    if fake:
        return sqlite_url(os.environ.get(FAKE_DB_ENV) or FAKE_DB)
    url = os.environ.get(DB_URL_ENV) or sqlite_url(DEFAULT_DB)
    fake_path = Path(os.environ.get(FAKE_DB_ENV) or FAKE_DB).resolve()
    parsed = make_url(url)
    if parsed.get_backend_name() == "sqlite" and parsed.database and Path(parsed.database).resolve() == fake_path:
        raise RuntimeError(f"refusing to start: real-model mode can't use the fake-mode database {fake_path.name} "
                           f"({DB_URL_ENV}); fake and real requests must be logged apart")
    return url


def sqlite_url(path: Union[str, Path]) -> str:
    return f"sqlite:///{Path(path)}"


def _sqlite_pragmas(dbapi_connection, _record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")       # readers don't block the writer; a crash keeps every commit
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def open_engine(url: str) -> Engine:
    """An engine for `url`, schema checked (see migrate_or_check). SQLite files get WAL, a busy timeout and
    foreign keys; Postgres gets pre-ping, since Neon suspends idle databases and drops their connections."""
    parsed = make_url(url)
    if parsed.get_backend_name() == "sqlite":
        if parsed.database and parsed.database != ":memory:":
            Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(url, connect_args={"check_same_thread": False, "timeout": 30})
        event.listen(engine, "connect", _sqlite_pragmas)
    else:
        engine = create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5, pool_recycle=300)
    try:
        migrate_or_check(engine)
    except Exception:
        engine.dispose()
        raise
    return engine


# ---------- migrations ----------

def alembic_config(connection=None):
    from alembic.config import Config
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    if connection is not None:
        config.attributes["connection"] = connection
    return config


def head_revision() -> str:
    from alembic.script import ScriptDirectory
    return ScriptDirectory.from_config(alembic_config()).get_current_head()


def current_revision(engine: Engine) -> Optional[str]:
    from alembic.runtime.migration import MigrationContext
    with engine.connect() as connection:
        return MigrationContext.configure(connection).get_current_revision()


def is_legacy_log(engine: Engine) -> Optional[int]:
    """A request log written before Alembic (its own schema_version table): its version, else None."""
    names = inspect(engine).get_table_names()
    if "schema_version" not in names or "alembic_version" in names:
        return None
    with engine.connect() as connection:
        found = connection.exec_driver_sql("SELECT version FROM schema_version").fetchone()
    return found[0] if found else 0


def migrate_or_check(engine: Engine) -> None:
    """SQLite: upgrade to the latest revision. Others: refuse unless already there. A pre-Alembic SQLite log is
    refused either way, with the way to import it."""
    legacy = is_legacy_log(engine)
    if legacy is not None:
        raise RuntimeError(f"{engine.url.database} is a request log from before the database layer (schema version "
                           f"{legacy}). Import it with: venv/bin/python scripts/import_sqlite_log.py "
                           f"{engine.url.database}  (it copies the rows into {DB_URL_ENV}; the file is only read)")
    if engine.dialect.name == "sqlite":
        upgrade(engine)
        return
    found, head = current_revision(engine), head_revision()
    if found != head:
        raise RuntimeError(f"refusing to start: the database is at revision {found or 'none'}, this code needs "
                           f"{head}. Run `alembic upgrade head` with {DB_URL_ENV} set (a deploy step).")


def upgrade(engine: Engine, revision: str = "head") -> None:
    from alembic import command
    with engine.begin() as connection:
        command.upgrade(alembic_config(connection), revision)
