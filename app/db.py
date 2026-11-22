"""Triage's database (Phase 5): SQLAlchemy Core on SQLite (local dev, tests) or Postgres (production, Neon).

    DATABASE_URL      e.g. postgresql+psycopg://user:pass@host/db?sslmode=require, or sqlite:///path/to/file
                      (default: sqlite:///data/triage.sqlite3)
    TRIAGE_FAKE_DB    fake-model mode's own SQLite file (default data/fake.sqlite3). Fake mode never uses
                      DATABASE_URL, so fake answers can't reach the real log, even when .env sets it.
    DATABASE_URL_UNPOOLED   optional: a direct (not pooled) Postgres URL, used only by `alembic upgrade head`.
                      Neon recommends running schema changes on a direct connection; DATABASE_URL is used if unset.
    TRIAGE_DB_PREPARED_STATEMENTS   auto (default) / on / off. Off is needed behind a transaction-mode connection
                      pooler (PgBouncer, Neon's "-pooler" host), which can hand each transaction a different server
                      connection, where psycopg's prepared statements don't exist. "auto": off for a "-pooler" host.

Neon scales an idle database to zero after 5 minutes; the next connection wakes it (usually a few hundred ms).
Postgres connections get a CONNECT_TIMEOUT-second connect timeout (unless the URL sets connect_timeout), and the
app's engine retries a failed connect once after CONNECT_RETRY_DELAY seconds, so the first request after a sleep
waits for the wake instead of failing.

The schema is owned by Alembic (migrations/). Opening a database checks it is at the latest revision, cheaply: one
SELECT on alembic_version, compared with migrations/HEAD (the head revision, written when the Docker image is built;
without it, Alembic is asked). Alembic itself is imported only when the database is behind, and then upgrades it in
this process: SQLite on its own connection, Postgres on DATABASE_URL_UNPOOLED (Neon's direct connection) when the
engine is DATABASE_URL's. A database newer than this code is refused. `alembic upgrade head` still works by hand.

Tables here are what the code reads and writes (the request log and labels; the demo counters);
migrations/versions/ is what creates them. A test checks that the two match.
"""
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path
from typing import Optional, Union, get_args

from sqlalchemy import (Boolean, Column, DateTime, Double, Engine, ForeignKey, Index, Integer, LargeBinary,
                        MetaData, Table, Text, create_engine, event, func, inspect, select)
from sqlalchemy.engine import make_url
from sqlalchemy.pool import StaticPool

from app.registry import PROJECT_ROOT

DB_URL_ENV, FAKE_DB_ENV = "DATABASE_URL", "TRIAGE_FAKE_DB"
UNPOOLED_URL_ENV, PREPARED_ENV = "DATABASE_URL_UNPOOLED", "TRIAGE_DB_PREPARED_STATEMENTS"
DEFAULT_DB = PROJECT_ROOT / "data" / "triage.sqlite3"
FAKE_DB = PROJECT_ROOT / "data" / "fake.sqlite3"
MIGRATIONS_DIR = PROJECT_ROOT / "migrations"
SCHEMA_HEAD_FILE = MIGRATIONS_DIR / "HEAD"       # written at image build time (Dockerfile); absent in development
CONNECT_TIMEOUT = 10            # seconds per connection attempt (Neon suggests about 10 for a waking compute)
CONNECT_RETRY_DELAY = 1.0       # seconds before the one retry

logger = logging.getLogger("triage.db")

metadata = MetaData()


def _base_type(t):
    """Optional[X] -> X; X -> X."""
    args = [a for a in get_args(t) if a is not type(None)]
    return args[0] if args else t


_SQL_TYPE = {str: Text, int: Integer, float: Double, bool: Boolean, bytes: LargeBinary, list: Text}   # list: JSON


def _columns(dataclass_type, required=(), foreign_keys=None) -> list:
    foreign_keys = foreign_keys or {}
    return [Column(f.name, _SQL_TYPE[_base_type(f.type)](), *([foreign_keys[f.name]] if f.name in foreign_keys else []),
                   nullable=f.name not in required)
            for f in fields(dataclass_type)]


def _tables():
    from app.api.request_log import LabelRow, RequestRow        # the dataclasses are the column lists
    requests = Table("requests", metadata, Column("id", Integer, primary_key=True, autoincrement=True),
                     *_columns(RequestRow, required=("request_id",), foreign_keys={
                         # deleting a user keeps their request rows, unlinked
                         "user_id": ForeignKey("users.id", name="fk_requests_user_id_users", ondelete="SET NULL")}))
    Index("idx_requests_request_id", requests.c.request_id, unique=True)
    Index("idx_requests_key_id", requests.c.key_id)
    Index("idx_requests_timestamp", requests.c.timestamp)
    Index("idx_requests_user_id", requests.c.user_id)
    # the dashboard's stats queries (migration 0009)
    Index("idx_requests_model_ts", requests.c.requested_model, requests.c.timestamp)
    Index("idx_requests_user_ts", requests.c.user_id, requests.c.timestamp)
    Index("idx_requests_user_id_id", requests.c.user_id, requests.c.id)
    labels = Table("labels", metadata, Column("id", Integer, primary_key=True, autoincrement=True),
                   *_columns(LabelRow))
    Index("idx_labels_request_id", labels.c.request_id)
    Index("idx_labels_created_at", labels.c.created_at)
    return requests, labels


requests_table, labels_table = _tables()

# The DemoLimiter's counters (app/api/demo.py), per UTC day
visitor_day_table = Table("visitor_day", metadata,
                          Column("day", Text, primary_key=True), Column("kind", Text, primary_key=True),
                          Column("visitor", Text, primary_key=True), Column("requests", Integer, nullable=False))
users_table = Table(
    "users", metadata,
    Column("id", Text, primary_key=True),                      # a UUID, as text
    Column("email", Text, nullable=False),                     # normalized (credentials.normalize_email)
    Column("password_hash", Text, nullable=False),             # argon2id
    Column("display_name", Text), Column("preferred_provider", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False))
Index("idx_users_email", users_table.c.email, unique=True)
sessions_table = Table(
    "sessions", metadata,
    Column("id", Text, primary_key=True),
    Column("user_id", Text, ForeignKey("users.id", name="fk_sessions_user_id_users", ondelete="CASCADE"),
           nullable=False),
    Column("token_hash", Text, nullable=False),                # SHA-256 of the token; the token is never stored
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("last_used_at", DateTime(timezone=True), nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=False),  # the absolute expiry
    Column("revoked_at", DateTime(timezone=True)))
Index("idx_sessions_token_hash", sessions_table.c.token_hash, unique=True)
Index("idx_sessions_user_id", sessions_table.c.user_id)
auth_attempts_table = Table(
    "auth_attempts", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("kind", Text, nullable=False), Column("subject", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False))
Index("idx_auth_attempts_lookup", auth_attempts_table.c.kind, auth_attempts_table.c.subject,
      auth_attempts_table.c.created_at)

provider_keys_table = Table(
    "provider_keys", metadata,
    Column("user_id", Text, ForeignKey("users.id", name="fk_provider_keys_user_id_users", ondelete="CASCADE"),
           primary_key=True),
    Column("provider", Text, primary_key=True),
    Column("ciphertext", LargeBinary, nullable=False), Column("nonce", LargeBinary, nullable=False),
    Column("enc_key_id", Text, nullable=False),                # which TRIAGE_ENCRYPTION_KEYS key encrypted it
    Column("last4", Text, nullable=False),
    Column("status", Text, nullable=False), Column("status_detail", Text),
    Column("verified_at", DateTime(timezone=True)),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False))

api_keys_table = Table(
    "api_keys", metadata,
    Column("id", Text, primary_key=True),
    Column("user_id", Text, ForeignKey("users.id", name="fk_api_keys_user_id_users", ondelete="CASCADE"),
           nullable=False),
    Column("name", Text, nullable=False),
    Column("public_id", Text, nullable=False),                 # "trk_" + 8 characters; the full key is never stored
    Column("secret_hash", Text, nullable=False),               # HMAC-SHA256 of the full key (TRIAGE_HASH_SECRET)
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("last_used_at", DateTime(timezone=True)), Column("revoked_at", DateTime(timezone=True)))
Index("idx_api_keys_public_id", api_keys_table.c.public_id, unique=True)
Index("idx_api_keys_user_id", api_keys_table.c.user_id)

account_day_table = Table(
    "account_day", metadata,
    Column("day", Text, primary_key=True),
    Column("user_id", Text, ForeignKey("users.id", name="fk_account_day_user_id_users", ondelete="CASCADE"),
           primary_key=True),
    Column("requests", Integer, nullable=False))

password_reset_codes_table = Table(
    "password_reset_codes", metadata,
    Column("user_id", Text, ForeignKey("users.id", name="fk_password_reset_codes_user_id_users", ondelete="CASCADE"),
           primary_key=True),                                  # one code per user: a new one replaces it
    Column("id", Text, nullable=False),                        # a UUID per code (the email's idempotency key)
    Column("code_hash", Text, nullable=False),                 # HMAC of the code (TRIAGE_HASH_SECRET); never the code
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=False),
    Column("attempts", Integer, nullable=False),               # tries so far, counted before each comparison
    Column("used_at", DateTime(timezone=True)),
    Column("last_sent_at", DateTime(timezone=True), nullable=False))   # the resend cooldown
password_reset_tokens_table = Table(
    "password_reset_tokens", metadata,
    Column("token_hash", Text, primary_key=True),              # SHA-256 of the "rst_" token; never the token
    Column("user_id", Text, ForeignKey("users.id", name="fk_password_reset_tokens_user_id_users",
                                       ondelete="CASCADE"), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=False),
    Column("used_at", DateTime(timezone=True)))
Index("idx_password_reset_tokens_user_id", password_reset_tokens_table.c.user_id, unique=True)

demo_day_table = Table("demo_day", metadata,
                       Column("day", Text, primary_key=True), Column("requests", Integer, nullable=False),
                       Column("tokens", Integer, nullable=False), Column("calls", Integer, nullable=False))


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


def migration_url() -> str:
    """The URL `alembic upgrade head` uses: DATABASE_URL_UNPOOLED if set (Neon's direct connection), else the
    normal one. Same checks as database_url (never the fake-mode file)."""
    return os.environ.get(UNPOOLED_URL_ENV) or database_url()


def is_pooler_host(host: Optional[str]) -> bool:
    """Neon's pooled endpoints are named <endpoint>-pooler.<region>...; anything else is a direct connection"""
    return bool(host) and "-pooler." in host.lower()


def prepared_statements_off(url: str) -> bool:
    """True when psycopg must not prepare statements for this URL (see TRIAGE_DB_PREPARED_STATEMENTS above)"""
    parsed = make_url(url)
    if not parsed.get_backend_name().startswith("postgresql"):
        return False
    setting = (os.environ.get(PREPARED_ENV) or "auto").strip().lower()
    if setting not in {"auto", "on", "off"}:
        raise RuntimeError(f"{PREPARED_ENV} must be auto, on or off, got {setting!r}")
    if setting != "auto":
        return setting == "off"
    return is_pooler_host(parsed.host)


def postgres_connect_args(url: str) -> dict:
    """psycopg connection arguments for a Postgres URL ({} for anything else): connect_timeout (CONNECT_TIMEOUT
    unless the URL sets its own), and prepare_threshold=None (prepared statements off) behind a pooler"""
    parsed = make_url(url)
    if not parsed.get_backend_name().startswith("postgresql"):
        return {}
    args = {} if "connect_timeout" in parsed.query else {"connect_timeout": CONNECT_TIMEOUT}
    if prepared_statements_off(url):
        args["prepare_threshold"] = None
    return args


def connect_with_one_retry(dialect, cargs, cparams, sleep=time.sleep):
    """Open a DB-API connection; on a connection error (e.g. Neon still waking), wait CONNECT_RETRY_DELAY seconds
    and try once more. A second failure raises as usual."""
    try:
        return dialect.connect(*cargs, **cparams)
    except dialect.loaded_dbapi.OperationalError as error:
        logger.warning("database connect failed (%s); retrying once", type(error).__name__)
        sleep(CONNECT_RETRY_DELAY)
        return dialect.connect(*cargs, **cparams)


def _retry_connects(engine: Engine) -> None:
    @event.listens_for(engine, "do_connect")
    def _do_connect(dialect, _record, cargs, cparams):
        return connect_with_one_retry(dialect, cargs, cparams)


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
    foreign keys; Postgres gets pre-ping (Neon suspends idle databases and drops their connections), a connect
    timeout and one connect retry (see the module docstring)."""
    parsed = make_url(url)
    if parsed.get_backend_name() == "sqlite":
        if parsed.database and parsed.database != ":memory:":
            Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)
        memory = not parsed.database or parsed.database == ":memory:"
        # In memory: one connection shared by every thread (each new connection would be a new, empty database)
        engine = create_engine(url, connect_args={"check_same_thread": False, "timeout": 30},
                               **({"poolclass": StaticPool} if memory else {}))
        event.listen(engine, "connect", _sqlite_pragmas)
    else:
        engine = create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5, pool_recycle=300,
                               connect_args=postgres_connect_args(url))
        _retry_connects(engine)
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


def expected_head() -> str:
    """The latest revision this code knows: migrations/HEAD when the image recorded it (no Alembic import), else
    Alembic's own answer (development)."""
    try:
        recorded = SCHEMA_HEAD_FILE.read_text().strip()
    except OSError:
        recorded = ""
    return recorded or head_revision()


def stored_revision(engine: Engine) -> Optional[str]:
    """The revision recorded in the database (alembic_version), with plain SQL; None if it has none yet."""
    if not inspect(engine).has_table("alembic_version"):
        return None
    with engine.connect() as connection:
        row = connection.exec_driver_sql("SELECT version_num FROM alembic_version").fetchone()
    return row[0] if row else None


def migrate_or_check(engine: Engine) -> None:
    """At the latest revision: nothing to do (and Alembic is never imported). Behind: upgrade it here. Ahead of this
    code (e.g. a rollback that skipped the downgrade): refuse. A pre-Alembic SQLite log is refused, with the way to
    import it."""
    legacy = is_legacy_log(engine)
    if legacy is not None:
        raise RuntimeError(f"{engine.url.database} is a request log from before the database layer (schema version "
                           f"{legacy}). Import it with: venv/bin/python scripts/import_sqlite_log.py "
                           f"{engine.url.database}  (it copies the rows into {DB_URL_ENV}; the file is only read)")
    head, found = expected_head(), stored_revision(engine)
    if found == head:
        return
    if found is not None and found not in known_revisions():
        raise RuntimeError(f"refusing to start: the database is at revision {found}, newer than this code (which "
                           f"knows up to {head}). Run the newer code's `alembic downgrade {head}` first.")
    logger.warning("database at revision %s, upgrading to %s", found or "none", head)
    if engine.dialect.name == "sqlite":
        upgrade(engine)
    else:
        url = _migration_url_for(engine)
        migrations = create_engine(url, connect_args=postgres_connect_args(url))
        try:
            upgrade(migrations)
        finally:
            migrations.dispose()
    after = stored_revision(engine)
    if after != head:
        raise RuntimeError(f"refusing to start: the upgrade left the database at {after or 'none'}, not {head}")


def known_revisions() -> set:
    from alembic.script import ScriptDirectory
    return {script.revision for script in ScriptDirectory.from_config(alembic_config()).walk_revisions()}


def _migration_url_for(engine: Engine) -> str:
    """DATABASE_URL_UNPOOLED when this engine is DATABASE_URL's (Neon: schema changes on the direct connection), else
    the engine's own URL (a test database, a URL given in code)."""
    configured = os.environ.get(DB_URL_ENV)
    same = configured and make_url(configured).render_as_string(hide_password=False) == \
        engine.url.render_as_string(hide_password=False)
    return migration_url() if same else engine.url.render_as_string(hide_password=False)


def upgrade(engine: Engine, revision: str = "head") -> None:
    from alembic import command
    with engine.begin() as connection:
        command.upgrade(alembic_config(connection), revision)


# ---------- transactions that run alone ----------

@contextmanager
def exclusive_transaction(engine: Engine, lock_key: int):
    """A transaction that runs alone among those using the same lock_key, across threads and instances, so a
    check-then-write inside it can't race. SQLite: BEGIN IMMEDIATE (one writer per file). Postgres: a
    transaction-level advisory lock on lock_key, released at commit or rollback. Commits on success, rolls
    back on any exception."""
    if engine.dialect.name == "sqlite":
        # The driver would otherwise start the transaction only at the first write, after the reads
        with engine.connect() as connection:
            connection = connection.execution_options(isolation_level="AUTOCOMMIT")
            connection.exec_driver_sql("BEGIN IMMEDIATE")
            try:
                yield connection
                connection.exec_driver_sql("COMMIT")
            except BaseException:
                connection.exec_driver_sql("ROLLBACK")
                raise
        return
    with engine.begin() as connection:
        if engine.dialect.name == "postgresql":
            connection.execute(select(func.pg_advisory_xact_lock(lock_key)))
        yield connection


def upsert(engine: Engine, table):
    """INSERT for this engine's dialect, which has on_conflict_do_update / on_conflict_do_nothing."""
    if engine.dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        from sqlalchemy.dialects.sqlite import insert
    return insert(table)


def aware(value):
    """A datetime read back from the database as an aware UTC datetime (SQLite returns naive ones)."""
    from datetime import timezone
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=timezone.utc)
