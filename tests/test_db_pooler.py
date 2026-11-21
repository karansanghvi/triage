"""Prepared statements behind Neon's connection pooler, and the migration URL."""
import os

import pytest
from sqlalchemy import create_engine, text

from app import db

POOLED = "postgresql+psycopg://u:p@ep-cool-name-123456-pooler.us-west-2.aws.neon.tech/neondb?sslmode=require"
DIRECT = "postgresql+psycopg://u:p@ep-cool-name-123456.us-west-2.aws.neon.tech/neondb?sslmode=require"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (db.PREPARED_ENV, db.UNPOOLED_URL_ENV):
        monkeypatch.delenv(name, raising=False)


def test_a_pooler_host_turns_prepared_statements_off():
    assert db.is_pooler_host("ep-cool-name-123456-pooler.us-west-2.aws.neon.tech")
    assert db.postgres_connect_args(POOLED) == {"prepare_threshold": None}


def test_a_direct_host_keeps_them():
    assert not db.is_pooler_host("ep-cool-name-123456.us-west-2.aws.neon.tech")
    assert db.postgres_connect_args(DIRECT) == {}


def test_the_setting_overrides_detection(monkeypatch):
    monkeypatch.setenv(db.PREPARED_ENV, "off")
    assert db.postgres_connect_args(DIRECT) == {"prepare_threshold": None}
    monkeypatch.setenv(db.PREPARED_ENV, "on")
    assert db.postgres_connect_args(POOLED) == {}
    monkeypatch.setenv(db.PREPARED_ENV, "sometimes")
    with pytest.raises(RuntimeError, match="auto, on or off"):
        db.postgres_connect_args(POOLED)


def test_sqlite_is_never_affected(monkeypatch):
    monkeypatch.setenv(db.PREPARED_ENV, "off")
    assert db.postgres_connect_args("sqlite:///x.sqlite3") == {}


def test_migrations_use_the_unpooled_url_when_set(monkeypatch):
    monkeypatch.setenv(db.DB_URL_ENV, POOLED)
    assert db.migration_url() == POOLED
    monkeypatch.setenv(db.UNPOOLED_URL_ENV, DIRECT)
    assert db.migration_url() == DIRECT


def test_alembic_env_uses_the_migration_url_and_connect_args():
    source = (db.MIGRATIONS_DIR / "env.py").read_text()
    assert "migration_url()" in source and "postgres_connect_args(url)" in source


# ---------- on a real pooled endpoint (opt-in, like tests/test_db_postgres.py) ----------

def _pooled(url):
    """The test database's pooled endpoint: a Neon direct host gets "-pooler" after its endpoint id"""
    from sqlalchemy.engine import make_url
    parsed = make_url(url)
    host = parsed.host or ""
    if "-pooler." not in host and host.startswith("ep-") and "." in host:
        first, rest = host.split(".", 1)
        parsed = parsed.set(host=f"{first}-pooler.{rest}")
    return parsed.render_as_string(hide_password=False)


@pytest.mark.postgres
def test_repeated_queries_work_through_the_pooler():
    from dotenv import load_dotenv
    from app.registry import PROJECT_ROOT
    load_dotenv(PROJECT_ROOT / ".env")
    url = os.environ.get("TRIAGE_TEST_DATABASE_URL")
    if not url:
        pytest.skip("TRIAGE_TEST_DATABASE_URL not set")
    url = _pooled(url)
    if not db.is_pooler_host(db.make_url(url).host):
        pytest.skip("the test database has no Neon pooled endpoint")
    engine = create_engine(url, pool_size=2, connect_args=db.postgres_connect_args(url))
    try:
        # psycopg prepares a statement after 5 runs by default; through a transaction pooler that breaks
        for i in range(12):
            with engine.begin() as connection:
                assert connection.execute(text("SELECT :n + 1"), {"n": i}).scalar() == i + 1
                assert connection.connection.dbapi_connection.prepare_threshold is None
    finally:
        engine.dispose()
