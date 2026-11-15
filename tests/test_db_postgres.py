"""The database layer on Postgres. Skipped unless TRIAGE_TEST_DATABASE_URL is set, e.g. a Neon test branch:

    TRIAGE_TEST_DATABASE_URL='postgresql+psycopg://user:pass@host/db?sslmode=require' venv/bin/python -m pytest -m postgres

Each test works in its own new schema, dropped afterwards, so the database's existing tables are never touched.
Use Neon's direct (not pooled) connection string: the pooler doesn't pass the search_path option through.
"""
import os
import uuid
from dataclasses import fields

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, text

from app import db
from app.api.request_log import RequestRow

URL = os.environ.get("TRIAGE_TEST_DATABASE_URL")
pytestmark = [pytest.mark.postgres, pytest.mark.skipif(not URL, reason="TRIAGE_TEST_DATABASE_URL not set")]


@pytest.fixture
def schema_url():
    """A fresh schema, as the default search_path of every connection made with the returned URL."""
    name = f"triage_test_{uuid.uuid4().hex[:12]}"
    admin = create_engine(URL)
    with admin.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{name}"'))
    separator = "&" if "?" in URL else "?"
    yield f"{URL}{separator}options=-csearch_path%3D{name}"
    with admin.begin() as connection:
        connection.execute(text(f'DROP SCHEMA "{name}" CASCADE'))
    admin.dispose()


def test_postgres_is_refused_until_migrated_then_matches_the_tables(schema_url):
    raw = create_engine(schema_url)
    with pytest.raises(RuntimeError, match="alembic upgrade head"):
        db.migrate_or_check(raw)
    db.upgrade(raw)
    db.migrate_or_check(raw)                                        # now at head: accepted
    with raw.connect() as connection:
        assert compare_metadata(MigrationContext.configure(connection), db.metadata) == []
        cols = [r[0] for r in connection.execute(text(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'requests' "
            "AND table_schema = current_schema() ORDER BY ordinal_position"))]
    assert cols == ["id"] + [f.name for f in fields(RequestRow)]
    raw.dispose()
