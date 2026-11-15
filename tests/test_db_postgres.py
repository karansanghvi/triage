"""The database layer on Postgres. Skipped unless TRIAGE_TEST_DATABASE_URL is set, e.g. a Neon test branch:

    TRIAGE_TEST_DATABASE_URL='postgresql+psycopg://user:pass@host/db?sslmode=require' venv/bin/python -m pytest -m postgres

Each test works in its own new schema, dropped afterwards, so the database's existing tables are never touched.
Neon's pooled endpoint rejects the search_path startup option, so a "-pooler" host is switched to its direct
host here (Neon's naming: ep-x-pooler.region... -> ep-x.region...). A plain `pytest` skips these (conftest.py).
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

from sqlalchemy.engine import make_url


def _direct(url):
    """A Neon pooled URL as its direct (unpooled) one; any other URL unchanged."""
    if not url:
        return url
    parsed = make_url(url)
    if parsed.host and "-pooler." in parsed.host:
        parsed = parsed.set(host=parsed.host.replace("-pooler.", ".", 1))
    return parsed.render_as_string(hide_password=False)


from dotenv import load_dotenv                                       # noqa: E402

from app.registry import PROJECT_ROOT                                # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")             # the same whether or not another module loaded it first
URL = _direct(os.environ.get("TRIAGE_TEST_DATABASE_URL"))
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
