"""Shared test setup.

app/providers/base.py calls load_dotenv() at import, so a developer's .env (e.g. TRIAGE_PUBLIC=on for live
testing) would otherwise leak into every test. The deployment switches below are cleared for each test; a test
about one of them sets it itself with monkeypatch.
"""
import pytest

DEPLOYMENT_ENV = ("TRIAGE_PUBLIC", "TRIAGE_ACCOUNTS", "TRIAGE_CORS_ORIGINS", "TRIAGE_SERVER_PROVIDERS", "TRIAGE_TRUSTED_PROXY_HOPS",
                  "DATABASE_URL", "TRIAGE_FAKE_DB")


@pytest.fixture(autouse=True)
def _no_deployment_env(monkeypatch):
    for name in DEPLOYMENT_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _no_real_database(monkeypatch, tmp_path):
    """The default database files (app/db.py) point into the test's tmp_path, so no test can create or write
    data/triage.sqlite3 or data/fake.sqlite3, even one that forgets to set DATABASE_URL / TRIAGE_FAKE_DB."""
    from app import db
    monkeypatch.setattr(db, "DEFAULT_DB", tmp_path / "default-db" / "triage.sqlite3")
    monkeypatch.setattr(db, "FAKE_DB", tmp_path / "default-db" / "fake.sqlite3")


def pytest_configure(config):
    config.addinivalue_line("markers", "postgres: needs TRIAGE_TEST_DATABASE_URL (a Postgres test database)")


def pytest_collection_modifyitems(config, items):
    """Postgres tests use the network, so they run only when asked for (pytest -m postgres), even if .env sets
    TRIAGE_TEST_DATABASE_URL: a plain `pytest` never leaves the machine."""
    if "postgres" in (config.getoption("markexpr") or ""):
        return
    skip = pytest.mark.skip(reason="Postgres test: run with -m postgres")
    for item in items:
        if "postgres" in item.keywords:
            item.add_marker(skip)
