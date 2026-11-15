"""Shared test setup.

app/providers/base.py calls load_dotenv() at import, so a developer's .env (e.g. TRIAGE_PUBLIC=on for live
testing) would otherwise leak into every test. The deployment switches below are cleared for each test; a test
about one of them sets it itself with monkeypatch.
"""
import pytest

DEPLOYMENT_ENV = ("TRIAGE_PUBLIC", "TRIAGE_CORS_ORIGINS", "TRIAGE_SERVER_PROVIDERS", "TRIAGE_TRUSTED_PROXY_HOPS",
                  "TRIAGE_DEMO_DB", "DATABASE_URL", "TRIAGE_FAKE_DB")


@pytest.fixture(autouse=True)
def _no_deployment_env(monkeypatch):
    for name in DEPLOYMENT_ENV:
        monkeypatch.delenv(name, raising=False)


def pytest_configure(config):
    config.addinivalue_line("markers", "postgres: needs TRIAGE_TEST_DATABASE_URL (a Postgres test database)")
