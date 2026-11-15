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


# ---------- the DemoLimiter on Postgres: no limit is ever overshot ----------

import threading                                                     # noqa: E402
from datetime import datetime, timedelta, timezone                   # noqa: E402

from app.api.demo import DemoLimiter, DemoLimitError, DemoSettings   # noqa: E402

START = datetime(2026, 11, 13, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def instances(schema_url):
    """Three limiters, each with its own engine (its own connection pool and in-process lock), like three
    server instances on one database: only the database's own locking keeps them from overshooting."""
    raw = create_engine(schema_url)
    db.upgrade(raw)
    raw.dispose()
    made = []

    def build(settings, n=3, clock=lambda: START):
        for _ in range(n):
            made.append(DemoLimiter(settings, clock=clock, engine=db.open_engine(schema_url)))
        return made[-n:]

    yield build
    for limiter in made:
        limiter.engine.dispose()


def run_together(calls):
    """Start every call at once (a barrier), wait for all; returns (results, refusal codes)."""
    barrier = threading.Barrier(len(calls))
    results, refused, lock = [], [], threading.Lock()

    def go(call):
        barrier.wait()
        try:
            out = call()
            with lock:
                results.append(out)
        except DemoLimitError as error:
            with lock:
                refused.append(error.code)

    threads = [threading.Thread(target=go, args=(c,)) for c in calls]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results, refused


def test_many_admissions_for_one_visitor_never_exceed_its_daily_limit(instances):
    limiters = instances(DemoSettings(visitor_daily_requests=10, daily_requests=1_000, daily_tokens=1_000_000,
                                      daily_model_calls=1_000, per_minute_requests=1_000))
    results, refused = run_together([lambda l=limiters[i % 3]: l.admit_demo("v1", 1) for i in range(30)])
    assert len(results) == 10 and refused == ["demo_limit_reached"] * 20
    assert sorted(a.remaining for a in results) == list(range(10))
    assert limiters[0].status("v1").used == 10


def test_many_admissions_never_exceed_the_token_budget(instances):
    limiters = instances(DemoSettings(visitor_daily_requests=1_000, daily_requests=1_000, daily_tokens=1_000,
                                      daily_model_calls=1_000, per_minute_requests=1_000))
    results, refused = run_together([lambda l=limiters[i % 3], i=i: l.admit_demo(f"v{i}", 100) for i in range(30)])
    assert len(results) == 10 and set(refused) == {"demo_paused"} and len(refused) == 20
    with limiters[0].engine.connect() as connection:
        assert connection.execute(text("SELECT tokens, requests FROM demo_day")).one() == (1_000, 10)


def test_many_admissions_never_exceed_the_daily_request_total(instances):
    limiters = instances(DemoSettings(visitor_daily_requests=1_000, daily_requests=7, daily_tokens=1_000_000,
                                      daily_model_calls=1_000, per_minute_requests=1_000))
    results, _ = run_together([lambda l=limiters[i % 3], i=i: l.admit_demo(f"v{i}", 1) for i in range(24)])
    assert len(results) == 7


def test_public_byok_admissions_never_exceed_the_visitor_cap(instances):
    limiters = instances(DemoSettings(public_byok_visitor_daily_requests=5))
    results, refused = run_together([lambda l=limiters[i % 3]: l.admit_public_byok("v1") for i in range(15)])
    assert len(results) == 5 and refused == ["public_limit_reached"] * 10


def test_settling_concurrently_loses_no_update(instances):
    limiters = instances(DemoSettings(visitor_daily_requests=1_000, daily_requests=1_000, daily_tokens=1_000_000,
                                      daily_model_calls=1_000, per_minute_requests=1_000))
    admissions = [limiters[0].admit_demo(f"v{i}", 100) for i in range(12)]
    run_together([lambda l=limiters[i % 3], a=a: l.settle(a, tokens=40, calls=2) for i, a in enumerate(admissions)])
    with limiters[0].engine.connect() as connection:
        assert connection.execute(text("SELECT tokens, calls FROM demo_day")).one() == (12 * 40, 12 * 2)


def test_settle_applies_to_the_admission_day_and_cleanup_works(instances):
    now = [START.replace(hour=23, minute=59)]
    (limiter,) = instances(DemoSettings(), n=1, clock=lambda: now[0])
    admission = limiter.admit_demo("v1", 500)
    now[0] = START + timedelta(days=1)                                 # the request ends after midnight
    limiter.settle(admission, tokens=200, calls=3)
    with limiter.engine.connect() as connection:
        rows = connection.execute(text("SELECT day, tokens, calls FROM demo_day ORDER BY day")).all()
    assert rows == [("2026-11-13", 200, 3)]
    now[0] = START + timedelta(days=9)
    assert limiter.cleanup() == 2 and limiter.stored_days() == []


# ---------- accounts on Postgres ----------

from fastapi.testclient import TestClient                            # noqa: E402

from app.accounts.ratelimit import AuthRateLimiter, RateLimited      # noqa: E402
from app.accounts.store import AccountStore                          # noqa: E402
from app.api.auth import Auth                                        # noqa: E402
from app.api.main import create_app                                  # noqa: E402
from app.api.request_log import RequestRow, SQLRequestLog            # noqa: E402
from tests.api_helpers import API_KEY, make_deps                     # noqa: E402

PASSWORD = "correct horse battery"


@pytest.fixture
def migrated_url(schema_url):
    raw = create_engine(schema_url)
    db.upgrade(raw)
    raw.dispose()
    return schema_url


@pytest.fixture
def pg_app(migrated_url):
    engine = db.open_engine(migrated_url)
    log = SQLRequestLog(engine)
    deps = make_deps(engine=engine, log=log, trusted_proxy_hops=1)
    yield TestClient(create_app(deps=deps, auth=Auth([API_KEY]), accounts=True)), deps, log
    engine.dispose()


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def test_the_whole_account_lifecycle_on_postgres(pg_app):
    http, deps, log = pg_app
    ip = {"X-Forwarded-For": "203.0.113.5"}
    r = http.post("/v1/auth/signup", json={"email": " Ada@Example.com ", "password": PASSWORD, "display_name": "Ada"},
                  headers=ip)
    assert r.status_code == 201
    token, user_id = r.json()["session_token"], r.json()["user"]["id"]
    assert http.post("/v1/auth/signup", json={"email": "ada@example.com", "password": PASSWORD},
                     headers=ip).status_code == 409
    assert http.get("/v1/auth/me", headers=bearer(token)).json()["email"] == "ada@example.com"
    other = http.post("/v1/auth/login", json={"email": "ada@example.com", "password": PASSWORD},
                      headers=ip).json()["session_token"]
    assert http.patch("/v1/account", json={"preferred_provider": "gemini"},
                      headers=bearer(token)).json()["preferred_provider"] == "gemini"
    assert http.post("/v1/account/password", json={"current_password": PASSWORD, "new_password": "a brand new secret"},
                     headers=bearer(token)).status_code == 204
    assert http.get("/v1/auth/me", headers=bearer(other)).status_code == 401
    log.write(RequestRow(request_id="r0", timestamp="t", key_id="k", requested_model="auto", stream=False,
                         user_id=user_id))
    assert http.request("DELETE", "/v1/account", json={"password": "a brand new secret"},
                        headers=bearer(token)).status_code == 204
    assert [(row.request_id, row.user_id) for row in log.rows()] == [("r0", None)]     # ON DELETE SET NULL
    assert http.get("/v1/auth/me", headers=bearer(token)).status_code == 401
    with deps.engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM sessions")).scalar() == 0


def test_concurrent_signups_with_one_email_create_one_account(pg_app):
    http, deps, _ = pg_app
    results, _ = run_together([lambda n=n: http.post(
        "/v1/auth/signup", json={"email": "race@example.com", "password": PASSWORD},
        headers={"X-Forwarded-For": f"203.0.113.{n}"}).status_code for n in range(8)])
    assert sorted(results) == [201] + [409] * 7


def test_session_expiry_with_postgres_timestamps(migrated_url):
    engine = db.open_engine(migrated_url)
    now = [datetime.now(timezone.utc)]
    store = AccountStore(engine, clock=lambda: now[0])
    user = store.create_user("ada@example.com", "$argon2id$fake", None)
    token, session = store.create_session(user.id)
    assert store.session_for_token(token)[0].expires_at == session.expires_at          # tz survives the round trip
    now[0] += timedelta(days=7, seconds=1)
    assert store.session_for_token(token) is None
    engine.dispose()


def test_concurrent_login_attempts_never_overshoot_the_ip_limit(migrated_url):
    limiters = [AuthRateLimiter(db.open_engine(migrated_url)) for _ in range(3)]

    def attempt(limiter):
        limiter.hit("login_ip", "visitor-x")
        return True

    calls = [lambda l=limiters[i % 3]: attempt(l) for i in range(25)]
    barrier = threading.Barrier(len(calls))
    allowed, refused, lock = [], [], threading.Lock()

    def go(call):
        barrier.wait()
        try:
            call()
            with lock:
                allowed.append(1)
        except RateLimited:
            with lock:
                refused.append(1)

    threads = [threading.Thread(target=go, args=(c,)) for c in calls]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(allowed) == 10 and len(refused) == 15
    for limiter in limiters:
        limiter.engine.dispose()


def test_wrong_current_passwords_hit_the_per_email_limit_on_postgres(pg_app):
    http, _, _ = pg_app
    token = http.post("/v1/auth/signup", json={"email": "ada@example.com", "password": PASSWORD},
                      headers={"X-Forwarded-For": "203.0.113.5"}).json()["session_token"]
    for _ in range(5):
        assert http.post("/v1/account/password", json={"current_password": "wrong password here",
                                                        "new_password": "a brand new secret"},
                         headers=bearer(token)).status_code == 403
    r = http.request("DELETE", "/v1/account", json={"password": PASSWORD}, headers=bearer(token))
    assert r.status_code == 429 and int(r.headers["retry-after"]) > 0
