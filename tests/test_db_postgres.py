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


# ---------- saved provider keys on Postgres ----------

import secrets as _secrets                                           # noqa: E402

from app.accounts.encryption import KeyRing                          # noqa: E402
from app.accounts.verify_keys import Verdict                         # noqa: E402

SECRET_KEY = "sk-test-SECRET123"


@pytest.fixture
def pg_keys_app(migrated_url):
    engine = db.open_engine(migrated_url)
    calls = []
    deps = make_deps(engine=engine, log=SQLRequestLog(engine), trusted_proxy_hops=1,
                     keyring=KeyRing([("k1", _secrets.token_bytes(32))]),
                     key_verifier=lambda p, k: calls.append((p, k)) or Verdict("valid", None, True))
    http = TestClient(create_app(deps=deps, auth=Auth([API_KEY]), accounts=True))
    token = http.post("/v1/auth/signup", json={"email": "ada@example.com", "password": PASSWORD},
                      headers={"X-Forwarded-For": "203.0.113.9"}).json()["session_token"]
    yield http, deps, bearer(token), calls
    engine.dispose()


def test_provider_keys_lifecycle_on_postgres(pg_keys_app):
    http, deps, auth, calls = pg_keys_app
    r = http.put("/v1/account/provider-keys/anthropic", json={"key": SECRET_KEY}, headers=auth)
    assert r.status_code == 200 and r.json()["last4"] == "T123" and r.json()["status_detail"] is None
    with deps.engine.connect() as connection:
        row = connection.execute(text("SELECT ciphertext, nonce, enc_key_id FROM provider_keys")).one()
    assert SECRET_KEY.encode() not in bytes(row.ciphertext) and len(bytes(row.nonce)) == 12 and row.enc_key_id == "k1"
    assert http.post("/v1/account/provider-keys/anthropic/verify", headers=auth).json()["status"] == "valid"
    assert calls[-1] == ("anthropic", SECRET_KEY)                                   # bytea round trip decrypts
    assert [k["provider"] for k in http.get("/v1/account/provider-keys", headers=auth).json()] == ["anthropic"]
    assert http.delete("/v1/account/provider-keys/anthropic", headers=auth).status_code == 204
    assert http.get("/v1/account/provider-keys", headers=auth).json() == []


def test_deleting_the_account_cascades_to_provider_keys_on_postgres(pg_keys_app):
    http, deps, auth, _ = pg_keys_app
    http.put("/v1/account/provider-keys/groq", json={"key": "gsk_testkey_eeee"}, headers=auth)
    assert http.request("DELETE", "/v1/account", json={"password": PASSWORD}, headers=auth).status_code == 204
    with deps.engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM provider_keys")).scalar() == 0


def test_provider_key_rate_limit_on_postgres(pg_keys_app):
    http, _, auth, calls = pg_keys_app
    for i in range(20):
        assert http.put("/v1/account/provider-keys/openai", json={"key": f"sk-test-KEY-{i:04d}"},
                        headers=auth).status_code == 200
    assert http.put("/v1/account/provider-keys/openai", json={"key": SECRET_KEY}, headers=auth).status_code == 429
    assert len(calls) == 20


def test_status_detail_round_trips_on_postgres(migrated_url):
    engine = db.open_engine(migrated_url)
    deps = make_deps(engine=engine, log=SQLRequestLog(engine), trusted_proxy_hops=1,
                     keyring=KeyRing([("k1", _secrets.token_bytes(32))]),
                     key_verifier=lambda p, k: Verdict("unverified", "Groq answered 503: try later", False, 503))
    http = TestClient(create_app(deps=deps, auth=Auth([API_KEY]), accounts=True))
    token = http.post("/v1/auth/signup", json={"email": "ada@example.com", "password": PASSWORD},
                      headers={"X-Forwarded-For": "203.0.113.9"}).json()["session_token"]
    http.put("/v1/account/provider-keys/groq", json={"key": "gsk_testkey_ffff"}, headers=bearer(token))
    (saved,) = http.get("/v1/account/provider-keys", headers=bearer(token)).json()
    assert (saved["status"], saved["status_detail"]) == ("unverified", "Groq answered 503: try later")
    engine.dispose()


# ---------- account API keys on Postgres ----------

from app.accounts.api_keys import ApiKeyStore, TooManyKeys           # noqa: E402


def test_api_key_lifecycle_on_postgres(pg_app):
    http, deps, _ = pg_app
    auth = bearer(http.post("/v1/auth/signup", json={"email": "ada@example.com", "password": PASSWORD},
                            headers={"X-Forwarded-For": "203.0.113.5"}).json()["session_token"])
    created = http.post("/v1/account/api-keys", json={"name": "laptop"}, headers=auth).json()
    key = created["key"]
    with deps.engine.connect() as connection:
        row = connection.execute(text("SELECT public_id, secret_hash FROM api_keys")).one()
    assert row.public_id == key[:12] and key not in row.secret_hash
    assert http.get("/v1/models", headers=bearer(key)).status_code == 200          # an account caller now
    assert http.get("/v1/routing-config", headers=bearer(key)).status_code == 403   # never an admin
    assert http.get("/v1/account/api-keys", headers=auth).json()[0]["last_used_at"] is not None
    assert http.delete(f"/v1/account/api-keys/{created['id']}", headers=auth).status_code == 204
    assert http.get("/v1/models", headers=bearer(key)).status_code == 401
    assert http.request("DELETE", "/v1/account", json={"password": PASSWORD}, headers=auth).status_code == 204
    with deps.engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM api_keys")).scalar() == 0


def test_the_active_key_cap_holds_across_instances_on_postgres(migrated_url):
    engine = db.open_engine(migrated_url)
    user = AccountStore(engine).create_user("ada@example.com", "$argon2id$fake", None)
    stores = [ApiKeyStore(db.open_engine(migrated_url), b"secret") for _ in range(3)]

    def make(store):
        try:
            store.create(user.id, "k")
            return "made"
        except TooManyKeys:
            return "refused"

    results, _ = run_together([lambda s=stores[i % 3]: make(s) for i in range(26)])
    assert results.count("made") == 20 and results.count("refused") == 6
    for store in stores:
        store.engine.dispose()
    engine.dispose()


# ---------- account mode on Postgres ----------

from app.accounts.limits import AccountLimiter, AccountLimitError, AccountLimits   # noqa: E402
from app.api.fake_models import FakeBehavior                                       # noqa: E402
from tests.api_helpers import FakePredictor                                        # noqa: E402

ASK_PG = {"model": "auto", "messages": [{"role": "user", "content": "What is 2 + 3?"}]}


@pytest.fixture
def pg_account_app(migrated_url):
    engine = db.open_engine(migrated_url)
    deps = make_deps({"claude-haiku": FakeBehavior(fail_before="auth")}, engine=engine, log=SQLRequestLog(engine),
                     predictor=FakePredictor(0.95, 0.95), trusted_proxy_hops=1,
                     keyring=KeyRing([("k1", _secrets.token_bytes(32))]),
                     key_verifier=lambda p, k: Verdict("valid", None, True, 200))
    http = TestClient(create_app(deps=deps, auth=Auth([API_KEY]), accounts=True))
    yield http, deps
    engine.dispose()


def pg_user(http, email, n):
    r = http.post("/v1/auth/signup", json={"email": email, "password": PASSWORD},
                  headers={"X-Forwarded-For": f"203.0.113.{n}"}).json()
    return bearer(r["session_token"]), r["user"]["id"]


def test_account_chat_isolation_and_rejected_saved_keys_on_postgres(pg_account_app):
    http, deps = pg_account_app
    ada, ada_id = pg_user(http, "ada@example.com", 1)
    bob, _ = pg_user(http, "bob@example.com", 2)
    for auth in (ada, bob):
        http.put("/v1/account/provider-keys/groq", json={"key": "gsk_saved_SECRET_1111"}, headers=auth)
    http.put("/v1/account/provider-keys/anthropic", json={"key": "sk-ant-saved-SECRET-2222"}, headers=ada)
    key = http.post("/v1/account/api-keys", json={"name": "x"}, headers=ada).json()["key"]
    r = http.post("/v1/chat/completions", json=ASK_PG, headers=bearer(key))
    assert r.status_code == 200 and r.headers["x-triage-mode"] == "account" and deps.fake_models.last_api_key.startswith("gsk_saved")
    ada_request = r.headers["x-triage-request-id"]
    bob_request = http.post("/v1/chat/completions", json=ASK_PG, headers=bob).headers["x-triage-request-id"]
    assert http.get(f"/v1/requests/{ada_request}", headers=ada).json()["user_id"] == ada_id
    assert http.get(f"/v1/requests/{bob_request}", headers=bearer(key)).status_code == 404
    assert http.get("/v1/stats", headers=ada).json()["requests"] == 1
    r = http.post("/v1/chat/completions", json=ASK_PG | {"model": "claude-haiku"}, headers=ada)
    assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_provider_key"
    statuses = {k["provider"]: k["status"] for k in http.get("/v1/account/provider-keys", headers=ada).json()}
    assert statuses == {"anthropic": "invalid", "groq": "valid"}
    rows = {row.request_id: row for row in deps.log.rows()}
    assert rows[ada_request].mode == "account" and rows[ada_request].training_eligible is False
    assert all("SECRET" not in str(row.to_dict()) for row in rows.values())


def test_the_daily_account_limit_holds_across_instances_on_postgres(migrated_url):
    engine = db.open_engine(migrated_url)
    user = AccountStore(engine).create_user("ada@example.com", "$argon2id$fake", None)
    limiters = [AccountLimiter(AccountLimits(requests_per_minute=1_000, requests_per_day=10),
                               db.open_engine(migrated_url)) for _ in range(3)]

    def admit(limiter):
        try:
            limiter.admit(user.id)
            return "admitted"
        except AccountLimitError:
            return "refused"

    results, _ = run_together([lambda l=limiters[i % 3]: admit(l) for i in range(25)])
    assert results.count("admitted") == 10 and results.count("refused") == 15
    for limiter in limiters:
        limiter.engine.dispose()
    engine.dispose()
