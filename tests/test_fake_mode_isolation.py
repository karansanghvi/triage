"""Fake-model mode (TRIAGE_FAKE_MODELS=1) must never share the real response cache or request log.

Bug this guards against: a real-mode request was served "Fake answer from groq-small..." with X-Triage-Cache: hit,
because fake mode had cached its answers in data/cache/responses.sqlite3 under the same keys real mode reads.
Every path here is redirected to tmp_path, so the tests never touch the real files.
"""
import hashlib

import pytest
from fastapi.testclient import TestClient

from app import db as db_module
from app.api import deps as deps_module
from app.api import request_log
from app.api.auth import Auth
from app.api.fake_models import FakeBehavior, FakeModels
from app.api.main import create_app
from app.cache import ResponseCache
from tests.api_helpers import ADMIN_KEY, API_KEY, FakePredictor

RAIN = [{"role": "user", "content": "Write two sentences about rain."}]
REAL_TEXT = "Real answer from groq-small."


@pytest.fixture
def paths(tmp_path, monkeypatch):
    p = {"real_cache": tmp_path / "cache" / "responses.sqlite3", "fake_cache": tmp_path / "cache" / "fake.sqlite3",
         "real_log": tmp_path / "logs" / "requests.sqlite3", "fake_log": tmp_path / "logs" / "fake.sqlite3"}
    monkeypatch.setattr(deps_module, "DEFAULT_CACHE_PATH", p["real_cache"])
    monkeypatch.setattr(deps_module, "FAKE_CACHE_PATH", p["fake_cache"])
    monkeypatch.setattr(db_module, "DEFAULT_DB", p["real_log"])
    monkeypatch.setattr(db_module, "FAKE_DB", p["fake_log"])
    for name in ("TRIAGE_CACHE", "TRIAGE_CACHE_PATH", "DATABASE_URL", "TRIAGE_FAKE_DB", "TRIAGE_LOG", "TRIAGE_FAKE_MODELS",
                 "TRIAGE_SERVER_PROVIDERS", "TRIAGE_TRUSTED_PROXY_HOPS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TRIAGE_HASH_SECRET", "isolation-test-secret")
    monkeypatch.setattr(deps_module, "load_predictor_checked", lambda: (FakePredictor(), "fake-classifier v1"))
    # "real" mode's provider call: a stand-in that answers with real-looking text (no network)
    monkeypatch.setattr(deps_module, "send_request", FakeModels({"groq-small": FakeBehavior(text=REAL_TEXT)}))
    return p


def ask(deps):
    http = TestClient(create_app(deps=deps, auth=Auth([API_KEY], [ADMIN_KEY])))
    r = http.post("/v1/chat/completions", json={"model": "groq-small", "messages": RAIN},
                  headers={"Authorization": f"Bearer {API_KEY}"})
    assert r.status_code == 200, r.text
    return r


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_real_cache(path):
    """The real cache as it exists in practice: already holding other entries (Phase 2's collector)."""
    cache = ResponseCache(path)
    cache._conn.execute("INSERT INTO responses VALUES ('phase2-key', 'groq-large', '{}', 0, 9e18)")
    cache._conn.commit()
    cache._conn.close()


def test_a_fake_mode_answer_is_never_served_in_real_mode(paths, monkeypatch):
    monkeypatch.setenv("TRIAGE_FAKE_MODELS", "1")
    fake = deps_module.load_deps()
    first = ask(fake)
    assert first.json()["choices"][0]["message"]["content"].startswith("Fake answer from groq-small")
    assert ask(fake).headers["x-triage-cache"] == "hit"                  # fake mode still caches, in its own file
    monkeypatch.delenv("TRIAGE_FAKE_MODELS")
    real = deps_module.load_deps()
    r = ask(real)
    assert r.json()["choices"][0]["message"]["content"] == REAL_TEXT
    assert r.headers["x-triage-cache"] != "hit"


def test_the_real_cache_and_log_are_untouched_by_a_fake_mode_request(paths, monkeypatch):
    seed_real_cache(paths["real_cache"])
    request_log.SQLiteRequestLog(paths["real_log"]).close()
    before = digest(paths["real_cache"]), digest(paths["real_log"])
    monkeypatch.setenv("TRIAGE_FAKE_MODELS", "1")
    fake = deps_module.load_deps()
    ask(fake)
    assert (digest(paths["real_cache"]), digest(paths["real_log"])) == before
    assert fake.cached_sender.cache.path == paths["fake_cache"] and paths["fake_cache"].exists()
    assert fake.log.engine.url.database == str(paths["fake_log"]) and len(fake.log.rows()) == 1


@pytest.mark.parametrize("fake, env, target", [
    (True, "TRIAGE_CACHE_PATH", "real_cache"), (False, "TRIAGE_CACHE_PATH", "fake_cache"),
    (False, "DATABASE_URL", "fake_log")])
def test_each_mode_refuses_the_other_modes_files_even_when_named_explicitly(paths, monkeypatch, fake, env, target):
    if fake:
        monkeypatch.setenv("TRIAGE_FAKE_MODELS", "1")
    value = db_module.sqlite_url(paths[target]) if env == "DATABASE_URL" else str(paths[target])
    monkeypatch.setenv(env, value)
    with pytest.raises(RuntimeError, match="refusing to start"):
        deps_module.load_deps()
    assert not paths[target].exists()


def test_the_cache_can_still_be_turned_off(paths, monkeypatch):
    monkeypatch.setenv("TRIAGE_FAKE_MODELS", "1")
    monkeypatch.setenv("TRIAGE_CACHE", "off")
    assert deps_module.load_deps().cached_sender is None


def test_fake_mode_keeps_its_own_demo_counters(paths, monkeypatch):
    """The demo counters live in the same database as the log, so fake mode's are in its own file too."""
    monkeypatch.setenv("TRIAGE_FAKE_MODELS", "1")
    fake = deps_module.load_deps()
    fake.demo_limiter.admit_demo("visitor-1", 100)
    assert fake.demo_limiter.engine is fake.log.engine
    assert fake.demo_limiter.engine.url.database == str(paths["fake_log"])
    monkeypatch.delenv("TRIAGE_FAKE_MODELS")
    real = deps_module.load_deps()
    assert real.demo_limiter.engine.url.database == str(paths["real_log"])
    assert real.demo_limiter.status("visitor-1").used == 0 and fake.demo_limiter.status("visitor-1").used == 1


def test_fake_mode_ignores_database_url(paths, monkeypatch):
    """.env may set DATABASE_URL (the real database): fake mode still logs only to its own file."""
    monkeypatch.setenv("DATABASE_URL", db_module.sqlite_url(paths["real_log"]))
    monkeypatch.setenv("TRIAGE_FAKE_MODELS", "1")
    fake = deps_module.load_deps()
    ask(fake)
    assert fake.log.engine.url.database == str(paths["fake_log"]) and len(fake.log.rows()) == 1
    assert not paths["real_log"].exists()
