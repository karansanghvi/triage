"""Fake-model mode (TRIAGE_FAKE_MODELS=1) must never share the real response cache or request log.

Bug this guards against: a real-mode request was served "Fake answer from groq-small..." with X-Triage-Cache: hit,
because fake mode had cached its answers in data/cache/responses.sqlite3 under the same keys real mode reads.
Every path here is redirected to tmp_path, so the tests never touch the real files.
"""
import hashlib

import pytest
from fastapi.testclient import TestClient

from app.api import demo as demo_module
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
         "real_log": tmp_path / "logs" / "requests.sqlite3", "fake_log": tmp_path / "logs" / "fake.sqlite3",
         "real_demo": tmp_path / "demo" / "usage.sqlite3", "fake_demo": tmp_path / "demo" / "fake_usage.sqlite3"}
    monkeypatch.setattr(deps_module, "DEFAULT_CACHE_PATH", p["real_cache"])
    monkeypatch.setattr(deps_module, "FAKE_CACHE_PATH", p["fake_cache"])
    monkeypatch.setattr(request_log, "DEFAULT_LOG_DB", p["real_log"])
    monkeypatch.setattr(request_log, "FAKE_LOG_DB", p["fake_log"])
    monkeypatch.setattr(demo_module, "DEFAULT_DEMO_DB", p["real_demo"])      # never the real data/demo either
    monkeypatch.setattr(demo_module, "FAKE_DEMO_DB", p["fake_demo"])
    for name in ("TRIAGE_CACHE", "TRIAGE_CACHE_PATH", "TRIAGE_LOG_DB", "TRIAGE_LOG", "TRIAGE_FAKE_MODELS",
                 "TRIAGE_DEMO_DB", "TRIAGE_SERVER_PROVIDERS", "TRIAGE_TRUSTED_PROXY_HOPS"):
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
    assert fake.log.path == paths["fake_log"] and len(fake.log.rows()) == 1


@pytest.mark.parametrize("fake, env, target", [
    (True, "TRIAGE_CACHE_PATH", "real_cache"), (True, "TRIAGE_LOG_DB", "real_log"),
    (False, "TRIAGE_CACHE_PATH", "fake_cache"), (False, "TRIAGE_LOG_DB", "fake_log")])
def test_each_mode_refuses_the_other_modes_files_even_when_named_explicitly(paths, monkeypatch, fake, env, target):
    if fake:
        monkeypatch.setenv("TRIAGE_FAKE_MODELS", "1")
    monkeypatch.setenv(env, str(paths[target]))
    with pytest.raises(RuntimeError, match="refusing to start"):
        deps_module.load_deps()
    assert not paths[target].exists()


def test_the_cache_can_still_be_turned_off(paths, monkeypatch):
    monkeypatch.setenv("TRIAGE_FAKE_MODELS", "1")
    monkeypatch.setenv("TRIAGE_CACHE", "off")
    assert deps_module.load_deps().cached_sender is None


def test_fake_mode_keeps_its_own_demo_counters(paths, monkeypatch):
    monkeypatch.setenv("TRIAGE_FAKE_MODELS", "1")
    deps_module.load_deps()
    monkeypatch.delenv("TRIAGE_FAKE_MODELS")
    deps_module.load_deps()
    assert paths["fake_demo"].exists() and paths["real_demo"].exists()
