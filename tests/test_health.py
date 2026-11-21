"""GET /healthz (liveness, never the database) and GET /readyz (plus a cached database check)."""
import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from app.api import health
from app.api.main import create_app
from tests.api_helpers import make_deps


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_the_database_result_is_reused_for_the_ttl():
    clock, calls = Clock(), []
    cache = health.DbHealthCache(lambda: calls.append(1) or True, ttl=60, clock=clock)
    assert cache.status().ok and len(calls) == 1
    clock.now += 59.9
    assert cache.status().ok and len(calls) == 1                 # still cached
    clock.now += 0.2
    assert cache.status().ok and len(calls) == 2                 # 60 s later: checked again


def test_a_failure_is_cached_too_and_recovers_after_the_ttl():
    clock, results = Clock(), iter([False, True])
    cache = health.DbHealthCache(lambda: next(results), ttl=60, clock=clock)
    assert not cache.status().ok
    clock.now += 30
    assert not cache.status().ok                                  # a blip doesn't flap back and forth
    clock.now += 31
    assert cache.status().ok


def test_a_probe_that_raises_counts_as_down():
    def boom():
        raise RuntimeError("connection refused")
    assert not health.DbHealthCache(boom, clock=Clock()).status().ok


def test_select_one_on_a_real_database():
    assert health.select_one_probe(create_engine("sqlite://"))() is True


def test_a_hung_database_gives_up_after_the_timeout():
    class Hung:
        def connect(self):
            time.sleep(3)
            raise RuntimeError("never")
    start = time.monotonic()
    assert health.select_one_probe(Hung(), timeout=0.2)() is False
    assert time.monotonic() - start < 1


class LoadedPredictor:
    """Looks like a loaded predictor that uses embeddings"""
    models = {"low": object(), "medium": object()}
    feature_set = "both"


def _client(deps):
    return TestClient(create_app(deps=deps, public=False, accounts=False))


class NoDatabase:
    """An engine that fails the test if anything connects to it"""
    url = "postgresql+psycopg://nowhere"

    def connect(self):
        raise AssertionError("/healthz touched the database")

    begin = connect


def test_healthz_is_200_when_the_models_are_loaded_and_never_touches_the_database(monkeypatch):
    monkeypatch.setattr("app.router.embeddings.is_loaded", lambda: True)
    deps = make_deps(predictor=LoadedPredictor())
    deps.engine = NoDatabase()
    app = create_app(deps=deps, public=False, accounts=False)
    with TestClient(app) as client:
        responses = [client.get("/healthz") for _ in range(3)]
    for response in responses:
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "classifier": "ready", "embedding_model": "ready"}
        assert response.headers["cache-control"] == "no-store"
    assert app.state.db_health is None                     # no database check was even built


def test_healthz_stays_200_when_the_database_is_down(monkeypatch):
    monkeypatch.setattr("app.router.embeddings.is_loaded", lambda: True)
    app = create_app(deps=make_deps(predictor=LoadedPredictor()), public=False, accounts=False)
    app.state.db_health = health.DbHealthCache(lambda: False)
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 503


def test_health_endpoints_need_no_key_and_show_no_details(monkeypatch):
    monkeypatch.setattr("app.router.embeddings.is_loaded", lambda: True)
    with _client(make_deps(predictor=LoadedPredictor())) as client:
        bodies = client.get("/healthz").text + client.get("/readyz").text
    assert "sqlite" not in bodies and "classifier_c" not in bodies and "Traceback" not in bodies


def test_healthz_is_503_when_the_embedding_model_isnt_loaded(monkeypatch):
    monkeypatch.setattr("app.router.embeddings.is_loaded", lambda: False)
    with _client(make_deps(predictor=LoadedPredictor())) as client:
        response = client.get("/healthz")
    assert response.status_code == 503 and response.json()["embedding_model"] == "not loaded"


def test_healthz_and_readyz_are_503_without_a_classifier():
    with _client(make_deps()) as client:                    # the API tests' fake predictor has no models
        for path in ("/healthz", "/readyz"):
            response = client.get(path)
            assert response.status_code == 503 and response.json()["classifier"] == "not loaded"


def test_readyz_is_200_when_everything_is_ready(monkeypatch):
    monkeypatch.setattr("app.router.embeddings.is_loaded", lambda: True)
    with _client(make_deps(predictor=LoadedPredictor())) as client:
        response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "classifier": "ready", "embedding_model": "ready", "database": "ok"}
    assert response.headers["cache-control"] == "no-store"


def test_readyz_is_503_when_the_database_is_down_and_reuses_the_result(monkeypatch):
    monkeypatch.setattr("app.router.embeddings.is_loaded", lambda: True)
    calls = []
    app = create_app(deps=make_deps(predictor=LoadedPredictor()), public=False, accounts=False)
    app.state.db_health = health.DbHealthCache(lambda: calls.append(1) or False)
    with TestClient(app) as client:
        first, second = client.get("/readyz"), client.get("/readyz")
    assert first.status_code == second.status_code == 503
    assert first.json()["database"] == "unreachable"
    assert len(calls) == 1                                  # the second check reused the first result


def test_readyz_gives_the_database_two_seconds_and_reuses_results_briefly():
    assert health.DB_TIMEOUT == 2.0
    assert health.DB_TTL <= 15                              # a burst shares one query; a later check is fresh


@pytest.mark.parametrize("feature_set", ["features", "task_only"])
def test_a_classifier_without_embeddings_doesnt_need_the_model(monkeypatch, feature_set):
    monkeypatch.setattr("app.router.embeddings.is_loaded", lambda: False)

    class P:
        models = {"low": object()}
    P.feature_set = feature_set
    assert health.model_status(type("D", (), {"predictor": P()})()) == (True, True)
