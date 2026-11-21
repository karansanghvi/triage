"""GET /healthz: is this server ready to route requests?

Ready means the classifier and the embedding model are loaded in this process (checked on every call: it's free),
and the database answered `SELECT 1` within DB_TIMEOUT seconds. The database result is cached for DB_TTL seconds:
the host's health checks come every few seconds, and without the cache each one would be a database round trip
(keeping a serverless database awake) and a brief network blip would flip the server to unhealthy and back.

Never calls a model provider, and never returns anything but status words (no URLs, versions or errors).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from sqlalchemy import text

DB_TTL = 60.0       # seconds a database result is reused
DB_TIMEOUT = 2.0    # seconds the database gets to answer


@dataclass(frozen=True)
class DbStatus:
    ok: bool
    checked_at: float           # clock() time of the check


class DbHealthCache:
    """Runs `probe` at most once per `ttl` seconds and reuses its result in between. `probe` returns True or
    False (or raises, which counts as False). Thread-safe: concurrent callers share one probe."""

    def __init__(self, probe: Callable[[], bool], ttl: float = DB_TTL, clock: Callable[[], float] = time.monotonic):
        self.probe, self.ttl, self.clock = probe, ttl, clock
        self._last: Optional[DbStatus] = None
        self._lock = threading.Lock()

    def status(self) -> DbStatus:
        with self._lock:
            now = self.clock()
            if self._last is not None and now - self._last.checked_at < self.ttl:
                return self._last
            try:
                ok = bool(self.probe())
            except Exception:
                ok = False
            self._last = DbStatus(ok=ok, checked_at=now)
            return self._last


def select_one_probe(engine, timeout: float = DB_TIMEOUT) -> Callable[[], bool]:
    """`SELECT 1` on the engine, given at most `timeout` seconds (run in a thread, so a hung connect can't block
    the caller longer than that; the thread finishes or fails on its own)"""
    def probe() -> bool:
        result: dict = {}

        def run():
            try:
                with engine.connect() as connection:
                    result["ok"] = connection.execute(text("SELECT 1")).scalar() == 1
            except Exception:
                result["ok"] = False

        worker = threading.Thread(target=run, name="healthz-db", daemon=True)
        worker.start()
        worker.join(timeout)
        return bool(result.get("ok"))
    return probe


def model_status(deps) -> tuple[bool, bool]:
    """(classifier loaded, embedding model loaded) for this process. Never loads anything."""
    predictor = getattr(deps, "predictor", None) if deps is not None else None
    classifier = bool(getattr(predictor, "models", None))
    if not classifier:
        return False, False
    feature_set = getattr(predictor, "feature_set", None)
    if feature_set not in {"embeddings", "both"}:
        return True, True                       # this classifier doesn't use embeddings
    from app.router.embeddings import is_loaded
    return True, is_loaded()


def health_report(deps, db_cache: Optional[DbHealthCache]) -> tuple[int, dict]:
    """(HTTP status, body). 200 only when everything is ready."""
    classifier, embeddings = model_status(deps)
    database = db_cache.status().ok if db_cache is not None else False
    ready = classifier and embeddings and database
    body = {
        "status": "ok" if ready else "unavailable",
        "classifier": "ready" if classifier else "not loaded",
        "embedding_model": "ready" if embeddings else "not loaded",
        "database": "ok" if database else "unreachable",
    }
    return (200 if ready else 503), body
