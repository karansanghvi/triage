"""GET /healthz (liveness) and GET /readyz (readiness).

/healthz: the classifier and the embedding model are loaded in this process. It never touches the database, so
Docker's healthcheck and uptime monitors (every 30 seconds or so) can't keep a serverless database awake: Neon
scales to zero after 5 idle minutes, and an always-awake compute would use more than the Free plan's CU-hours.

/readyz: the same, plus the database answered `SELECT 1` within DB_TIMEOUT seconds. For the deploy smoke test and
manual checks only; each call can wake the database. A result is reused for DB_TTL seconds, so a burst of checks
runs one query.

Neither calls a model provider, and neither returns anything but status words (no URLs, versions or errors).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from sqlalchemy import text

DB_TTL = 10.0       # seconds a /readyz database result is reused
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


def liveness_report(deps) -> tuple[int, dict]:
    """/healthz: (HTTP status, body). 200 when the classifier and embedding model are loaded. No database."""
    classifier, embeddings = model_status(deps)
    alive = classifier and embeddings
    body = {
        "status": "ok" if alive else "unavailable",
        "classifier": "ready" if classifier else "not loaded",
        "embedding_model": "ready" if embeddings else "not loaded",
    }
    return (200 if alive else 503), body


def readiness_report(deps, db_cache: Optional[DbHealthCache]) -> tuple[int, dict]:
    """/readyz: (HTTP status, body). 200 only when the models are loaded and the database answered."""
    code, body = liveness_report(deps)
    database = db_cache.status().ok if db_cache is not None else False
    ready = code == 200 and database
    return (200 if ready else 503), {**body, "status": "ok" if ready else "unavailable",
                                     "database": "ok" if database else "unreachable"}
