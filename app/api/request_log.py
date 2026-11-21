"""One row per API request (Phase 4 Steps 5, 8 and 10).

    SQLRequestLog       what the app uses: the database (app/db.py; DATABASE_URL, default data/triage.sqlite3)
    SQLiteRequestLog    the same in one SQLite file of its own (tests, scripts)
    InMemoryRequestLog  for tests (or TRIAGE_LOG=memory)

Never stores API keys (only key_id, a short hash) or prompt/answer text. The prompt is kept only as an
HMAC-SHA256 keyed with TRIAGE_HASH_SECRET (prompt_hmac): it can match repeats of the same prompt, but without
the secret it can't be checked against a guessed prompt the way a plain hash could.
For model="auto" requests that answered, the classifier's inputs are kept for retraining (Step 8): the 14
handcrafted features and the 384-dim embedding, with FEATURE_VERSION and the embedding model's name.
Raw text is never stored in any mode yet. NOTE: demo mode will later store raw text, behind a visible
notice to the user; that is not built.
"""
import hashlib
import hmac
import json
import os
import secrets
import threading
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

LOG_KIND_ENV, HASH_SECRET_ENV = "TRIAGE_LOG", "TRIAGE_HASH_SECRET"
VECTOR_FIELDS = ("features", "embedding")          # skipped by reads that don't need them (stats, lookups)
PRIVATE_FIELDS = ("prompt_hmac",)                  # never returned by an endpoint


@dataclass
class RequestRow:
    request_id: str
    timestamp: str                         # UTC ISO 8601, when the request arrived
    key_id: str                            # first 8 hex characters of the key's SHA-256
    requested_model: str                   # "auto" or a pass-through name
    stream: bool
    status: str = "ok"                     # ok | error
    http_status: int = 200
    error_kind: Optional[str] = None
    # routing
    quality: Optional[str] = None          # X-Triage-Quality as sent (None = default)
    tau: Optional[float] = None
    task_type: Optional[str] = None
    task_type_guessed: Optional[bool] = None
    classifier: Optional[str] = None       # "name v<version>"
    config_version: Optional[int] = None
    p_low: Optional[float] = None
    p_medium: Optional[float] = None
    tier: Optional[str] = None
    primary_model: Optional[str] = None
    served_model: Optional[str] = None
    fell_back: Optional[bool] = None
    attempts: list = field(default_factory=list)   # [{model, outcome, error_kind}]
    # generation settings actually used
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    settings_match_training: Optional[bool] = None # auto only: same as Phase 2 (Step 8 drops mismatches)
    # usage and cost (None when unknown, e.g. a stream that failed part-way)
    cache_hit: Optional[bool] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    tokens_estimated: Optional[bool] = None
    cost: Optional[float] = None           # what was actually paid
    price_as_cost: Optional[float] = None  # realistic paid price (0 on a cache hit)
    strong_cost: Optional[float] = None    # the strong model at the same token counts (an estimate)
    saved: Optional[float] = None          # strong_cost - price_as_cost
    latency_ms: Optional[float] = None
    ttft_ms: Optional[float] = None
    prompt_hmac: Optional[str] = None      # HMAC-SHA256 of the messages, keyed with TRIAGE_HASH_SECRET
    # verification (Step 7)
    verified: Optional[str] = "off"        # off | sync (background labels live in the labels table)
    escalated: Optional[bool] = False
    checker_verdict: Optional[str] = None  # the deciding check's verdict: good | bad | error
    decided_by: Optional[str] = None       # format | ai_checker | ai_checker_error | none (sync only)
    checker_version: Optional[int] = None  # the AI checker's version, when it ran
    mode: Optional[str] = None             # demo | byok | api ("api" for key-authenticated requests)
    preferred_provider: Optional[str] = None   # X-Triage-Prefer-Provider when applied (BYOK, key sent, "auto")
    user_id: Optional[str] = None          # the account that made the request (accounts, later); None otherwise
    escalated_from: Optional[str] = None   # the tier the router picked
    escalated_to: Optional[str] = None     # the tier whose answer was served
    escalation_extra_cost: Optional[float] = None       # price_as cost of the extra answers
    escalation_extra_latency_ms: Optional[float] = None
    check_cost: Optional[float] = None     # price_as cost of the AI checks
    check_latency_ms: Optional[float] = None
    # Step 8 storage (auto requests that answered; None otherwise)
    features: Optional[list] = None        # the 14 handcrafted features, in features.FEATURE_NAMES order
    embedding: Optional[bytes] = None      # float32 little-endian, 384 values (embedding_from_bytes)
    feature_version: Optional[str] = None  # features.FEATURE_VERSION
    embedding_model: Optional[str] = None  # embeddings.MODEL_NAME
    training_eligible: bool = False        # auto, ok, no fallback, Phase 2 settings, not a cache hit (Step 8)
    # Phase 5 (migration 0008; last, as the column was added last)
    routed_by: Optional[str] = None        # classifier | small_talk | requested_model (what decided the tier)

    def to_dict(self) -> dict:
        return asdict(self)

    def public_dict(self) -> dict:
        """For GET /v1/requests/{id}: everything except the raw vectors (vectors_stored says if they exist) and
        prompt_hmac (a fingerprint of the prompt: it's for matching repeats inside Triage, not for callers)."""
        out = {k: v for k, v in asdict(self).items() if k not in VECTOR_FIELDS and k not in PRIVATE_FIELDS}
        out["vectors_stored"] = self.feature_version is not None or self.embedding_model is not None
        return out


def prompt_hmac(messages: list, secret: bytes) -> str:
    """HMAC-SHA256 of the conversation (canonical JSON), keyed with the server's hash secret."""
    text = json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hmac.new(secret, text, hashlib.sha256).hexdigest()


def hash_secret_from_env(required: bool) -> bytes:
    """TRIAGE_HASH_SECRET as bytes. Missing: an error if required (the database log keeps hashes across restarts,
    so they must use a fixed secret), otherwise a random per-process secret (the in-memory log)."""
    value = os.environ.get(HASH_SECRET_ENV, "")
    if value:
        return value.encode("utf-8")
    if required:
        raise RuntimeError(f"refusing to start: set {HASH_SECRET_ENV} (a long random string) so prompts are "
                           "stored as a keyed HMAC in the request log")
    return secrets.token_bytes(32)


def embedding_to_bytes(vector) -> bytes:
    return np.asarray(vector, dtype="<f4").tobytes()


def embedding_from_bytes(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype="<f4").copy()


def is_training_eligible(row: RequestRow) -> bool:
    """Only answers the classifier's labels would describe: model="auto", answered without an error, by the
    tier's own first model (no fallback), not escalated by verification (then the served answer isn't from the
    model the classifier predicted for), with the settings the classifier learned from (Phase 2's), and not
    served from the response cache (a cached answer is a repeat: it adds no new information and would
    over-weight repeated prompts in retraining), and only from an "api" request (a Triage server key).
    An allow-list: BYOK, demo, account and any future or unknown mode are never eligible (their prompts are a
    user's or a public visitor's). Small talk is never eligible: the classifier didn't decide it, and greetings
    would teach it nothing."""
    return (row.mode == "api" and row.requested_model == "auto" and row.status == "ok" and row.fell_back is False
            and not row.escalated and row.settings_match_training is True and not row.cache_hit
            and row.routed_by != "small_talk")


@dataclass
class LabelRow:
    """A background label (app/api/background.py): the served answer graded against the strong model's
    answer. No text. NOISY: the reference can be wrong too (see background.py)."""
    request_id: str
    created_at: str                        # UTC ISO 8601
    status: str                            # labeled | skipped
    skip_reason: Optional[str] = None      # daily_cap, no_reference, error, model_failed, error:<type>
    passed: Optional[bool] = None
    judge_score: Optional[int] = None      # 1-5, after the constraint cap
    judge_version: Optional[int] = None
    judge_model: Optional[str] = None
    reference_model: Optional[str] = None  # the strong model that wrote the reference answer
    served_model: Optional[str] = None
    served_tier: Optional[str] = None
    calls: int = 0                         # strong answer + judge calls this label used
    tokens: int = 0
    cost: Optional[float] = None           # price_as


class RequestLog(Protocol):
    def write(self, row: RequestRow) -> None: ...
    # oldest first; key_id / user_id = only that key's / account's; include_vectors=False leaves the vectors None
    def rows(self, key_id: Optional[str] = None, include_vectors: bool = True,
             user_id: Optional[str] = None) -> list: ...
    def get(self, request_id: str, include_vectors: bool = True) -> Optional[RequestRow]: ...
    def write_label(self, label: LabelRow) -> None: ...
    def labels(self, request_id: Optional[str] = None) -> list: ...
    def label_usage_since(self, created_at: str) -> tuple: ...      # (calls, tokens) of labels since then
    # retention: delete request rows older than `cutoff` (and their labels); returns how many request rows went
    def delete_older_than(self, cutoff: str, batch_size: int = 1000) -> int: ...


# ---------- the database (app/db.py: SQLite locally, Postgres in production) ----------

_JSON = {"attempts", "features"}            # stored as JSON text
_REQUEST_NAMES = [f.name for f in fields(RequestRow)]
_LABEL_NAMES = [f.name for f in fields(LabelRow)]


def _to_sql(name, value):
    if value is not None and name in _JSON:
        return json.dumps(value)
    return value


def _from_sql(name, value):
    if value is None:
        return None
    if name in _JSON:
        return json.loads(value)
    if name == "embedding":
        return bytes(value)                 # Postgres drivers may hand back a memoryview
    return value


class SQLRequestLog:
    """The request log in the database (app/db.py). Each write is its own transaction; the engine's pool gives
    every worker thread its own connection. `owns_engine`: close() disposes it (a log opened on its own)."""

    persistent = True                       # outlives the process: prompt HMACs need a fixed TRIAGE_HASH_SECRET

    def __init__(self, engine, owns_engine: bool = False):
        from app import db
        self.engine, self._owns_engine = engine, owns_engine
        self._requests, self._labels = db.requests_table, db.labels_table

    def write(self, row: RequestRow) -> None:
        values = {n: _to_sql(n, getattr(row, n)) for n in _REQUEST_NAMES}
        with self.engine.begin() as connection:
            connection.execute(self._requests.insert(), values)

    def delete_older_than(self, cutoff: str, batch_size: int = 1000) -> int:
        """Retention: delete request rows whose timestamp (UTC ISO 8601, so text order is time order) is before
        `cutoff`, and their labels. In batches of `batch_size`, each its own short transaction, oldest first
        (idx_requests_timestamp), so the table is never locked for long and a crash keeps every finished batch."""
        from sqlalchemy import delete, select
        requests, labels = self._requests, self._labels
        removed = 0
        while True:
            with self.engine.begin() as connection:
                batch = connection.execute(
                    select(requests.c.id, requests.c.request_id).where(requests.c.timestamp < cutoff)
                    .order_by(requests.c.timestamp).limit(batch_size)).fetchall()
                if not batch:
                    return removed
                connection.execute(delete(labels).where(labels.c.request_id.in_([r.request_id for r in batch])))
                connection.execute(delete(requests).where(requests.c.id.in_([r.id for r in batch])))
            removed += len(batch)
            if len(batch) < batch_size:
                return removed

    def _select(self, where=None, include_vectors: bool = True) -> list:
        from sqlalchemy import select
        names = _REQUEST_NAMES if include_vectors else [n for n in _REQUEST_NAMES if n not in VECTOR_FIELDS]
        query = select(*[self._requests.c[n] for n in names]).order_by(self._requests.c.id)   # skipped: not read
        if where is not None:
            query = query.where(where)
        with self.engine.connect() as connection:
            found = connection.execute(query).fetchall()
        rows = []
        for values in found:
            data = {n: _from_sql(n, v) for n, v in zip(names, values)}
            if data["attempts"] is None:
                data["attempts"] = []
            if data["training_eligible"] is None:
                data["training_eligible"] = False
            rows.append(RequestRow(**data))
        return rows

    def rows(self, key_id: Optional[str] = None, include_vectors: bool = True,
             user_id: Optional[str] = None) -> list:
        from sqlalchemy import and_
        c = self._requests.c
        conditions = [cond for cond in (None if key_id is None else c.key_id == key_id,
                                        None if user_id is None else c.user_id == user_id) if cond is not None]
        return self._select(and_(*conditions) if conditions else None, include_vectors)

    def get(self, request_id: str, include_vectors: bool = True) -> Optional[RequestRow]:
        found = self._select(self._requests.c.request_id == request_id, include_vectors)
        return found[0] if found else None

    # ----- labels -----

    def write_label(self, label: LabelRow) -> None:
        with self.engine.begin() as connection:
            connection.execute(self._labels.insert(), {n: _to_sql(n, getattr(label, n)) for n in _LABEL_NAMES})

    def labels(self, request_id: Optional[str] = None) -> list:
        from sqlalchemy import select
        query = select(*[self._labels.c[n] for n in _LABEL_NAMES]).order_by(self._labels.c.id)
        if request_id is not None:
            query = query.where(self._labels.c.request_id == request_id)
        with self.engine.connect() as connection:
            found = connection.execute(query).fetchall()
        return [LabelRow(**{n: _from_sql(n, v) for n, v in zip(_LABEL_NAMES, values)}) for values in found]

    def label_usage_since(self, created_at: str) -> tuple:
        from sqlalchemy import func, select
        c = self._labels.c
        query = select(func.coalesce(func.sum(c.calls), 0), func.coalesce(func.sum(c.tokens), 0)) \
            .where(c.created_at >= created_at)
        with self.engine.connect() as connection:
            calls, tokens = connection.execute(query).one()
        return int(calls), int(tokens)

    def run(self, fn):
        """fn(connection): read-only queries outside this module (the stats in app/api/usage_sql.py)."""
        with self.engine.connect() as connection:
            return fn(connection)

    def close(self) -> None:
        if self._owns_engine:
            self.engine.dispose()


class InMemoryRequestLog(SQLRequestLog):
    """For tests (or TRIAGE_LOG=memory): the same SQL log on a private in-memory SQLite database, so every log
    answers the same SQL queries (the stats endpoints aggregate in SQL). Built with create_all, not the
    migrations (fast; tests/test_db.py checks the two match). Foreign keys aren't enforced: a log on its own
    has no users table to point at. One connection is shared by every thread, so each call holds a lock."""

    persistent = False

    def __init__(self):
        from sqlalchemy import create_engine
        from sqlalchemy.pool import StaticPool
        from app import db
        engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
        db.metadata.create_all(engine)
        super().__init__(engine, owns_engine=True)
        self._lock = threading.RLock()

    def write(self, row: RequestRow) -> None:
        with self._lock:
            super().write(row)

    def delete_older_than(self, cutoff: str, batch_size: int = 1000) -> int:
        with self._lock:
            return super().delete_older_than(cutoff, batch_size)

    def _select(self, where=None, include_vectors: bool = True) -> list:
        with self._lock:
            return super()._select(where, include_vectors)

    def write_label(self, label: LabelRow) -> None:
        with self._lock:
            super().write_label(label)

    def labels(self, request_id: Optional[str] = None) -> list:
        with self._lock:
            return super().labels(request_id)

    def label_usage_since(self, created_at: str) -> tuple:
        with self._lock:
            return super().label_usage_since(created_at)

    def run(self, fn):
        """fn(connection) under the lock: the stats queries (app/api/usage_sql.py) go through here."""
        with self._lock, self.engine.connect() as connection:
            return fn(connection)


class SQLiteRequestLog(SQLRequestLog):
    """A request log in its own SQLite file (tests, scripts), created and upgraded on open."""

    def __init__(self, path=None):
        from app import db
        self.path = Path(path or db.DEFAULT_DB)
        super().__init__(db.open_engine(db.sqlite_url(self.path)), owns_engine=True)


def open_request_log(fake: bool = False, engine=None):
    """The app's log: in the database unless TRIAGE_LOG=memory. `engine`: the app's shared one; None opens
    app.db.database_url(fake): DATABASE_URL, or in fake-model mode its own SQLite file, so fake answers never land
    in the real log (they would look like real traffic to stats and retraining)."""
    if os.environ.get(LOG_KIND_ENV, "sql").lower() == "memory":
        return InMemoryRequestLog()
    if engine is not None:
        return SQLRequestLog(engine)
    from app import db
    return SQLRequestLog(db.open_engine(db.database_url(fake)), owns_engine=True)
