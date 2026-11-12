"""One row per API request (Phase 4 Steps 5, 8 and 10).

    SQLiteRequestLog    what the app uses (TRIAGE_LOG_DB, default data/logs/requests.sqlite3): survives restarts
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
import sqlite3
import threading
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Optional, Protocol, get_args

import numpy as np

from app.registry import PROJECT_ROOT

LOG_DB_ENV, LOG_KIND_ENV, HASH_SECRET_ENV = "TRIAGE_LOG_DB", "TRIAGE_LOG", "TRIAGE_HASH_SECRET"
DEFAULT_LOG_DB = PROJECT_ROOT / "data" / "logs" / "requests.sqlite3"
FAKE_LOG_DB = PROJECT_ROOT / "data" / "logs" / "fake_requests.sqlite3"   # TRIAGE_FAKE_MODELS=1 runs log here
SCHEMA_VERSION = 3        # 2: prompt_sha256 -> prompt_hmac. 3: verification fields and the labels table (Step 7)
VECTOR_FIELDS = ("features", "embedding")          # skipped by reads that don't need them (stats, lookups)


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

    def to_dict(self) -> dict:
        return asdict(self)

    def public_dict(self) -> dict:
        """For GET /v1/requests/{id}: everything except the raw vectors; vectors_stored says if they exist."""
        out = {k: v for k, v in asdict(self).items() if k not in VECTOR_FIELDS}
        out["vectors_stored"] = self.feature_version is not None or self.embedding_model is not None
        return out


def prompt_hmac(messages: list, secret: bytes) -> str:
    """HMAC-SHA256 of the conversation (canonical JSON), keyed with the server's hash secret."""
    text = json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hmac.new(secret, text, hashlib.sha256).hexdigest()


def hash_secret_from_env(required: bool) -> bytes:
    """TRIAGE_HASH_SECRET as bytes. Missing: an error if required (the SQLite log keeps hashes across restarts,
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
    over-weight repeated prompts in retraining)."""
    return (row.requested_model == "auto" and row.status == "ok" and row.fell_back is False
            and not row.escalated and row.settings_match_training is True and not row.cache_hit)


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
    # oldest first; key_id = only that key's; include_vectors=False leaves features and embedding None
    def rows(self, key_id: Optional[str] = None, include_vectors: bool = True) -> list: ...
    def get(self, request_id: str, include_vectors: bool = True) -> Optional[RequestRow]: ...
    def write_label(self, label: LabelRow) -> None: ...
    def labels(self, request_id: Optional[str] = None) -> list: ...
    def label_usage_since(self, created_at: str) -> tuple: ...      # (calls, tokens) of labels since then


class InMemoryRequestLog:
    def __init__(self):
        self._rows, self._by_id, self._labels = [], {}, []
        self._lock = threading.Lock()

    def write_label(self, label: LabelRow) -> None:
        with self._lock:
            self._labels.append(label)

    def labels(self, request_id: Optional[str] = None) -> list:
        with self._lock:
            return [l for l in self._labels if request_id is None or l.request_id == request_id]

    def label_usage_since(self, created_at: str) -> tuple:
        with self._lock:
            recent = [l for l in self._labels if l.created_at >= created_at]
        return sum(l.calls for l in recent), sum(l.tokens for l in recent)

    def write(self, row: RequestRow) -> None:
        with self._lock:
            self._rows.append(row)
            self._by_id[row.request_id] = row

    @staticmethod
    def _view(r: RequestRow, include_vectors: bool) -> RequestRow:
        return r if include_vectors else replace(r, features=None, embedding=None)

    def rows(self, key_id: Optional[str] = None, include_vectors: bool = True) -> list:
        with self._lock:
            return [self._view(r, include_vectors) for r in self._rows if key_id is None or r.key_id == key_id]

    def get(self, request_id: str, include_vectors: bool = True) -> Optional[RequestRow]:
        with self._lock:
            r = self._by_id.get(request_id)
        return None if r is None else self._view(r, include_vectors)


# ---------- SQLite ----------

_FIELDS = [f for f in fields(RequestRow)]
_LABEL_FIELDS = [f for f in fields(LabelRow)]


def _base_type(t):
    """Optional[X] -> X; X -> X."""
    args = [a for a in get_args(t) if a is not type(None)]
    return args[0] if args else t


_SQL_TYPE = {str: "TEXT", int: "INTEGER", float: "REAL", bool: "INTEGER", bytes: "BLOB", list: "TEXT"}
_JSON = {"attempts", "features"}            # stored as JSON text
_BOOL = {f.name for f in _FIELDS + _LABEL_FIELDS if _base_type(f.type) is bool}


def _column(f) -> str:
    return f"{f.name} {_SQL_TYPE[_base_type(f.type)]}" + (" NOT NULL UNIQUE" if f.name == "request_id" else "")


SCHEMA = (
    "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)",
    "CREATE TABLE IF NOT EXISTS requests (id INTEGER PRIMARY KEY AUTOINCREMENT, "
    + ", ".join(_column(f) for f in _FIELDS) + ")",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_requests_request_id ON requests (request_id)",
    "CREATE INDEX IF NOT EXISTS idx_requests_key_id ON requests (key_id)",
    "CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON requests (timestamp)",
    "CREATE TABLE IF NOT EXISTS labels (id INTEGER PRIMARY KEY AUTOINCREMENT, "
    + ", ".join(_column(f) for f in _LABEL_FIELDS) + ")",
    "CREATE INDEX IF NOT EXISTS idx_labels_request_id ON labels (request_id)",
    "CREATE INDEX IF NOT EXISTS idx_labels_created_at ON labels (created_at)",
)


class SQLiteRequestLog:
    """The request log in one SQLite file, in WAL mode (readers don't block the writer, and a crash keeps
    every committed row). One connection shared by the server's worker threads, behind a lock; each write
    is its own transaction."""

    def __init__(self, path=None):
        self.path = Path(path or os.environ.get(LOG_DB_ENV) or DEFAULT_LOG_DB)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None, timeout=30)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("PRAGMA busy_timeout=30000")
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            for statement in SCHEMA:
                self._db.execute(statement)
            found = self._db.execute("SELECT version FROM schema_version").fetchall()
            if not found:
                self._db.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
            self._db.execute("COMMIT")
        if found and found[0][0] != SCHEMA_VERSION:
            self._db.close()
            raise RuntimeError(f"{self.path} has request-log schema version {found[0][0]}, this code writes "
                               f"version {SCHEMA_VERSION}; migrate it or point {LOG_DB_ENV} at a new file")
        self._names = [f.name for f in _FIELDS]

    @staticmethod
    def _to_sql(name, value):
        if value is None:
            return None
        if name in _JSON:
            return json.dumps(value)
        if name in _BOOL:
            return int(bool(value))
        return value

    @staticmethod
    def _from_sql(name, value):
        if value is None:
            return None
        if name in _JSON:
            return json.loads(value)
        if name in _BOOL:
            return bool(value)
        return value

    def write(self, row: RequestRow) -> None:
        values = [self._to_sql(n, getattr(row, n)) for n in self._names]
        sql = f"INSERT INTO requests ({', '.join(self._names)}) VALUES ({', '.join('?' * len(self._names))})"
        with self._lock:
            self._db.execute(sql, values)

    def _select(self, where: str = "", args: tuple = (), include_vectors: bool = True) -> list:
        names = self._names if include_vectors else [n for n in self._names if n not in VECTOR_FIELDS]
        sql = f"SELECT {', '.join(names)} FROM requests {where} ORDER BY id"     # skipped columns aren't read
        with self._lock:
            found = self._db.execute(sql, args).fetchall()
        rows = []
        for values in found:
            data = {n: self._from_sql(n, v) for n, v in zip(names, values)}
            if data["attempts"] is None:
                data["attempts"] = []
            if data["training_eligible"] is None:
                data["training_eligible"] = False
            rows.append(RequestRow(**data))
        return rows

    def rows(self, key_id: Optional[str] = None, include_vectors: bool = True) -> list:
        if key_id is not None:
            return self._select("WHERE key_id = ?", (key_id,), include_vectors)
        return self._select(include_vectors=include_vectors)

    def get(self, request_id: str, include_vectors: bool = True) -> Optional[RequestRow]:
        found = self._select("WHERE request_id = ?", (request_id,), include_vectors)
        return found[0] if found else None

    # ----- labels -----

    def write_label(self, label: LabelRow) -> None:
        names = [f.name for f in _LABEL_FIELDS]
        values = [self._to_sql(n, getattr(label, n)) for n in names]
        sql = f"INSERT INTO labels ({', '.join(names)}) VALUES ({', '.join('?' * len(names))})"
        with self._lock:
            self._db.execute(sql, values)

    def labels(self, request_id: Optional[str] = None) -> list:
        names = [f.name for f in _LABEL_FIELDS]
        where, args = ("WHERE request_id = ?", (request_id,)) if request_id is not None else ("", ())
        with self._lock:
            found = self._db.execute(f"SELECT {', '.join(names)} FROM labels {where} ORDER BY id", args).fetchall()
        return [LabelRow(**{n: self._from_sql(n, v) for n, v in zip(names, values)}) for values in found]

    def label_usage_since(self, created_at: str) -> tuple:
        with self._lock:
            calls, tokens = self._db.execute("SELECT COALESCE(SUM(calls), 0), COALESCE(SUM(tokens), 0) FROM labels "
                                             "WHERE created_at >= ?", (created_at,)).fetchone()
        return int(calls), int(tokens)

    def close(self) -> None:
        with self._lock:
            self._db.close()


def open_request_log(fake: bool = False):
    """The app's log: SQLite unless TRIAGE_LOG=memory. The file is TRIAGE_LOG_DB if set, else the real log, or
    FAKE_LOG_DB in fake-model mode, so fake answers never land in the real log (they would look like real
    traffic to stats and retraining). Fake mode refuses the real log's path even when it is set explicitly,
    and real mode refuses the fake log's path."""
    if os.environ.get(LOG_KIND_ENV, "sqlite").lower() == "memory":
        return InMemoryRequestLog()
    path = Path(os.environ.get(LOG_DB_ENV) or (FAKE_LOG_DB if fake else DEFAULT_LOG_DB))
    forbidden = DEFAULT_LOG_DB if fake else FAKE_LOG_DB
    if path.resolve() == forbidden.resolve():
        raise RuntimeError(f"refusing to start: {'fake' if fake else 'real'}-model mode can't use {forbidden.name} "
                           f"({LOG_DB_ENV}); fake and real requests must be logged apart")
    return SQLiteRequestLog(path)
