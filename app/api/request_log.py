"""One row per API request (Phase 4 Step 5). Step 10 swaps the in-memory log for SQLite.

Never stores API keys (only key_id, a short hash) or prompt/answer text (only the prompt's SHA-256).
"""
import threading
from dataclasses import asdict, dataclass, field
from typing import Optional, Protocol


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
    prompt_sha256: Optional[str] = None
    # verification (filled in from Step 7 on)
    verified: Optional[str] = "off"
    escalated: Optional[bool] = False
    checker_verdict: Optional[str] = None
    decided_by: Optional[str] = None
    checker_version: Optional[int] = None  # or the judge's version, whichever decided
    mode: Optional[str] = None             # demo | byok | api (filled in later)

    def to_dict(self) -> dict:
        return asdict(self)


class RequestLog(Protocol):
    def write(self, row: RequestRow) -> None: ...
    def rows(self) -> list: ...
    def get(self, request_id: str) -> Optional[RequestRow]: ...


class InMemoryRequestLog:
    def __init__(self):
        self._rows, self._by_id = [], {}
        self._lock = threading.Lock()

    def write(self, row: RequestRow) -> None:
        with self._lock:
            self._rows.append(row)
            self._by_id[row.request_id] = row

    def rows(self) -> list:
        with self._lock:
            return list(self._rows)

    def get(self, request_id: str) -> Optional[RequestRow]:
        with self._lock:
            return self._by_id.get(request_id)
