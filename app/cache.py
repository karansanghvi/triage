"""Exact-match response cache: repeated temperature=0 requests cost nothing."""
import dataclasses
import hashlib
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

from app.providers.base import (
    DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, Response, send_request,
)
from app.registry import PROJECT_ROOT, ModelConfig

logger = logging.getLogger("triage.cache")

DEFAULT_CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "responses.sqlite3"
DEFAULT_TTL_SECONDS = 24 * 60 * 60       # saved answers expire after 24 hours
CACHE_VERSION = 1                        # bump this to throw away every old entry at once

# Finish reasons meaning the answer was cut off by the token limit: never cache those
TRUNCATED_FINISH_REASONS = {"length", "max_tokens"}


def is_cacheable(temperature: float, stream: bool) -> bool:
    """Only temperature=0, non-streaming requests are cached.

    At temperature 0 the model gives (nearly) the same answer every time,
    so a saved answer is a fair substitute. Above 0, variety is the point.
    """
    return float(temperature) == 0.0 and not stream


def make_cache_key(messages: list, model: ModelConfig, temperature: float, max_tokens: int,
                   reasoning_effort: Optional[str] = None, response_format: Optional[dict] = None) -> str:
    """A fingerprint of everything that affects the answer.

    sort_keys makes the same data always produce the same text,
    and float() stops 0 and 0.0 from becoming two different keys.
    The timeout isn't included, because it doesn't change the answer.
    """
    payload = {
        "version": CACHE_VERSION,
        "provider": model.provider,
        "model_id": model.model_id,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    if reasoning_effort is not None:   # only added when set, so existing keys don't change
        payload["reasoning_effort"] = reasoning_effort
    if response_format is not None:
        payload["response_format"] = response_format
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _worth_caching(response: Response) -> bool:
    """Don't save empty answers or answers cut off by the token limit."""
    if response.cached or not response.text.strip():
        return False
    return (response.finish_reason or "").lower() not in TRUNCATED_FINISH_REASONS


class ResponseCache:
    """Saved answers in a SQLite file, so they survive restarts."""

    def __init__(self, path=DEFAULT_CACHE_PATH, ttl_seconds: float = DEFAULT_TTL_SECONDS, clock=time.time):
        # time.time (real date and time), not time.monotonic: expiry must survive restarts
        self.path = Path(path)
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()     # safe if several requests use the cache at once
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS responses (
                    key TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                )
                """
            )

    def get(self, key: str) -> Optional[Response]:
        """The saved Response for this key, or None if missing or expired."""
        with self._lock:
            row = self._conn.execute(
                "SELECT response_json, expires_at FROM responses WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            response_json, expires_at = row
            if expires_at <= self._clock():
                with self._conn:
                    self._conn.execute("DELETE FROM responses WHERE key = ?", (key,))
                return None

        try:
            data = json.loads(response_json)
            known = {f.name for f in dataclasses.fields(Response)}
            return Response(**{k: v for k, v in data.items() if k in known})
        except (ValueError, TypeError):
            # A row saved by an older version of Response: treat it as a miss
            return None

    def set(self, key: str, response: Response) -> None:
        now = self._clock()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?, ?)",
                (key, response.model_name, json.dumps(dataclasses.asdict(response)),
                 now, now + self.ttl_seconds),
            )

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]

    def purge_expired(self) -> int:
        """Delete every expired entry. Returns how many were removed."""
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "DELETE FROM responses WHERE expires_at <= ?", (self._clock(),)
            )
            return cursor.rowcount

    def clear(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM responses")

    def close(self) -> None:
        self._conn.close()


class CachedSender:
    """Wraps send_request: same inputs and outputs, but checks the cache first.

    Plug it into the reliability layer with ReliableClient(send_fn=CachedSender(cache)).
    """

    def __init__(self, cache: ResponseCache, send_fn=send_request):
        self.cache = cache
        self._send = send_fn
        self.hits = 0
        self.misses = 0
        self.saved_cost = 0.0     # total reported cost the cache avoided

    def __call__(
        self,
        messages: list,
        model: ModelConfig,
        stream: bool = False,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        **settings,
    ):
        if not is_cacheable(temperature, stream):
            return self._send(
                messages, model, stream=stream, temperature=temperature,
                max_tokens=max_tokens, **settings,
            )

        key = make_cache_key(messages, model, temperature, max_tokens, settings.get("reasoning_effort"),
                             settings.get("response_format"))
        started_at = time.perf_counter()
        saved = self.cache.get(key)

        if saved is not None:
            lookup_ms = (time.perf_counter() - started_at) * 1000
            self.hits += 1
            self.saved_cost += saved.reported_cost
            logger.info("cache hit: %s (saved $%.6f)", model.name, saved.reported_cost)
            return dataclasses.replace(
                saved,
                cached=True,
                latency_ms=round(lookup_ms, 1),
                cost=0.0,
                reported_cost=0.0,
                cache_saved_cost=saved.reported_cost,
            )

        self.misses += 1
        response = self._send(
            messages, model, temperature=temperature, max_tokens=max_tokens, **settings,
        )
        if _worth_caching(response):
            self.cache.set(key, response)
        return response

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0