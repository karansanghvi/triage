"""GET /v1/stats/public: live aggregates for the dashboard's public Overview tab (no auth).

Only model="auto" requests that the classifier routed (small talk excluded; usage_sql.auto_routed), in the
range asked for (7d / 30d / all). Fake-model mode logs to its own database (app/db.py), so its requests are never
in a real server's log; a fake-mode server refuses this endpoint outright.

Nothing is published below THRESHOLD qualifying requests in the range: then only enough=false and the count,
so the page can say "N of 100 routed requests". Latency is shown only for models with at least
LATENCY_MIN_REQUESTS answered requests: fewer give noisy percentiles and could single out one visitor's traffic.

Computed in SQL (app/api/usage_sql.py) and cached per range for TTL_SECONDS in this process (each worker keeps its
own copy; fine at this traffic). Savings are against always using the strongest model, an estimate at the same
token counts.
"""
import threading
import time
from datetime import datetime, timezone
from typing import Callable

from sqlalchemy import and_

from app.api import usage_sql

THRESHOLD = 100
LATENCY_MIN_REQUESTS = 20
TTL_SECONDS = 300
TOTAL_FIELDS = ("requests", "ok", "errors", "priced_requests", "cost_usd", "strong_model_cost_usd", "saved_usd",
                "saved_share", "cache_hits", "fallbacks", "fallback_rate")


def compute(log, registry, range_name: str, now: datetime) -> dict:
    where = and_(usage_sql.auto_routed(), usage_sql.in_range(range_name, now))
    generated_at = now.astimezone(timezone.utc).isoformat()

    def query(connection) -> dict:
        qualifying = usage_sql.count(connection, where)
        if qualifying < THRESHOLD:
            return {"enough": False, "range": range_name, "threshold": THRESHOLD,
                    "qualifying_requests": qualifying, "generated_at": generated_at}
        bucket, first, last = usage_sql.bucketing(connection, where, range_name, now)
        totals = usage_sql.totals(connection, where)
        latency = usage_sql.latency_by_model(connection, where, min_requests=LATENCY_MIN_REQUESTS)
        return {"enough": True, "range": range_name, "threshold": THRESHOLD, "qualifying_requests": qualifying,
                "generated_at": generated_at, "bucket": bucket,
                "totals": {k: totals[k] for k in TOTAL_FIELDS},
                "series": usage_sql.series(connection, where, bucket, first, last),
                "tier_mix": usage_sql.shares(connection, where, usage_sql.c.tier),
                "errors_by_kind": usage_sql.errors_by_kind(connection, where),
                "latency_by_model": usage_sql.with_registry(latency, registry),
                "latency_min_requests": LATENCY_MIN_REQUESTS,
                "baseline": "always-strongest", "saved_is_estimate": True, "small_talk_excluded": True}

    return log.run(query)


class PublicStatsCache:
    """{range: (expires_at, result)}. One lock: a burst of requests for an expired range computes it once."""

    def __init__(self, ttl: float = TTL_SECONDS, clock: Callable[[], float] = time.monotonic):
        self.ttl, self.clock = ttl, clock
        self._entries, self._lock = {}, threading.Lock()

    def get(self, range_name: str, compute_fn: Callable[[], dict]) -> dict:
        with self._lock:
            found = self._entries.get(range_name)
            if found is not None and found[0] > self.clock():
                return found[1]
            result = compute_fn()
            self._entries[range_name] = (self.clock() + self.ttl, result)
            return result
