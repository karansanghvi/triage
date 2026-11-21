"""Housekeeping (Phase 5, privacy policy): what the policy promises about how long data is kept.

    1. the response cache (a local SQLite file): expired answers (older than the cache's 24-hour lifetime) are
       deleted, not just ignored. Hourly.
    2. the request log (the database): rows older than TRIAGE_REQUEST_LOG_RETENTION_DAYS (default 365) are deleted,
       with their labels, in small batches. Daily: every run wakes a serverless database (Neon scales to zero after
       5 idle minutes and bills each wake for at least that long), and a day's delay doesn't change "deleted after
       12 months".

Both run once at startup (the database is awake then anyway: the deploy just ran its migrations). Each step is
separate and never raises: a failure is logged (step and error type only) and the next hourly pass tries it again,
so housekeeping can't take the server down.
"""
import asyncio
import functools
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import anyio

logger = logging.getLogger("triage.maintenance")

RETENTION_ENV = "TRIAGE_REQUEST_LOG_RETENTION_DAYS"
DEFAULT_RETENTION_DAYS = 365
INTERVAL_SECONDS = 60 * 60                  # the loop's pass, and the cache purge
RETENTION_INTERVAL_SECONDS = 24 * 60 * 60   # the request log's retention (the database)
STEPS = ("cache", "retention")
BATCH_SIZE = 1000


def retention_days_from_env(environ=os.environ) -> int:
    """TRIAGE_REQUEST_LOG_RETENTION_DAYS: a whole number of days above 0 (default 365). Anything else refuses to
    start: a typo must not silently keep data longer, or delete it sooner, than the privacy policy says."""
    raw = (environ.get(RETENTION_ENV) or "").strip()
    if not raw:
        return DEFAULT_RETENTION_DAYS
    try:
        days = int(raw)
    except ValueError:
        days = 0
    if days <= 0:
        raise RuntimeError(f"refusing to start: {RETENTION_ENV} must be a whole number of days above 0")
    return days


def retention_cutoff(days: int, now: datetime) -> str:
    """The oldest timestamp kept, in the request log's own format (UTC ISO 8601)."""
    return (now - timedelta(days=days)).isoformat()


def run_once(deps, retention_days: int, clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
             batch_size: int = BATCH_SIZE, steps=STEPS) -> dict:
    """One housekeeping pass over `steps` ("cache", "retention"). Blocking (database): call it from a worker
    thread. Returns what each step did: a count, or "error:<type>". Never raises."""
    out = {}
    cache = getattr(getattr(deps, "cached_sender", None), "cache", None)
    if cache is not None and "cache" in steps:
        try:
            out["cache_expired_deleted"] = cache.purge_expired()
        except Exception as error:          # e.g. the cache file is locked or unreadable
            out["cache_expired_deleted"] = f"error:{type(error).__name__}"
            logger.warning("maintenance: cache purge failed (%s)", type(error).__name__)
    if "retention" not in steps:
        logger.info("maintenance: %s", out)
        return out
    try:
        cutoff = retention_cutoff(retention_days, clock())
        out["request_rows_deleted"] = deps.log.delete_older_than(cutoff, batch_size=batch_size)
    except Exception as error:              # e.g. the database is unreachable
        out["request_rows_deleted"] = f"error:{type(error).__name__}"
        logger.warning("maintenance: request log retention failed (%s)", type(error).__name__)
    logger.info("maintenance: %s", out)
    return out


async def run_forever(deps, retention_days: int, interval: float = INTERVAL_SECONDS,
                      retention_interval: float = RETENTION_INTERVAL_SECONDS, sleep=asyncio.sleep,
                      passes: Optional[int] = None, monotonic: Callable[[], float] = time.monotonic) -> None:
    """A pass at startup, then every `interval` seconds, until cancelled (at shutdown). Each pass purges the cache;
    retention runs in the first pass and then once `retention_interval` has passed since it last succeeded (a
    failed run is tried again next pass). passes: stop after that many (tests). Nothing escapes except
    cancellation."""
    done, retention_ok_at = 0, None
    while passes is None or done < passes:
        due = retention_ok_at is None or monotonic() - retention_ok_at >= retention_interval
        steps = STEPS if due else ("cache",)
        try:
            out = await anyio.to_thread.run_sync(functools.partial(run_once, deps, retention_days, steps=steps))
            if due and not str(out.get("request_rows_deleted", "")).startswith("error:"):
                retention_ok_at = monotonic()
        except asyncio.CancelledError:
            raise
        except Exception as error:          # run_once doesn't raise; this is the last resort
            logger.warning("maintenance: pass failed (%s)", type(error).__name__)
        done += 1
        if passes is not None and done >= passes:
            return
        await sleep(interval)
