"""app/api/usage_sql.py: the stats aggregations, computed in SQL (the in-memory log is SQLite too)."""
import math
import random
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import true

from app.api import usage_sql
from app.api.request_log import InMemoryRequestLog, RequestRow
from app.registry import load_registry

NOW = datetime(2026, 11, 19, 15, 30, tzinfo=timezone.utc)
_ids = iter(range(10**6))


def row(days_ago: float = 0, **kw) -> RequestRow:
    base = dict(request_id=f"r{next(_ids)}", timestamp=(NOW - timedelta(days=days_ago)).isoformat(), key_id="k",
                requested_model="auto", stream=False, status="ok", routed_by="classifier", tier="medium",
                served_model="groq-small", fell_back=False, cache_hit=False, price_as_cost=1.0, strong_cost=3.0,
                saved=2.0, latency_ms=100.0)
    base.update(kw)
    return RequestRow(**base)


def log_with(*rows) -> InMemoryRequestLog:
    log = InMemoryRequestLog()
    for r in rows:
        log.write(r)
    return log


def q(log, fn, where=None):
    return log.run(lambda connection: fn(connection, true() if where is None else where))


# ---------- the savings rule ----------

def test_only_auto_routed_requests_are_credited_with_savings():
    log = log_with(row(), row(requested_model="groq-small"),           # named model: its cost counts, no saving
                   row(routed_by="small_talk", tier="low"),            # small talk: no saving
                   row(routed_by=None))                                # before migration 0008: classifier
    t = q(log, usage_sql.totals)
    assert t["requests"] == 4 and t["priced_requests"] == 4 and t["cost_usd"] == 4.0
    assert t["credited_requests"] == 2
    assert t["strong_model_cost_usd"] == 6.0 and t["saved_usd"] == 4.0 and t["saved_share"] == pytest.approx(2 / 3, abs=1e-4)


def test_savings_are_recomputed_never_read_from_the_saved_column():
    log = log_with(row(saved=999.0), row(requested_model="groq-large", saved=50.0))   # stale stored values
    assert q(log, usage_sql.totals)["saved_usd"] == 2.0


def test_unpriced_requests_count_but_add_no_money():
    log = log_with(row(), row(price_as_cost=None, strong_cost=None, status="error", error_kind="timeout"),
                   row(strong_cost=None))                              # no high-tier model for these providers
    t = q(log, usage_sql.totals)
    assert (t["requests"], t["ok"], t["errors"], t["priced_requests"], t["credited_requests"]) == (3, 2, 1, 1, 1)
    assert t["cost_usd"] == 1.0 and t["saved_usd"] == 2.0


def test_an_empty_log_has_zero_totals_and_no_shares():
    t = q(InMemoryRequestLog(), usage_sql.totals)
    assert t["requests"] == 0 and t["saved_share"] is None and t["fallback_rate"] is None and t["cost_usd"] == 0.0


def test_fallback_rate_counts_only_requests_that_tried_a_model():
    log = log_with(row(fell_back=True), row(), row(), row(fell_back=True),
                   row(fell_back=None, status="error", error_kind="invalid_request"))
    t = q(log, usage_sql.totals)
    assert t["fallbacks"] == 2 and t["fallback_rate"] == 0.5


def test_auto_routed_condition_matches_the_rule():
    log = log_with(row(), row(routed_by=None), row(routed_by="small_talk"), row(requested_model="groq-small"),
                   row(requested_model="groq-small", routed_by="requested_model"))
    assert q(log, usage_sql.count, usage_sql.auto_routed()) == 2


# ---------- ranges and buckets ----------

def test_ranges_start_at_midnight_utc_of_the_first_day():
    assert usage_sql.range_start("7d", NOW).isoformat() == "2026-11-13T00:00:00+00:00"
    assert usage_sql.range_start("30d", NOW).isoformat() == "2026-10-21T00:00:00+00:00"
    assert usage_sql.range_start("all", NOW) is None


def test_parse_range():
    assert usage_sql.parse_range(None) == "30d" and usage_sql.parse_range("all") == "all"
    for bad in ("", "7", "90d", "ALL"):
        with pytest.raises(usage_sql.RangeError):
            usage_sql.parse_range(bad)


def test_range_filter_keeps_rows_from_the_first_days_midnight():
    start = usage_sql.range_start("7d", NOW)
    log = log_with(*[RequestRow(request_id=f"t{i}", timestamp=ts, key_id="k", requested_model="auto", stream=False)
                     for i, ts in enumerate([(start - timedelta(microseconds=1)).isoformat(), start.isoformat(),
                                             (start + timedelta(seconds=1, microseconds=5)).isoformat()])])
    assert q(log, usage_sql.count, usage_sql.in_range("7d", NOW)) == 2
    assert q(log, usage_sql.count, usage_sql.in_range("all", NOW)) == 3


def test_daily_series_is_continuous_and_sums_to_the_totals():
    log = log_with(row(0), row(0), row(2), row(2, requested_model="groq-small"), row(10))
    where = usage_sql.in_range("7d", NOW)

    def both(connection, w):
        bucket, first, last = usage_sql.bucketing(connection, w, "7d", NOW)
        return (bucket, first, last), usage_sql.series(connection, w, bucket, first, last), usage_sql.totals(connection, w)
    (bucket, first, last), points, t = q(log, both, where)
    assert bucket == "day" and [p["start"] for p in points] == [f"2026-11-{d}" for d in range(13, 20)]
    assert [p["requests"] for p in points] == [0, 0, 0, 0, 2, 0, 2]
    assert sum(p["requests"] for p in points) == t["requests"] == 4
    assert sum(p["saved_usd"] for p in points) == t["saved_usd"] == 6.0        # the named model saves nothing
    assert sum(p["cost_usd"] for p in points) == t["cost_usd"] == 4.0


def test_all_uses_days_up_to_92_days_and_months_beyond():
    def bucketing(log):
        return log.run(lambda connection: usage_sql.bucketing(connection, true(), "all", NOW))
    assert bucketing(InMemoryRequestLog()) == ("day", None, None)
    assert bucketing(log_with(row(92)))[0] == "day"
    bucket, first, last = bucketing(log_with(row(93), row(1)))
    assert bucket == "month"
    points = log_with(row(93), row(1)).run(lambda connection: usage_sql.series(connection, true(), bucket, first, last))
    assert [p["start"] for p in points] == ["2026-08", "2026-09", "2026-10", "2026-11"]
    assert [p["requests"] for p in points] == [1, 0, 0, 1]


def test_month_labels_cross_a_year():
    from datetime import date
    assert usage_sql._labels("month", date(2025, 11, 30), date(2026, 2, 1)) == ["2025-11", "2025-12", "2026-01", "2026-02"]


# ---------- mixes, errors, latency ----------

def test_tier_shares_count_answered_requests_only():
    log = log_with(row(tier="low"), row(tier="medium"), row(tier="medium"), row(tier="high"),
                   row(tier="high", status="error", error_kind="timeout"), row(tier=None))
    assert q(log, lambda conn, w: usage_sql.shares(conn, w, usage_sql.c.tier)) == {"high": 0.25, "low": 0.25, "medium": 0.5}


def test_errors_by_kind():
    log = log_with(row(), row(status="error", error_kind="timeout"), row(status="error", error_kind="timeout"),
                   row(status="error", error_kind="rate_limit"), row(status="error", error_kind=None))
    assert q(log, usage_sql.errors_by_kind) == {"rate_limit": 1, "timeout": 2, "unknown": 1}


def nearest_rank(values, p):
    ordered = sorted(values)
    return ordered[math.ceil(p * len(ordered)) - 1]


@pytest.mark.parametrize("n", [1, 2, 19, 20, 21, 100, 137])
def test_latency_percentiles_are_nearest_rank(n):
    rng = random.Random(n)
    small = [round(rng.uniform(50, 5000), 1) for _ in range(n)]
    large = [round(rng.uniform(50, 9000), 1) for _ in range(n + 3)]
    log = log_with(*[row(latency_ms=v) for v in small], *[row(served_model="groq-large", latency_ms=v) for v in large])
    found = {m["model"]: m for m in q(log, usage_sql.latency_by_model)}
    for model, values in (("groq-small", small), ("groq-large", large)):
        assert found[model]["requests"] == len(values)
        assert found[model]["p50_ms"] == nearest_rank(values, 0.50)
        assert found[model]["p95_ms"] == nearest_rank(values, 0.95)


def test_latency_leaves_out_cache_hits_errors_and_quiet_models():
    log = log_with(*[row(latency_ms=100.0) for _ in range(3)], row(latency_ms=0.1, cache_hit=True),
                   row(latency_ms=9000.0, status="error", error_kind="timeout"), row(latency_ms=None),
                   row(served_model="groq-large", latency_ms=500.0))
    found = q(log, lambda conn, w: usage_sql.latency_by_model(conn, w, min_requests=2))
    assert found == [{"model": "groq-small", "requests": 3, "p50_ms": 100.0, "p95_ms": 100.0}]


def test_with_registry_adds_provider_and_tier():
    registry = load_registry()
    items = usage_sql.with_registry([{"model": "groq-small"}, {"model": "retired-model"}, {"model": None}], registry)
    assert items[0]["provider"] == registry["groq-small"].provider and items[0]["tier"] == registry["groq-small"].tier
    assert items[1]["provider"] == items[1]["tier"] == "unknown"
    assert items[2]["provider"] == items[2]["tier"] == "unknown"


def test_grouped_by_key():
    log = log_with(row(key_id="a", days_ago=3), row(key_id="a", days_ago=1, requested_model="groq-small"),
                   row(key_id="b", days_ago=2), row(key_id="session"))
    found = {g["key"]: g for g in q(log, lambda conn, w: usage_sql.grouped(conn, w, usage_sql.c.key_id))}
    assert found["a"]["requests"] == 2 and found["a"]["cost_usd"] == 2.0 and found["a"]["saved_usd"] == 2.0
    assert found["a"]["last_request_at"] == (NOW - timedelta(days=1)).isoformat()
    assert set(found) == {"a", "b", "session"}


# ---------- pages ----------

def test_pages_walk_every_row_once_newest_first():
    log = log_with(*[row(request_id=f"p{i}") for i in range(7)])
    seen, before = [], None
    while True:
        items, before = log.run(lambda connection: usage_sql.page(connection, true(), before, 3))
        seen += [i["request_id"] for i in items]
        if before is None:
            break
    assert seen == [f"p{i}" for i in reversed(range(7))]


def test_a_page_has_no_text_hash_or_vectors_and_credits_savings_by_the_rule():
    log = log_with(row(request_id="named", requested_model="groq-small", prompt_hmac="abc", features=[1.0] * 14),
                   row(request_id="auto", prompt_hmac="def"))
    items, before = log.run(lambda connection: usage_sql.page(connection, true(), None, 10))
    assert before is None and [i["request_id"] for i in items] == ["auto", "named"]
    assert all(not {"prompt_hmac", "features", "embedding", "id"} & set(i) for i in items)
    assert items[0]["saved_usd"] == 2.0 and items[1]["saved_usd"] is None and items[1]["cost_usd"] == 1.0
    assert items[0]["stream"] is False and items[0]["cache_hit"] is False


def test_an_exactly_full_last_page_has_no_next_cursor():
    log = log_with(*[row() for _ in range(3)])
    items, before = log.run(lambda connection: usage_sql.page(connection, true(), None, 3))
    assert len(items) == 3 and before is None
