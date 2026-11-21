"""SQL aggregations over the request log, for the stats endpoints (Phase 5 dashboard).

Everything is computed in the database (SQLite locally, Postgres in production): no endpoint loads every row.
Each function takes a connection and a WHERE condition on the requests table, so the endpoints only decide
whose rows they mean (everyone's auto-routed requests, one account's, one key's).

Savings rule (one place): a request is credited with savings only if Triage chose its model, i.e.
model="auto" and not small talk (greetings skip the classifier). A named model's "savings" are the caller's
choice, not Triage's, and small talk would inflate what the classifier is credited with. Savings are always
strong_cost - price_as_cost, recomputed here, never the stored `saved` column: rows written before this rule
have `saved` filled in for named models too.

Rows from before migration 0008 have routed_by NULL. Small talk arrived in that same migration, so NULL never
means small talk.

Time: timestamps are UTC ISO 8601 text, so text order is time order and substr() gives the day ("2026-11-19")
or month ("2026-11") on both databases. A "7d" range is today and the 6 full UTC days before it.

Percentiles are nearest-rank (the value at rank ceil(p * n)), from ROW_NUMBER() and COUNT() window functions:
both databases have those, while percentile_cont exists only on Postgres.
"""
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional

from sqlalchemy import and_, case, func, or_, select, true

from app import db

RANGES = {"7d": 7, "30d": 30, "all": None}
DEFAULT_RANGE = "30d"
DAY_BUCKETS_UP_TO_DAYS = 92          # an "all" range longer than this is bucketed by month
MONEY_DIGITS, SHARE_DIGITS, MS_DIGITS = 8, 4, 1

requests_table = db.requests_table
c = requests_table.c


class RangeError(ValueError):
    pass


# ---------- conditions ----------

def parse_range(value: Optional[str]) -> str:
    """"7d", "30d" or "all"; None means the default (30d). Anything else is a RangeError."""
    if value is None:
        return DEFAULT_RANGE
    if value not in RANGES:
        raise RangeError(f"range must be one of {', '.join(RANGES)}")
    return value


def range_start(range_name: str, now: datetime) -> Optional[datetime]:
    """Midnight UTC of the range's first day; None for "all"."""
    days = RANGES[range_name]
    if days is None:
        return None
    first = now.astimezone(timezone.utc).date() - timedelta(days=days - 1)
    return datetime.combine(first, time(0), tzinfo=timezone.utc)


def in_range(range_name: str, now: datetime):
    start = range_start(range_name, now)
    return true() if start is None else c.timestamp >= start.isoformat()


def not_small_talk():
    return or_(c.routed_by.is_(None), c.routed_by != "small_talk")


def auto_routed():
    """Triage chose the model: model="auto" and not small talk. Also the savings rule (see the docstring)."""
    return and_(c.requested_model == "auto", not_small_talk())


def _priced():
    return and_(c.price_as_cost.isnot(None), c.strong_cost.isnot(None))


def _credited():
    return and_(_priced(), auto_routed())


def _count_if(condition):
    return func.coalesce(func.sum(case((condition, 1), else_=0)), 0)


def _sum_if(condition, value):
    return func.coalesce(func.sum(case((condition, value), else_=0.0)), 0.0)


def _money(x) -> float:
    return round(float(x or 0.0), MONEY_DIGITS)


def _savings(strong, credited_cost) -> dict:
    strong, credited_cost = float(strong or 0.0), float(credited_cost or 0.0)
    return {"strong_model_cost_usd": _money(strong), "saved_usd": _money(strong - credited_cost),
            "saved_share": round(1 - credited_cost / strong, SHARE_DIGITS) if strong else None}


# ---------- aggregates ----------

def count(connection, where) -> int:
    return int(connection.execute(select(func.count()).select_from(requests_table).where(where)).scalar() or 0)


def totals(connection, where) -> dict:
    """Counts and money. cost_usd: every priced request. strong_model_cost_usd, saved_usd, saved_share: only the
    credited ones (credited_requests; see the savings rule), so saved_usd is not strong - cost_usd when named
    models are in `where`. fallback_rate: among requests whose fallback is known (a model was tried).
    requests = ok + errors + stopped: a stopped request (the client went away part-way) isn't an error, but its
    estimated cost is in the money like any other priced request."""
    credited = _credited()
    found = connection.execute(select(
        func.count(), _count_if(c.status == "ok"), _count_if(c.status == "stopped"), _count_if(c.stream.is_(True)),
        _count_if(c.cache_hit.is_(True)), _count_if(c.fell_back.is_(True)), _count_if(c.fell_back.isnot(None)),
        _count_if(_priced()), _sum_if(_priced(), c.price_as_cost),
        _count_if(credited), _sum_if(credited, c.strong_cost), _sum_if(credited, c.price_as_cost),
    ).select_from(requests_table).where(where)).one()
    requests, ok, stopped, streams, cache_hits, fallbacks, tried, priced, cost, credited_n, strong, credited_cost = found
    return {"requests": int(requests), "ok": int(ok), "errors": int(requests) - int(ok) - int(stopped),
            "stopped": int(stopped), "streams": int(streams),
            "cache_hits": int(cache_hits), "fallbacks": int(fallbacks),
            "fallback_rate": round(int(fallbacks) / int(tried), SHARE_DIGITS) if tried else None,
            "priced_requests": int(priced), "cost_usd": _money(cost), "credited_requests": int(credited_n),
            **_savings(strong, credited_cost)}


def shares(connection, where, column) -> dict:
    """{value: share} of `column` among answered requests (status ok) where it is set."""
    condition = and_(where, c.status == "ok", column.isnot(None))
    found = connection.execute(select(column, func.count()).where(condition).group_by(column)).all()
    total = sum(n for _, n in found)
    return {value: round(n / total, SHARE_DIGITS) for value, n in sorted(found)} if total else {}


def errors_by_kind(connection, where) -> dict:
    kind = func.coalesce(c.error_kind, "unknown")
    found = connection.execute(select(kind, func.count()).where(and_(where, c.status == "error")).group_by(kind)).all()
    return {k: int(n) for k, n in sorted(found)}


def latency_by_model(connection, where, min_requests: int = 1) -> list:
    """[{model, requests, p50_ms, p95_ms}], busiest first. Answered requests with a latency, cache hits left out
    (they take about 0 ms and would drag the percentiles down). Models with fewer than min_requests such
    requests are left out."""
    condition = and_(where, c.status == "ok", c.latency_ms.isnot(None), c.served_model.isnot(None),
                     or_(c.cache_hit.is_(None), c.cache_hit.is_(False)))
    ranked = select(c.served_model.label("model"), c.latency_ms.label("ms"),
                    func.row_number().over(partition_by=c.served_model, order_by=c.latency_ms).label("rn"),
                    func.count().over(partition_by=c.served_model).label("n")).where(condition).subquery()
    r = ranked.c

    def nearest_rank(percent: int):          # rn >= ceil(p * n), in integers: rn * 100 >= percent * n
        return func.min(case((r.rn * 100 >= r.n * percent, r.ms)))

    found = connection.execute(select(r.model, func.max(r.n), nearest_rank(50), nearest_rank(95))
                               .group_by(r.model).having(func.max(r.n) >= min_requests)).all()
    out = [{"model": model, "requests": int(n), "p50_ms": round(float(p50), MS_DIGITS),
            "p95_ms": round(float(p95), MS_DIGITS)} for model, n, p50, p95 in found]
    return sorted(out, key=lambda m: (-m["requests"], m["model"]))


def with_registry(items: list, registry, name_key: str = "model") -> list:
    """Adds provider and tier from the model registry; "unknown" for a model no longer in it."""
    for item in items:
        model = registry.get(item[name_key]) if item[name_key] is not None else None
        item["provider"] = model.provider if model is not None else "unknown"
        item["tier"] = model.tier if model is not None else "unknown"
    return items


def grouped(connection, where, column) -> list:
    """[{key, requests, cost_usd, saved_usd, last_request_at}] per value of `column` (None included), busiest
    first. saved_usd follows the savings rule."""
    credited = _credited()
    found = connection.execute(select(
        column, func.count(), _sum_if(_priced(), c.price_as_cost), _sum_if(credited, c.strong_cost),
        _sum_if(credited, c.price_as_cost), func.max(c.timestamp),
    ).where(where).group_by(column)).all()
    out = [{"key": key, "requests": int(n), "cost_usd": _money(cost), "saved_usd": _money(strong - credited_cost),
            "last_request_at": last} for key, n, cost, strong, credited_cost, last in found]
    return sorted(out, key=lambda g: (-g["requests"], str(g["key"])))


# ---------- time buckets ----------

def bucketing(connection, where, range_name: str, now: datetime) -> tuple:
    """(bucket, first, last): "day" or "month", and the first and last day to show. For "all": from the first
    request in `where` (None, None when there is none), by month when that is over DAY_BUCKETS_UP_TO_DAYS ago."""
    today = now.astimezone(timezone.utc).date()
    start = range_start(range_name, now)
    if start is not None:
        return "day", start.date(), today
    first = connection.execute(select(func.min(c.timestamp)).where(where)).scalar()
    if first is None:
        return "day", None, None
    first_day = date.fromisoformat(first[:10])
    return ("day" if (today - first_day).days <= DAY_BUCKETS_UP_TO_DAYS else "month"), first_day, today


def _labels(bucket: str, first: date, last: date) -> list:
    if bucket == "day":
        return [(first + timedelta(days=i)).isoformat() for i in range((last - first).days + 1)]
    out, year, month = [], first.year, first.month
    while (year, month) <= (last.year, last.month):
        out.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


def series(connection, where, bucket: str, first: Optional[date], last: Optional[date]) -> list:
    """[{start, requests, cost_usd, strong_model_cost_usd, saved_usd}] for every bucket from first to last,
    empty ones included (zeros), so a chart's x axis is continuous."""
    if first is None:
        return []
    key = func.substr(c.timestamp, 1, 10 if bucket == "day" else 7)
    credited = _credited()
    found = connection.execute(select(
        key, func.count(), _sum_if(_priced(), c.price_as_cost), _sum_if(credited, c.strong_cost),
        _sum_if(credited, c.price_as_cost),
    ).where(where).group_by(key)).all()
    by_key = {k: (int(n), cost, strong, credited_cost) for k, n, cost, strong, credited_cost in found}
    out = []
    for label in _labels(bucket, first, last):
        n, cost, strong, credited_cost = by_key.get(label, (0, 0.0, 0.0, 0.0))
        out.append({"start": label, "requests": n, "cost_usd": _money(cost),
                    "strong_model_cost_usd": _money(strong), "saved_usd": _money(strong - credited_cost)})
    return out


# ---------- one page of requests ----------

PAGE_COLUMNS = ("request_id", "timestamp", "key_id", "requested_model", "served_model", "tier", "routed_by",
                "status", "http_status", "error_kind", "stream", "cache_hit", "fell_back", "latency_ms", "ttft_ms")


def page(connection, where, before_id: Optional[int], limit: int) -> tuple:
    """(items, next_before_id): newest first by the table's id (keyset pagination: a cursor never skips or
    repeats a row, even while new ones arrive). Only PAGE_COLUMNS plus cost and the credited saving are read;
    never text, prompt_hmac or the vectors. next_before_id is None on the last page."""
    credited = _credited()
    condition = where if before_id is None else and_(where, c.id < before_id)
    found = connection.execute(select(
        c.id, *[c[name] for name in PAGE_COLUMNS], c.price_as_cost,
        case((credited, c.strong_cost - c.price_as_cost), else_=None),
    ).where(condition).order_by(c.id.desc()).limit(limit + 1)).all()
    items = []
    for values in found[:limit]:
        item = dict(zip(PAGE_COLUMNS, values[1:1 + len(PAGE_COLUMNS)]))
        cost, saved = values[-2], values[-1]
        item["cost_usd"] = None if cost is None else _money(cost)
        item["saved_usd"] = None if saved is None else _money(saved)
        for flag in ("stream", "cache_hit", "fell_back"):
            item[flag] = None if item[flag] is None else bool(item[flag])
        items.append(item)
    next_before = found[limit - 1][0] if len(found) > limit else None
    return items, next_before

