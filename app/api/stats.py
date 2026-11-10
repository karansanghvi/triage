"""Savings summary for GET /v1/stats, from the request log."""
from collections import Counter


def _share(counter: Counter, total: int) -> dict:
    return {k: round(v / total, 4) for k, v in sorted(counter.items())} if total else {}


def summarize(rows: list) -> dict:
    """Costs count only requests whose cost is known (a stream that failed part-way has none).
    strong_cost is the strong model at the same token counts, so `saved` is an estimate."""
    ok = [r for r in rows if r.status == "ok"]
    priced = [r for r in rows if r.price_as_cost is not None and r.strong_cost is not None]
    cost = sum(r.price_as_cost for r in priced)
    strong = sum(r.strong_cost for r in priced)
    return {
        "requests": len(rows),
        "ok": len(ok),
        "errors": len(rows) - len(ok),
        "streams": sum(r.stream for r in rows),
        "priced_requests": len(priced),
        "cost_usd": round(cost, 8),
        "strong_model_cost_usd": round(strong, 8),
        "saved_usd": round(strong - cost, 8),
        "saved_share": round(1 - cost / strong, 4) if strong else None,
        "tier_mix": _share(Counter(r.tier for r in ok if r.tier), len([r for r in ok if r.tier])),
        "model_mix": _share(Counter(r.served_model for r in ok if r.served_model),
                            len([r for r in ok if r.served_model])),
        "cache_hits": sum(bool(r.cache_hit) for r in rows),
        "fallbacks": sum(bool(r.fell_back) for r in rows),
        "saved_is_estimate": True,
    }
