"""Savings summary for GET /v1/stats, computed in SQL (app/api/usage_sql.py).

Costs count only requests whose cost is known (a stream that failed part-way has none). strong_cost is the strong
model at the same token counts, so saved is an estimate. Savings follow usage_sql's rule: only model="auto"
requests are credited (credited_requests), so with named models in the mix saved_usd is not
strong_model_cost_usd - cost_usd. Small talk (greetings routed to the cheapest tier without the classifier) is
left out of every number and counted on its own: it would inflate the savings the classifier is credited with.
"""
from sqlalchemy import and_

from app.api import usage_sql


def summarize(connection, where) -> dict:
    counted = and_(where, usage_sql.not_small_talk())
    totals = usage_sql.totals(connection, counted)
    small_talk = usage_sql.count(connection, and_(where, usage_sql.c.routed_by == "small_talk"))
    return {
        "requests": totals["requests"], "ok": totals["ok"], "errors": totals["errors"], "stopped": totals["stopped"],
        "streams": totals["streams"],
        "priced_requests": totals["priced_requests"], "cost_usd": totals["cost_usd"],
        "credited_requests": totals["credited_requests"],
        "strong_model_cost_usd": totals["strong_model_cost_usd"], "saved_usd": totals["saved_usd"],
        "saved_share": totals["saved_share"],
        "tier_mix": usage_sql.shares(connection, counted, usage_sql.c.tier),
        "model_mix": usage_sql.shares(connection, counted, usage_sql.c.served_model),
        "cache_hits": totals["cache_hits"], "fallbacks": totals["fallbacks"],
        "saved_is_estimate": True,
        "savings_credited_to": 'model="auto" requests only',
        "small_talk_requests": small_talk,
    }
