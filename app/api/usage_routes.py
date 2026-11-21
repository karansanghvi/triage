"""An account's own usage, for the dashboard's "My usage" tab (Phase 5). Only when TRIAGE_ACCOUNTS is on.

    GET /v1/account/usage?range=7d|30d|all      totals, time buckets, tier mix, by model / provider / API key,
                                                errors by kind, latency per model
    GET /v1/account/requests?cursor=&limit=     recent requests, newest first (limit 1-100, default 50)

Either a session token (trs_, the website) or one of the account's API keys (trk_) works; both see every request
of the account, whichever key made it. A Triage server key (TRIAGE_API_KEYS) is not an account: 401.

No prompt or answer text exists to return (the log never stores it), and neither endpoint reads prompt_hmac or
the classifier vectors. Savings follow app/api/usage_sql.py's rule: only model="auto" requests that the classifier
routed are credited; named models and small talk are counted, with no saving.
"""
import base64
import binascii
from datetime import datetime, timezone
from typing import Optional

import anyio
from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import JSONResponse
from sqlalchemy import and_

from app.api import usage_sql
from app.api.auth import Caller
from app.api.errors import APIError, bad_request

WEBSITE_CHAT_KEY_ID, WEBSITE_CHAT_LABEL = "session", "Website chat"
DEFAULT_PAGE, MAX_PAGE = 50, 100
CURSOR_PREFIX = "v1:"
NEEDS_ACCOUNT = ("This endpoint needs an account: send a session token or one of your account's API keys (trk_...) "
                 "as Authorization: Bearer <token>.")
PRIVATE = {"Cache-Control": "no-store"}


def encode_cursor(before_id: int) -> str:
    return base64.urlsafe_b64encode(f"{CURSOR_PREFIX}{before_id}".encode()).decode().rstrip("=")


def decode_cursor(cursor: str) -> int:
    try:
        text = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)).decode()
        if not text.startswith(CURSOR_PREFIX):
            raise ValueError(text)
        before_id = int(text[len(CURSOR_PREFIX):])
        if before_id < 1:
            raise ValueError(text)
        return before_id
    except (ValueError, UnicodeDecodeError, binascii.Error):
        raise bad_request("cursor is not one this endpoint returned.", "cursor", "invalid_cursor") from None


def _iso(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    return value.astimezone(timezone.utc).isoformat()


def _later(a: Optional[str], b: Optional[str]) -> Optional[str]:
    found = [datetime.fromisoformat(x) for x in (a, b) if x is not None]
    return max(found).isoformat() if found else None


def key_labels(keys: list) -> dict:
    """{key_id in the log: (label, revoked)}. Account keys are logged by their public id."""
    labels = {WEBSITE_CHAT_KEY_ID: (WEBSITE_CHAT_LABEL, False)}
    for k in keys:
        labels[k.public_id] = (k.name, k.revoked_at is not None)
    return labels


def by_key(groups: list, keys: list) -> list:
    """Every key with requests in the range, plus active keys without any (requests 0). last_used: the newest of
    the key's last request in the range and its own last_used_at."""
    known = {k.public_id: k for k in keys}
    labels = key_labels(keys)
    out, seen = [], set()
    for g in groups:
        key_id = g["key"]
        seen.add(key_id)
        label, revoked = labels.get(key_id, (key_id, None))
        key = known.get(key_id)
        out.append({"key_id": key_id, "label": label,
                    "kind": "website_chat" if key_id == WEBSITE_CHAT_KEY_ID else "api_key", "revoked": revoked,
                    "requests": g["requests"], "cost_usd": g["cost_usd"], "saved_usd": g["saved_usd"],
                    "last_used": _later(_iso(g["last_request_at"]), _iso(key.last_used_at) if key else None)})
    for k in keys:
        if k.public_id not in seen and k.revoked_at is None:
            out.append({"key_id": k.public_id, "label": k.name, "kind": "api_key", "revoked": False, "requests": 0,
                        "cost_usd": 0.0, "saved_usd": 0.0, "last_used": _iso(k.last_used_at)})
    return out


def by_provider(models: list) -> list:
    totals = {}
    for m in models:
        t = totals.setdefault(m["provider"], {"provider": m["provider"], "requests": 0, "cost_usd": 0.0,
                                              "saved_usd": 0.0})
        t["requests"] += m["requests"]
        t["cost_usd"] = round(t["cost_usd"] + m["cost_usd"], usage_sql.MONEY_DIGITS)
        t["saved_usd"] = round(t["saved_usd"] + m["saved_usd"], usage_sql.MONEY_DIGITS)
    return sorted(totals.values(), key=lambda t: (-t["requests"], t["provider"]))


def usage(log, registry, keys: list, user_id: str, range_name: str, now: datetime) -> dict:
    c = usage_sql.c
    where = and_(c.user_id == user_id, usage_sql.in_range(range_name, now))

    def query(connection) -> dict:
        bucket, first, last = usage_sql.bucketing(connection, where, range_name, now)
        totals = usage_sql.totals(connection, where)
        totals["small_talk_requests"] = usage_sql.count(connection, and_(where, c.routed_by == "small_talk"))
        models = [{"model": g["key"], "requests": g["requests"], "cost_usd": g["cost_usd"],
                   "saved_usd": g["saved_usd"]} for g in usage_sql.grouped(connection, where, c.served_model)]
        return {"range": range_name, "bucket": bucket, "generated_at": now.isoformat(), "totals": totals,
                "series": usage_sql.series(connection, where, bucket, first, last),
                "tier_mix": usage_sql.shares(connection, where, c.tier),
                "by_model": usage_sql.with_registry(models, registry),
                "by_key_groups": usage_sql.grouped(connection, where, c.key_id),
                "errors_by_kind": usage_sql.errors_by_kind(connection, where),
                "latency_by_model": usage_sql.with_registry(usage_sql.latency_by_model(connection, where), registry)}

    out = log.run(query)
    out["by_provider"] = by_provider(out["by_model"])
    out["by_key"] = by_key(out.pop("by_key_groups"), keys)
    out |= {"saved_is_estimate": True, "baseline": "always-strongest",
            "savings_credited_to": 'model="auto" requests routed by the classifier (not small talk)'}
    return out


def recent(log, keys: list, user_id: str, before_id: Optional[int], limit: int) -> dict:
    items, next_before = log.run(lambda connection: usage_sql.page(connection, usage_sql.c.user_id == user_id,
                                                                   before_id, limit))
    labels = key_labels(keys)
    for item in items:
        item["key_label"] = labels.get(item["key_id"], (item["key_id"], None))[0]
    return {"data": items, "next_cursor": None if next_before is None else encode_cursor(next_before)}


def usage_router(account_caller) -> APIRouter:
    """account_caller(authorization) -> Optional[Caller]: main.py's check of trs_ / trk_ tokens."""
    router = APIRouter()

    async def account(authorization: Optional[str] = Header(None)) -> Caller:
        who = await account_caller(authorization)
        if who is None:
            raise APIError(401, NEEDS_ACCOUNT, code="invalid_api_key")
        return who

    @router.get("/v1/account/usage")
    async def account_usage(request: Request, range: Optional[str] = None, who: Caller = Depends(account)):
        try:
            range_name = usage_sql.parse_range(range)
        except usage_sql.RangeError as error:
            raise bad_request(str(error), "range", "invalid_value") from None
        deps, now = request.app.state.deps, datetime.now(timezone.utc)

        def work():
            keys = deps.api_keys.list(who.user_id)
            return usage(deps.log, deps.registry, keys, who.user_id, range_name, now)
        return JSONResponse(await anyio.to_thread.run_sync(work), headers=PRIVATE)

    @router.get("/v1/account/requests")
    async def account_requests(request: Request, cursor: Optional[str] = None, limit: Optional[str] = None,
                               who: Caller = Depends(account)):
        if limit is None:
            size = DEFAULT_PAGE
        elif not limit.isdigit() or not 1 <= int(limit) <= MAX_PAGE:
            raise bad_request(f"limit must be a whole number from 1 to {MAX_PAGE}.", "limit", "invalid_value")
        else:
            size = int(limit)
        before_id = None if cursor is None else decode_cursor(cursor)
        deps = request.app.state.deps

        def work():
            return recent(deps.log, deps.api_keys.list(who.user_id), who.user_id, before_id, size)
        return JSONResponse(await anyio.to_thread.run_sync(work), headers=PRIVATE)

    return router
