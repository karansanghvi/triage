"""GET /v1/account/usage and /v1/account/requests: an account's own usage (app/api/usage_routes.py)."""
from datetime import datetime, timedelta, timezone

import pytest

from app.api import usage_routes
from app.api.request_log import RequestRow
from app.registry import load_registry
from tests.api_helpers import API_KEY
from tests.test_account_mode import SAVED_GROQ, app, chat, save, signup, trk

_ids = iter(range(10**6))


def row(user_id, days_ago: float = 0, **kw) -> RequestRow:
    now = datetime.now(timezone.utc)
    base = dict(request_id=f"u{next(_ids)}", timestamp=(now - timedelta(days=days_ago)).isoformat(), key_id="session",
                requested_model="auto", stream=False, status="ok", routed_by="classifier", tier="medium",
                served_model="groq-small", fell_back=False, cache_hit=False, price_as_cost=1.0, strong_cost=3.0,
                latency_ms=100.0, user_id=user_id, mode="account", prompt_hmac="secret-hash")
    base.update(kw)
    return RequestRow(**base)


def keys(http, auth):
    return {k["name"]: k for k in http.get("/v1/account/api-keys", headers=auth).json()}


# ---------- who can call them ----------

def test_a_session_or_the_accounts_trk_key_works_and_a_server_key_doesnt():
    http, deps = app()
    ada, _ = signup(http)
    ada_key = trk(http, ada)
    for credential in (ada, ada_key):
        for path in ("/v1/account/usage", "/v1/account/requests"):
            r = http.get(path, headers=credential)
            assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    for credential in ({"Authorization": f"Bearer {API_KEY}"}, {}, {"Authorization": "Bearer trk_nope"}):
        for path in ("/v1/account/usage", "/v1/account/requests"):
            assert http.get(path, headers=credential).status_code == 401


def test_the_endpoints_dont_exist_without_accounts():
    from fastapi.testclient import TestClient
    from app.api.auth import Auth
    from app.api.main import create_app
    from tests.api_helpers import make_deps
    http = TestClient(create_app(deps=make_deps(), auth=Auth([API_KEY]), accounts=False))
    assert http.get("/v1/account/usage").status_code == 404
    assert http.get("/v1/account/requests").status_code == 404


# ---------- usage ----------

def test_usage_end_to_end_only_sees_the_accounts_own_requests():
    http, deps = app()
    ada, _ = signup(http)
    bob, _ = signup(http, "bob@example.com", 2)
    for auth in (ada, bob):
        save(http, auth, "groq", SAVED_GROQ)
    ada_key = trk(http, ada)
    chat(http, ada_key)
    chat(http, ada)
    chat(http, ada, {"model": "groq-small", "messages": [{"role": "user", "content": "Explain B-trees."}]})
    chat(http, bob)
    usage = http.get("/v1/account/usage", headers=ada_key).json()
    t = usage["totals"]
    assert t["requests"] == 3 and t["credited_requests"] == 2                # the named model isn't credited
    by_key = {k["kind"] if k["kind"] == "website_chat" else k["label"]: k for k in usage["by_key"]}
    assert by_key["website_chat"]["requests"] == 2 and by_key["website_chat"]["label"] == "Website chat"
    assert by_key["laptop"]["requests"] == 1 and by_key["laptop"]["revoked"] is False
    assert sum(m["requests"] for m in usage["by_model"]) == 3
    assert sum(p["requests"] for p in usage["by_provider"]) == 3
    assert http.get("/v1/account/usage", headers=bob).json()["totals"]["requests"] == 1


def test_usage_figures_by_key_model_and_provider():
    http, deps = app()
    ada, ada_id = signup(http)
    trk(http, ada, "laptop")
    trk(http, ada, "old")
    trk(http, ada, "unused")
    listed = keys(http, ada)
    http.delete(f"/v1/account/api-keys/{listed['old']['id']}", headers=ada)
    laptop, old = listed["laptop"]["public_id"], listed["old"]["public_id"]
    registry = load_registry()
    for r in [row(ada_id, 1, key_id=laptop), row(ada_id, 2, key_id=laptop, requested_model="groq-large",
                                                   routed_by="requested_model", served_model="groq-large", tier="high"),
              row(ada_id, 3, key_id=old), row(ada_id, 0, routed_by="small_talk", tier="low", served_model="groq-allam"),
              row(ada_id, 0, status="error", error_kind="rate_limit", served_model=None, latency_ms=None,
                  price_as_cost=None, strong_cost=None),
              row(ada_id, 0, served_model="retired-model"), row(ada_id, 40)]:          # 40 days: outside 30d
        deps.log.write(r)
    usage = http.get("/v1/account/usage", headers=ada).json()
    assert usage["range"] == "30d" and usage["bucket"] == "day" and len(usage["series"]) == 30
    t = usage["totals"]
    assert (t["requests"], t["errors"], t["credited_requests"], t["small_talk_requests"]) == (6, 1, 3, 1)
    assert t["cost_usd"] == 5.0 and t["saved_usd"] == 6.0
    assert usage["errors_by_kind"] == {"rate_limit": 1}
    by_key = {k["label"]: k for k in usage["by_key"]}
    assert set(by_key) == {"laptop", "old", "unused", "Website chat"}
    assert (by_key["laptop"]["requests"], by_key["laptop"]["cost_usd"], by_key["laptop"]["saved_usd"]) == (2, 2.0, 2.0)
    assert by_key["old"]["revoked"] is True and by_key["old"]["requests"] == 1
    assert by_key["unused"]["requests"] == 0 and by_key["unused"]["last_used"] is None
    models = {m["model"]: m for m in usage["by_model"]}
    assert models["groq-large"]["saved_usd"] == 0.0 and models["groq-large"]["tier"] == registry["groq-large"].tier
    assert models["retired-model"]["provider"] == "unknown" and models[None]["requests"] == 1
    providers = {p["provider"]: p["requests"] for p in usage["by_provider"]}
    assert providers["unknown"] == 2 and sum(providers.values()) == 6
    latency = {m["model"]: m for m in usage["latency_by_model"]}
    assert latency["groq-small"]["p50_ms"] == 100.0 and latency["groq-small"]["tier"] == registry["groq-small"].tier
    assert http.get("/v1/account/usage?range=all", headers=ada).json()["totals"]["requests"] == 7
    assert http.get("/v1/account/usage?range=7d", headers=ada).json()["totals"]["requests"] == 6
    r = http.get("/v1/account/usage?range=1y", headers=ada)
    assert r.status_code == 400 and r.json()["error"]["param"] == "range"


def test_last_used_is_the_newer_of_the_last_request_and_the_keys_own_last_use():
    class Key:
        def __init__(self, public_id, last_used_at):
            self.public_id, self.name, self.revoked_at, self.last_used_at = public_id, public_id, None, last_used_at
    t = datetime(2026, 11, 19, 12, tzinfo=timezone.utc)
    groups = [{"key": "a", "requests": 1, "cost_usd": 0.0, "saved_usd": 0.0, "last_request_at": (t - timedelta(days=2)).isoformat()},
              {"key": "b", "requests": 1, "cost_usd": 0.0, "saved_usd": 0.0, "last_request_at": t.isoformat()}]
    found = {k["key_id"]: k["last_used"] for k in usage_routes.by_key(groups, [Key("a", t - timedelta(hours=1)),
                                                                                Key("b", t - timedelta(days=5))])}
    assert found == {"a": (t - timedelta(hours=1)).isoformat(), "b": t.isoformat()}


# ---------- recent requests ----------

def test_recent_requests_page_newest_first_without_hashes_or_text():
    http, deps = app()
    ada, ada_id = signup(http)
    bob, bob_id = signup(http, "bob@example.com", 2)
    trk(http, ada, "laptop")
    laptop = keys(http, ada)["laptop"]["public_id"]
    for i in range(5):
        deps.log.write(row(ada_id, days_ago=0, request_id=f"a{i}", key_id=laptop if i % 2 else "session"))
    deps.log.write(row(bob_id, request_id="bobs"))
    deps.log.write(row(ada_id, request_id="named", requested_model="groq-small", routed_by="requested_model"))
    seen, cursor = [], None
    while True:
        r = http.get("/v1/account/requests", params={"limit": 3, **({"cursor": cursor} if cursor else {})}, headers=ada)
        assert r.status_code == 200 and "secret-hash" not in r.text and "prompt_hmac" not in r.text
        body = r.json()
        seen += body["data"]
        cursor = body["next_cursor"]
        if cursor is None:
            break
    assert [i["request_id"] for i in seen] == ["named", "a4", "a3", "a2", "a1", "a0"]
    first = seen[0]
    assert first["saved_usd"] is None and first["cost_usd"] == 1.0
    assert seen[1]["saved_usd"] == 2.0 and seen[1]["key_label"] == "Website chat" and seen[2]["key_label"] == "laptop"
    assert {"timestamp", "served_model", "tier", "latency_ms", "status", "routed_by", "key_label"} <= set(first)
    assert not {"prompt_hmac", "features", "embedding", "user_id", "id"} & set(first)


@pytest.mark.parametrize("params, param", [({"limit": "0"}, "limit"), ({"limit": "101"}, "limit"),
                                           ({"limit": "ten"}, "limit"), ({"cursor": "not-a-cursor"}, "cursor"),
                                           ({"cursor": usage_routes.encode_cursor(0)}, "cursor")])
def test_bad_limits_and_cursors_are_400(params, param):
    http, _ = app()
    ada, _ = signup(http)
    r = http.get("/v1/account/requests", params=params, headers=ada)
    assert r.status_code == 400 and r.json()["error"]["param"] == param


def test_cursors_round_trip():
    for n in (1, 42, 10**12):
        assert usage_routes.decode_cursor(usage_routes.encode_cursor(n)) == n
