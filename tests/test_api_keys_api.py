"""Account API keys over HTTP (/v1/account/api-keys) and, in later sections, trk_ keys on the rest of the API.
No network: an in-memory SQLite database."""
import logging

import pytest
from fastapi.testclient import TestClient

from app import db
from app.accounts import credentials
from app.api.auth import Auth
from app.api.main import create_app
from app.api.request_log import SQLRequestLog
from tests.api_helpers import ADMIN_KEY, API_KEY, make_deps

PASSWORD = "correct horse battery"


def app(engine=None, accounts=True, **kw):
    engine = engine or db.open_engine("sqlite://")
    deps = make_deps(engine=engine, log=SQLRequestLog(engine), trusted_proxy_hops=1, **kw)
    return TestClient(create_app(deps=deps, auth=Auth([API_KEY], [ADMIN_KEY]), accounts=accounts)), deps


def session(http, email="ada@example.com", n=1):
    r = http.post("/v1/auth/signup", json={"email": email, "password": PASSWORD},
                  headers={"X-Forwarded-For": f"203.0.113.{n}"})
    return {"Authorization": f"Bearer {r.json()['session_token']}"}


def create(http, auth, name="laptop"):
    return http.post("/v1/account/api-keys", json={"name": name}, headers=auth)


# ---------- create, list, revoke ----------

def test_create_shows_the_full_key_once_and_lists_never_do():
    http, _ = app()
    auth = session(http)
    r = create(http, auth, "  my laptop  ")
    assert r.status_code == 201 and r.headers["cache-control"] == "no-store"
    body = r.json()
    assert set(body) == {"id", "name", "public_id", "created_at", "key", "note"}
    assert credentials.looks_like_api_key(body["key"]) and body["public_id"] == body["key"][:12]
    assert body["name"] == "my laptop" and "only time" in body["note"]
    listed = http.get("/v1/account/api-keys", headers=auth)
    assert listed.json() == [{"id": body["id"], "name": "my laptop", "public_id": body["public_id"],
                              "created_at": body["created_at"], "last_used_at": None, "revoked_at": None}]
    assert body["key"] not in listed.text and body["key"][12:] not in listed.text


def test_list_is_newest_first_and_keeps_revoked_keys():
    http, _ = app()
    auth = session(http)
    ids = [create(http, auth, name).json()["id"] for name in ("first", "second", "third")]
    assert http.delete(f"/v1/account/api-keys/{ids[0]}", headers=auth).status_code == 204
    listed = http.get("/v1/account/api-keys", headers=auth).json()
    assert [k["name"] for k in listed] == ["third", "second", "first"]
    assert listed[2]["revoked_at"] is not None and listed[0]["revoked_at"] is None


def test_revoking_twice_is_harmless_and_unknown_or_other_users_keys_are_404():
    http, _ = app()
    ada, bob = session(http), session(http, "bob@example.com", 2)
    key_id = create(http, ada).json()["id"]
    for _ in range(2):
        assert http.delete(f"/v1/account/api-keys/{key_id}", headers=ada).status_code == 204
    bobs = create(http, bob).json()["id"]
    for path_id in (bobs, "no-such-id", "00000000-0000-0000-0000-000000000000"):
        r = http.delete(f"/v1/account/api-keys/{path_id}", headers=ada)
        assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
        assert r.json() == http.delete("/v1/account/api-keys/no-such-id", headers=ada).json()   # identical
    assert http.get("/v1/account/api-keys", headers=bob).json()[0]["revoked_at"] is None          # untouched


@pytest.mark.parametrize("name", ["", "   ", "x" * 61])
def test_a_name_is_required_and_at_most_sixty_characters(name):
    http, deps = app()
    r = create(http, session(http), name)
    assert r.status_code == 400 and r.json()["error"]["param"] == "name"


def test_a_missing_name_or_extra_field_is_400():
    http, _ = app()
    auth = session(http)
    assert http.post("/v1/account/api-keys", json={}, headers=auth).status_code == 400
    assert http.post("/v1/account/api-keys", json={"name": "a", "key": "trk_mine"}, headers=auth).status_code == 400


def test_at_most_twenty_active_keys():
    http, deps = app()
    auth = session(http)
    from app.accounts.api_keys import MAX_ACTIVE_KEYS
    user_id = http.get("/v1/auth/me", headers=auth).json()["id"]
    for i in range(MAX_ACTIVE_KEYS):                       # straight to the store: the creation limit is 10/hour
        deps.api_keys.create(user_id, f"k{i}")
    r = create(http, auth)
    assert r.status_code == 409 and r.json()["error"]["code"] == "too_many_api_keys" and "Revoke" in r.text
    first = http.get("/v1/account/api-keys", headers=auth).json()[-1]["id"]
    http.delete(f"/v1/account/api-keys/{first}", headers=auth)
    assert create(http, auth).status_code == 201


def test_creation_is_limited_to_ten_per_user_per_hour():
    http, _ = app()
    ada, bob = session(http), session(http, "bob@example.com", 2)
    for i in range(10):
        assert create(http, ada, f"k{i}").status_code == 201
    r = create(http, ada)
    assert r.status_code == 429 and r.json()["error"]["code"] == "too_many_attempts"
    assert 0 < int(r.headers["retry-after"]) <= 3600
    assert create(http, bob).status_code == 201
    assert http.get("/v1/account/api-keys", headers=ada).status_code == 200        # listing isn't limited


def test_deleting_the_account_deletes_its_api_keys():
    engine = db.open_engine("sqlite://")
    http, _ = app(engine)
    ada, bob = session(http), session(http, "bob@example.com", 2)
    create(http, ada)
    create(http, bob)
    assert http.request("DELETE", "/v1/account", json={"password": PASSWORD}, headers=ada).status_code == 204
    with engine.connect() as c:
        assert len(c.execute(db.api_keys_table.select()).all()) == 1


# ---------- session only ----------

@pytest.mark.parametrize("method, path", [("get", "/v1/account/api-keys"), ("post", "/v1/account/api-keys"),
                                          ("delete", "/v1/account/api-keys/some-id")])
def test_the_endpoints_need_a_session_token_not_any_api_key(method, path):
    http, _ = app()
    auth = session(http)
    trk = create(http, auth).json()["key"]
    for headers in ({}, {"Authorization": f"Bearer {API_KEY}"}, {"Authorization": f"Bearer {trk}"}):
        r = http.request(method.upper(), path, json={"name": "x"}, headers=headers)
        assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_session"


def test_every_account_endpoint_refuses_a_trk_key():
    http, _ = app()
    trk = {"Authorization": f"Bearer {create(http, session(http)).json()['key']}"}
    for method, path in (("get", "/v1/auth/me"), ("post", "/v1/auth/logout"), ("patch", "/v1/account"),
                         ("post", "/v1/account/password"), ("delete", "/v1/account"),
                         ("get", "/v1/account/provider-keys"), ("put", "/v1/account/provider-keys/openai")):
        r = http.request(method.upper(), path, json={}, headers=trk)
        assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_session", path


def test_accounts_off_means_no_api_key_endpoints():
    http, _ = app(accounts=False)
    assert http.get("/v1/account/api-keys").status_code == 404
