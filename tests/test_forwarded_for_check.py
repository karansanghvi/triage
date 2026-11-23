"""GET /v1/admin/forwarded-for: shows an admin their own request's proxy headers, to set TRIAGE_TRUSTED_PROXY_HOPS
on a host that doesn't document them (Render). Admin only; nothing is logged."""
import logging

from fastapi.testclient import TestClient

from app.api.auth import Auth
from app.api.main import create_app
from tests.api_helpers import ADMIN_KEY, API_KEY, make_deps

PATH = "/v1/admin/forwarded-for"


def _client(hops=1):
    return TestClient(create_app(deps=make_deps(trusted_proxy_hops=hops), auth=Auth([API_KEY], [ADMIN_KEY]),
                                 public=True, accounts=False))


def test_an_admin_sees_the_chain_and_the_address_the_api_would_use(caplog):
    headers = {"Authorization": f"Bearer {ADMIN_KEY}", "X-Forwarded-For": "203.0.113.7, 198.51.100.2",
               "CF-Connecting-IP": "203.0.113.7"}
    with caplog.at_level(logging.DEBUG), _client(hops=2) as client:
        response = client.get(PATH, headers=headers)
    body = response.json()
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    assert body["x_forwarded_for"] == ["203.0.113.7", "198.51.100.2"]
    assert body["trusted_proxy_hops"] == 2 and body["client_ip_used"] == "203.0.113.7"
    assert body["other_headers"] == {"cf-connecting-ip": "203.0.113.7"}
    assert "203.0.113.7" not in caplog.text                       # never logged


def test_with_one_hop_the_rightmost_entry_is_used():
    with _client(hops=1) as client:
        body = client.get(PATH, headers={"Authorization": f"Bearer {ADMIN_KEY}",
                                         "X-Forwarded-For": "1.2.3.4, 203.0.113.9"}).json()
    assert body["client_ip_used"] == "203.0.113.9"


def test_only_an_admin_key_can_read_it():
    with _client() as client:
        assert client.get(PATH).status_code == 401
        assert client.get(PATH, headers={"Authorization": f"Bearer {API_KEY}"}).status_code == 403
