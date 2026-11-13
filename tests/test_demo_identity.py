"""Demo visitor identity: client_ip (X-Forwarded-For with trusted proxy hops) and visitor_id (keyed hash). No network."""
import pytest
from starlette.requests import Request

from app.api.demo import UNKNOWN_CLIENT, client_ip, trusted_hops_from_env, visitor_id

SECRET = b"test-hash-secret"


def request(xff=None, client=("10.0.0.1", 50000), xff_headers=None):
    headers = []
    for value in (xff_headers or ([xff] if xff is not None else [])):
        headers.append((b"x-forwarded-for", value.encode()))
    return Request({"type": "http", "method": "POST", "path": "/", "headers": headers, "client": client})


# ---------- client_ip ----------

def test_zero_hops_uses_the_direct_connection_and_ignores_the_header():
    assert client_ip(request(), 0) == "10.0.0.1"
    assert client_ip(request("1.2.3.4"), 0) == "10.0.0.1"          # anyone can send X-Forwarded-For


def test_one_hop_uses_the_address_our_proxy_appended():
    assert client_ip(request("203.0.113.9"), 1) == "203.0.113.9"


def test_one_hop_a_faked_header_cant_choose_the_address():
    # the client sent "X-Forwarded-For: 1.1.1.1"; our proxy appended the real address it saw
    assert client_ip(request("1.1.1.1, 203.0.113.9"), 1) == "203.0.113.9"
    assert client_ip(request("6.6.6.6, 7.7.7.7, 203.0.113.9"), 1) == "203.0.113.9"


def test_two_hops_take_the_second_from_the_right():
    assert client_ip(request("1.1.1.1, 203.0.113.9, 10.1.1.1"), 2) == "203.0.113.9"


def test_too_few_entries_falls_back_to_the_connection():
    assert client_ip(request(), 1) == "10.0.0.1"                    # no header: the request skipped the proxy
    assert client_ip(request("203.0.113.9"), 2) == "10.0.0.1"


def test_repeated_headers_and_spaces_are_combined_in_order():
    assert client_ip(request(xff_headers=["1.1.1.1", " 203.0.113.9 "]), 1) == "203.0.113.9"
    assert client_ip(request("1.1.1.1,,  ,203.0.113.9"), 1) == "203.0.113.9"


def test_no_client_at_all():
    assert client_ip(request(client=None), 0) == UNKNOWN_CLIENT


@pytest.mark.parametrize("value, hops", [(None, 0), ("", 0), ("0", 0), ("1", 1), (" 2 ", 2)])
def test_trusted_hops_from_env(monkeypatch, value, hops):
    if value is None:
        monkeypatch.delenv("TRIAGE_TRUSTED_PROXY_HOPS", raising=False)
    else:
        monkeypatch.setenv("TRIAGE_TRUSTED_PROXY_HOPS", value)
    assert trusted_hops_from_env() == hops


@pytest.mark.parametrize("value", ["-1", "two"])
def test_bad_trusted_hops_are_rejected(monkeypatch, value):
    monkeypatch.setenv("TRIAGE_TRUSTED_PROXY_HOPS", value)
    with pytest.raises(ValueError):
        trusted_hops_from_env()


# ---------- visitor_id ----------

def test_visitor_id_is_a_16_char_keyed_hash_without_the_ip():
    vid = visitor_id("203.0.113.9", SECRET)
    assert len(vid) == 16 and int(vid, 16) >= 0
    assert "203" not in vid and vid == visitor_id("203.0.113.9", SECRET)
    assert vid != visitor_id("203.0.113.9", b"another-secret")
    assert vid != visitor_id("203.0.113.10", SECRET)


def test_ipv6_addresses_in_one_64_are_one_visitor():
    a = visitor_id("2001:db8:1:2:aaaa::1", SECRET)
    assert a == visitor_id("2001:db8:1:2:ffff:ffff:ffff:ffff", SECRET)
    assert a == visitor_id("[2001:db8:1:2::9]", SECRET)
    assert a != visitor_id("2001:db8:1:3::1", SECRET)


def test_ipv4_mapped_ipv6_is_the_ipv4_visitor():
    assert visitor_id("::ffff:203.0.113.9", SECRET) == visitor_id("203.0.113.9", SECRET)


def test_non_ip_clients_still_get_an_id():
    assert len(visitor_id("testclient", SECRET)) == 16
    assert visitor_id("testclient", SECRET) != visitor_id(UNKNOWN_CLIENT, SECRET)
