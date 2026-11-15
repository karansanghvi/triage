"""Checking a provider key against its list-models endpoint (app/accounts/verify_keys.py). No network: every
request goes to an httpx MockTransport."""
import httpx
import pytest

from app.accounts.verify_keys import ENDPOINTS, verify_key

SECRET = "sk-test-SECRET123"


def client(handler, seen=None):
    def record(request):
        if seen is not None:
            seen.append(request)
        return handler(request)
    return httpx.Client(transport=httpx.MockTransport(record))


def answer(status, body=None):
    return lambda request: httpx.Response(status, json=body if body is not None else {})


@pytest.mark.parametrize("provider, url, header, value", [
    ("openai", "https://api.openai.com/v1/models", "authorization", f"Bearer {SECRET}"),
    ("anthropic", "https://api.anthropic.com/v1/models", "x-api-key", SECRET),
    ("groq", "https://api.groq.com/openai/v1/models", "authorization", f"Bearer {SECRET}"),
    ("gemini", "https://generativelanguage.googleapis.com/v1beta/models", "x-goog-api-key", SECRET),
])
def test_one_get_to_each_providers_list_models_with_the_key_in_a_header(provider, url, header, value):
    seen = []
    assert verify_key(provider, SECRET, client(answer(200, {"data": []}), seen)).status == "valid"
    (request,) = seen
    assert request.method == "GET" and str(request.url) == url and request.headers[header] == value
    assert SECRET not in str(request.url)                          # never in the URL (Gemini included)
    if provider == "anthropic":
        assert request.headers["anthropic-version"]


def test_a_200_is_valid_and_answered():
    v = verify_key("openai", SECRET, client(answer(200)))
    assert (v.status, v.detail, v.answered) == ("valid", None, True)


@pytest.mark.parametrize("status", [401, 403])
def test_401_and_403_are_invalid(status):
    v = verify_key("groq", SECRET, client(answer(status, {"error": {"message": "Invalid API Key"}})))
    assert (v.status, v.answered) == ("invalid", True) and v.detail == f"Groq answered {status}: Invalid API Key"


def test_geminis_bad_key_400_is_invalid():
    body = {"error": {"code": 400, "message": "API key not valid. Please pass a valid API key.",
                      "status": "INVALID_ARGUMENT", "details": [{"reason": "API_KEY_INVALID"}]}}
    v = verify_key("gemini", SECRET, client(answer(400, body)))
    assert v.status == "invalid" and v.detail.startswith("Gemini answered 400: API key not valid")


@pytest.mark.parametrize("status", [400, 402, 404, 429, 500, 502, 503])
def test_other_answers_are_unverified(status):
    v = verify_key("anthropic", SECRET, client(answer(status, {"error": {"message": "try later"}})))
    assert (v.status, v.answered) == ("unverified", False) and v.detail == f"Anthropic answered {status}: try later"


def test_a_timeout_is_unverified():
    def slow(request):
        raise httpx.ReadTimeout("timed out", request=request)
    v = verify_key("openai", SECRET, client(slow))
    assert (v.status, v.detail, v.answered) == ("unverified", "OpenAI didn't answer within 10 s", False)


def test_a_network_error_is_unverified_with_only_its_type():
    def down(request):
        raise httpx.ConnectError(f"connection failed while sending {request.headers['authorization']}",
                                 request=request)
    v = verify_key("openai", SECRET, client(down))
    assert v.detail == "couldn't reach OpenAI (ConnectError)" and SECRET not in v.detail


@pytest.mark.parametrize("status", [401, 500])
def test_a_provider_echoing_the_key_is_redacted(status):
    body = {"error": {"message": f"Incorrect API key provided: {SECRET}. Find your key at the dashboard."}}
    v = verify_key("openai", SECRET, client(answer(status, body)))
    assert SECRET not in v.detail and "SECRET123" not in v.detail and "[redacted]" in v.detail


def test_odd_keys_that_dont_look_like_keys_are_still_redacted():
    odd = "my-own-odd-key-value-42"
    v = verify_key("groq", odd, client(answer(401, {"error": {"message": f"bad key {odd}"}})))
    assert odd not in v.detail


def test_details_are_short_and_non_json_bodies_are_fine():
    v = verify_key("groq", SECRET, client(answer(401, {"error": {"message": "x" * 1000}})))
    assert len(v.detail) <= 160
    v = verify_key("groq", SECRET, client(lambda r: httpx.Response(502, text="<html>Bad gateway</html>")))
    assert v.detail == "Groq answered 502"


def test_the_endpoints_are_the_documented_ones():
    assert ENDPOINTS == {"openai": "https://api.openai.com/v1/models",
                         "anthropic": "https://api.anthropic.com/v1/models",
                         "groq": "https://api.groq.com/openai/v1/models",
                         "gemini": "https://generativelanguage.googleapis.com/v1beta/models"}


def test_the_verdict_carries_the_providers_status():
    assert verify_key("gemini", SECRET, client(answer(403))).http_status == 403
    assert verify_key("gemini", SECRET, client(answer(200))).http_status == 200
    def slow(request):
        raise httpx.ReadTimeout("timed out", request=request)
    assert verify_key("gemini", SECRET, client(slow)).http_status is None
