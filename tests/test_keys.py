import pytest
from starlette.datastructures import Headers

from app.api.headers import resolve_keys
from app.providers.keys import KeyContext

SECRET = "sk-test-SECRET123"


def test_no_headers_is_demo():
    ctx = resolve_keys({})
    assert ctx.mode == "demo"
    assert ctx.providers == frozenset()


def test_one_key_is_byok_with_only_that_provider():
    ctx = resolve_keys({"x-groq-key": "gsk_abc"})
    assert ctx.mode == "byok"
    assert ctx.providers == {"groq"}
    assert ctx.key_for("groq") == "gsk_abc"
    assert ctx.key_for("gemini") is None


def test_multiple_keys():
    ctx = resolve_keys({"x-groq-key": "g1", "x-gemini-key": "m1"})
    assert ctx.providers == {"groq", "gemini"}


def test_blank_or_whitespace_header_counts_as_missing():
    ctx = resolve_keys({"x-groq-key": "   ", "x-gemini-key": ""})
    assert ctx.mode == "demo"
    assert ctx.providers == frozenset()


def test_value_is_trimmed():
    ctx = resolve_keys({"x-groq-key": "  gsk_abc  "})
    assert ctx.key_for("groq") == "gsk_abc"


def test_unrelated_headers_are_ignored():
    ctx = resolve_keys({"authorization": "Bearer triage-key", "x-triage-quality": "0.8"})
    assert ctx.mode == "demo"


def test_real_headers_are_case_insensitive():
    ctx = resolve_keys(Headers({"X-Groq-Key": "gsk_abc"}))
    assert ctx.key_for("groq") == "gsk_abc"


def test_key_never_appears_when_printed():
    ctx = KeyContext("byok", {"openai": SECRET})
    for text in (repr(ctx), str(ctx), f"{ctx}", f"failed for {ctx!r}"):
        assert SECRET not in text
    assert "openai=***" in repr(ctx)


def test_keys_cannot_be_changed_after_creation():
    ctx = KeyContext("byok", {"groq": "g1"})
    with pytest.raises(TypeError):
        ctx.keys["gemini"] = "m1"