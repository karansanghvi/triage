"""BYOK (bring your own key), Phase 5: per-request keys through the whole request path. No network.

SECRET is a user key; it must reach only the provider it belongs to, and never a log, a cache key, an error
body, a background task or the shared circuit breaker.
"""
import json
import logging
import random

import httpx
import httpx2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.auth import Auth
from app.api.fake_models import FakeBehavior
from app.api.main import create_app
from app.api.request_log import SQLiteRequestLog
from app.cache import CachedSender, ResponseCache, key_fingerprint, make_cache_key
from app.providers import anthropic_provider, gemini_provider, openai_provider
from app.providers.base import send_request
from app.providers.common import ProviderError
from app.providers.keys import REDACTED, KeyContext, redact
from app.registry import load_registry
from app.router.predictor import Prediction
from app.reliability import AllModelsFailed, BYOKKeyMissing, ReliableClient, load_routing
from tests.api_helpers import API_KEY, ASK, FakePredictor, make_deps

SECRET = "sk-test-SECRET123"
REGISTRY = load_registry()
MESSAGES = [{"role": "user", "content": "Hi"}]


class Wire:
    def __init__(self, body, lib=httpx, ctype="application/json"):
        self.body, self.lib, self.ctype, self.requests = body, lib, ctype, []

    def __call__(self, request):
        self.requests.append(request)
        return self.lib.Response(200, content=json.dumps(self.body), headers={"content-type": self.ctype})

    def client(self):
        return self.lib.Client(transport=self.lib.MockTransport(self))


OPENAI_REPLY = {"id": "c", "object": "chat.completion", "created": 0, "model": "m",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
ANTHROPIC_REPLY = {"id": "m", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
                   "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "stop_sequence": None,
                   "usage": {"input_tokens": 1, "output_tokens": 1}}
GEMINI_REPLY = {"candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2}}


@pytest.fixture
def server_keys(monkeypatch):
    """Server env keys that must NOT be used when a user key is given."""
    for env in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(env, "server-env-key")


# ---------- a: each adapter sends the user's key ----------

@pytest.mark.parametrize("model, module, reply, lib, header, prefix", [
    ("openai-luna", openai_provider, OPENAI_REPLY, httpx, "authorization", "Bearer "),
    ("groq-small", openai_provider, OPENAI_REPLY, httpx, "authorization", "Bearer "),
    ("claude-haiku", anthropic_provider, ANTHROPIC_REPLY, httpx2, "x-api-key", ""),
    ("gemini-flash", gemini_provider, GEMINI_REPLY, httpx, "x-goog-api-key", ""),
])
def test_adapter_sends_the_given_key_not_the_env_key(server_keys, monkeypatch, model, module, reply, lib, header,
                                                     prefix):
    wire = Wire(reply, lib)
    monkeypatch.setattr(module, "HTTP_CLIENT", wire.client())
    r = send_request(MESSAGES, REGISTRY[model], max_tokens=100, api_key=SECRET)
    assert r.text == "ok"
    (request,) = wire.requests
    assert request.headers[header] == prefix + SECRET
    assert "server-env-key" not in str(request.headers)


@pytest.mark.parametrize("model, module, reply, lib, header, prefix", [
    ("groq-small", openai_provider, OPENAI_REPLY, httpx, "authorization", "Bearer "),
    ("claude-haiku", anthropic_provider, ANTHROPIC_REPLY, httpx2, "x-api-key", ""),
    ("gemini-flash", gemini_provider, GEMINI_REPLY, httpx, "x-goog-api-key", ""),
])
def test_without_a_key_the_adapter_uses_the_env_key_as_before(server_keys, monkeypatch, model, module, reply, lib,
                                                              header, prefix):
    wire = Wire(reply, lib)
    monkeypatch.setattr(module, "HTTP_CLIENT", wire.client())
    send_request(MESSAGES, REGISTRY[model], max_tokens=100)
    assert wire.requests[0].headers[header] == prefix + "server-env-key"


def test_an_empty_given_key_is_an_error_not_the_env_key(server_keys):
    with pytest.raises(ProviderError) as error:
        send_request(MESSAGES, REGISTRY["groq-small"], max_tokens=100, api_key="")
    assert error.value.kind == "config"


def test_ollama_ignores_a_key(monkeypatch):
    from app.providers import ollama_provider
    seen = {}
    monkeypatch.setattr(ollama_provider, "send_openai_format", lambda *a, **kw: seen.update(kw) or "ok")
    ollama_provider.send(MESSAGES, "llama3.2:3b", api_key=SECRET, stream=False, max_tokens=1, timeout=1)
    assert seen["api_key"] == "ollama"


# ---------- b: the KeyContext travels with the request ----------


GROQ_ONLY = {"x-groq-key": SECRET}


def app(script=None, predictor=None, **kw):
    deps = make_deps(script, predictor=predictor or FakePredictor(0.4, 0.95), **kw)
    return TestClient(create_app(deps=deps, auth=Auth([API_KEY]))), deps


def chat(http, body=None, headers=None):
    return http.post("/v1/chat/completions", json=body or {"model": "auto", "messages": ASK},
                     headers={"Authorization": f"Bearer {API_KEY}"} | (headers or {}))


@pytest.mark.parametrize("stream", [False, True])
def test_the_users_key_reaches_the_provider(stream):
    http, deps = app()
    r = chat(http, {"model": "auto", "messages": ASK, "stream": stream}, GROQ_ONLY)
    assert r.status_code == 200
    assert deps.fake_models.last_api_key == SECRET
    assert all(c["byok_key"] for c in deps.fake_models.calls)


def test_no_key_request_sends_no_api_key():
    http, deps = app()
    assert chat(http).status_code == 200
    assert deps.fake_models.last_api_key is None and not any(c["byok_key"] for c in deps.fake_models.calls)


def test_the_guard_refuses_a_byok_call_without_its_providers_key():
    deps = make_deps()
    with pytest.raises(BYOKKeyMissing):
        deps.client._call(REGISTRY["claude-opus"], ASK, False, {}, KeyContext("byok", {"groq": SECRET}))


def test_api_key_cant_be_passed_around_the_keys_context():
    deps = make_deps()
    with pytest.raises(ValueError, match="KeyContext"):
        deps.client.send(ASK, "low", api_key=SECRET)
    with pytest.raises(ValueError, match="not both"):
        deps.client.send(ASK, "low", keys=KeyContext("byok", {"groq": SECRET}), providers={"groq"})


def test_keys_are_not_stored_on_deps_or_the_app():
    http, deps = app()
    chat(http, headers=GROQ_ONLY)
    for obj in (deps, deps.client, deps.config, http.app.state):
        assert SECRET not in repr(vars(obj))


# ---------- c: BYOK routing, tier fallback, errors, baseline ----------

DOWN = FakeBehavior(fail_before="server")
QUALITY = {"low": "0.1", "medium": "0.92", "high": "0.99"}        # FakePredictor(0.4, 0.95) -> these tiers


def providers_called(deps):
    return {REGISTRY[c["model"]].provider for c in deps.fake_models.calls}


@pytest.mark.parametrize("tier", ["low", "medium", "high"])
def test_groq_only_byok_routes_only_to_groq_models(tier):
    http, deps = app()
    r = chat(http, headers=GROQ_ONLY | {"X-Triage-Quality": QUALITY[tier]})
    assert r.status_code == 200 and r.headers["x-triage-tier"] == tier
    assert providers_called(deps) == {"groq"}


def test_groq_only_byok_never_falls_back_to_the_local_ollama():
    http, deps = app({"groq-allam": DOWN})
    r = chat(http, headers=GROQ_ONLY | {"X-Triage-Quality": "0.1"})
    assert r.status_code == 503 and providers_called(deps) == {"groq"}


def test_gemini_only_byok_on_a_low_request_escalates_to_a_tier_with_a_gemini_model():
    http, deps = app()
    r = chat(http, headers={"x-gemini-key": SECRET, "X-Triage-Quality": "0.1"})
    assert r.status_code == 200
    assert (r.headers["x-triage-tier"], r.headers["x-triage-model"]) == ("medium", "gemini-flash")
    assert "no model for your keys in low: used medium" in r.headers["x-triage-reason"]
    (row,) = deps.log.rows()
    assert (row.tier, row.served_model, row.primary_model, row.fell_back) == ("medium", "gemini-flash",
                                                                               "gemini-flash", False)


def test_tier_fallback_goes_up_first_then_down():
    from app.api.core import usable_tier
    routing = load_routing(REGISTRY)
    assert usable_tier("low", routing, REGISTRY, {"gemini"}) == "medium"
    assert usable_tier("high", routing, REGISTRY, {"groq"}) == "high"
    assert usable_tier("low", routing, REGISTRY, {"openrouter"}) is None


def test_byok_with_no_reachable_model_is_a_400_naming_the_key_headers():
    from tests.api_helpers import routing_raw
    raw = routing_raw()
    raw["fallback_chains"] = {t: [c[0]] for t, c in raw["fallback_chains"].items()}   # groq models only
    http, deps = app(raw=raw)
    r = chat(http, headers={"x-gemini-key": SECRET})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "no_usable_model"
    assert all(h in err["message"] for h in ("X-OpenAI-Key", "X-Anthropic-Key", "X-Groq-Key", "X-Gemini-Key"))
    assert SECRET not in r.text and deps.fake_models.calls == []


@pytest.mark.parametrize("headers", [{}, GROQ_ONLY])
def test_named_model_without_its_key_is_403_naming_the_header(headers):
    http, deps = app()
    r = chat(http, {"model": "claude-opus", "messages": ASK}, headers)
    assert r.status_code == 403
    err = r.json()["error"]
    assert err["code"] == "model_not_allowed" and "X-Anthropic-Key" in err["message"]
    assert deps.fake_models.calls == []


def test_named_model_with_its_key_is_served():
    http, deps = app()
    r = chat(http, {"model": "claude-opus", "messages": ASK}, {"x-anthropic-key": SECRET})
    assert r.status_code == 200 and r.json()["model"] == "claude-opus"


def test_byok_named_local_model_is_403():
    http, _ = app()
    r = chat(http, {"model": "ollama-llama", "messages": ASK}, GROQ_ONLY)
    assert r.status_code == 403 and "doesn't use" in r.json()["error"]["message"]


@pytest.mark.parametrize("headers, strong", [({"x-anthropic-key": SECRET}, "claude-opus"),
                                             ({"x-gemini-key": SECRET}, "gemini-flash"),
                                             (GROQ_ONLY, "groq-large"),
                                             ({}, "groq-large")])
def test_savings_baseline_is_the_strong_model_among_the_users_providers(headers, strong):
    http, deps = app()
    assert chat(http, headers=headers).status_code == 200
    (row,) = deps.log.rows()
    assert row.strong_cost == pytest.approx(REGISTRY[strong].reported_cost(row.input_tokens, row.output_tokens))
    assert row.saved == pytest.approx(row.strong_cost - row.price_as_cost)


# ---------- d: the shared circuit breaker ignores BYOK calls ----------

AUTH_FAIL = FakeBehavior(fail_before="auth")


def test_byok_auth_failure_doesnt_open_the_breaker_for_a_following_no_key_request():
    http, deps = app({"groq-small": AUTH_FAIL})
    for _ in range(5):                                   # well past failure_threshold (3)
        chat(http, headers=GROQ_ONLY | {"X-Triage-Quality": "0.92"})
    assert not deps.client.breaker.is_open("groq")
    deps.fake_models.script.clear()                      # the server's own groq key works
    r = chat(http, headers={"X-Triage-Quality": "0.92"})
    assert r.status_code == 200 and r.headers["x-triage-model"] == "groq-small"
    assert r.headers["x-triage-fallback"] == "none"


def test_an_open_server_breaker_doesnt_block_a_byok_request():
    http, deps = app({"groq-small": AUTH_FAIL})
    for _ in range(3):
        chat(http, headers={"X-Triage-Quality": "0.92"})            # the server's key fails: groq trips
    assert deps.client.breaker.is_open("groq")
    deps.fake_models.script.clear()
    r = chat(http, headers=GROQ_ONLY | {"X-Triage-Quality": "0.92"})
    assert r.status_code == 200 and r.headers["x-triage-model"] == "groq-small"


def test_byok_success_doesnt_reset_the_servers_failure_count():
    http, deps = app({"groq-small": AUTH_FAIL})
    chat(http, headers={"X-Triage-Quality": "0.92"})
    chat(http, headers={"X-Triage-Quality": "0.92"})
    before = dict(deps.client.breaker._failures)
    deps.fake_models.script.clear()
    chat(http, headers=GROQ_ONLY | {"X-Triage-Quality": "0.92"})
    assert dict(deps.client.breaker._failures) == before


# ---------- e: cache keys carry a key fingerprint, never the key ----------


OTHER_SECRET = "gsk_other-user-key-456"


def cached_app(tmp_path):
    return app(cache=ResponseCache(tmp_path / "cache.sqlite3"))


def test_two_user_keys_with_the_same_prompt_dont_share_a_cache_entry(tmp_path):
    http, deps = cached_app(tmp_path)
    assert chat(http, headers={"x-groq-key": SECRET}).headers["x-triage-cache"] == "miss"
    assert chat(http, headers={"x-groq-key": OTHER_SECRET}).headers["x-triage-cache"] == "miss"
    assert chat(http, headers={"x-groq-key": SECRET}).headers["x-triage-cache"] == "hit"
    assert chat(http).headers["x-triage-cache"] == "miss"            # a no-key request doesn't see BYOK answers
    assert len(deps.fake_models.calls) == 3


def test_the_key_never_enters_the_cache_file(tmp_path):
    http, deps = cached_app(tmp_path)
    chat(http, headers={"x-groq-key": SECRET})
    assert SECRET.encode() not in (tmp_path / "cache.sqlite3").read_bytes()
    keys = [k for (k,) in deps.cached_sender.cache._conn.execute("SELECT key FROM responses")]
    assert keys and all(SECRET not in k for k in keys)


def test_no_fingerprint_gives_todays_cache_key():
    m = REGISTRY["groq-small"]
    assert make_cache_key(ASK, m, 0, 4000, "low") == make_cache_key(ASK, m, 0, 4000, "low", None, None, None)
    assert make_cache_key(ASK, m, 0, 4000, "low", key_fingerprint=key_fingerprint(SECRET, b"s")) != \
        make_cache_key(ASK, m, 0, 4000, "low")


def test_fingerprint_is_a_keyed_16_char_hmac():
    fp = key_fingerprint(SECRET, b"secret-1")
    assert len(fp) == 16 and SECRET not in fp
    assert fp != key_fingerprint(SECRET, b"secret-2") and fp != key_fingerprint(OTHER_SECRET, b"secret-1")


def test_byok_without_a_hash_secret_skips_the_cache(tmp_path):
    calls = []
    sender = CachedSender(ResponseCache(tmp_path / "c.sqlite3"), send_fn=lambda *a, **kw: calls.append(kw) or None)
    sender(ASK, REGISTRY["groq-small"], temperature=0, max_tokens=10, api_key=SECRET)
    assert sender.cache.count() == 0 and sender.hits == sender.misses == 0 and calls[0]["api_key"] == SECRET


# ---------- f: redaction, request mode ----------



ECHO_401 = FakeBehavior(fail_before="auth", echo_key=True)


@pytest.fixture
def captured_logs(caplog):
    """Everything logged: the root (caplog) plus the stderr JSON request log, which doesn't propagate."""
    caplog.set_level(logging.DEBUG)
    lines = []
    handler = logging.Handler()
    handler.emit = lambda record: lines.append(handler.format(record))
    requests_logger = logging.getLogger("triage.requests")
    requests_logger.addHandler(handler)
    yield lambda: caplog.text + "\n".join(lines)
    requests_logger.removeHandler(handler)


@pytest.mark.parametrize("stream", [False, True])
def test_a_key_echoed_by_a_provider_401_appears_nowhere(tmp_path, captured_logs, stream):
    db = tmp_path / "requests.sqlite3"
    http, deps = app({"groq-small": ECHO_401, "groq-allam": ECHO_401, "groq-large": ECHO_401},
                     log=SQLiteRequestLog(db), cache=ResponseCache(tmp_path / "cache.sqlite3"))
    bodies = []
    for quality in ("0.1", "0.92", "0.99"):
        r = chat(http, {"model": "auto", "messages": ASK, "stream": stream}, GROQ_ONLY | {"X-Triage-Quality": quality})
        bodies.append(r.text + str(r.headers))
    r = chat(http, {"model": "groq-small", "messages": ASK, "stream": stream}, GROQ_ONLY)
    bodies.append(r.text + str(r.headers))
    assert all(SECRET not in b for b in bodies)
    assert SECRET not in captured_logs()
    deps.log.close()
    for f in tmp_path.iterdir():                                   # the SQLite log (and WAL), the cache
        assert SECRET.encode() not in f.read_bytes(), f.name
    assert all(row.mode == "byok" for row in SQLiteRequestLog(db).rows())


def test_attempt_messages_are_redacted():
    deps = make_deps({"groq-small": ECHO_401})
    with pytest.raises(AllModelsFailed) as error:
        deps.client.send(ASK, "medium", keys=KeyContext("byok", {"groq": SECRET}))
    (attempt,) = error.value.attempts
    assert attempt.error_kind == "auth" and REDACTED in attempt.message and SECRET not in attempt.message
    assert SECRET not in str(error.value)


def test_send_request_redacts_a_key_echoed_by_the_real_sdk(server_keys, monkeypatch):
    class Echo401(Wire):
        def __call__(self, request):
            self.requests.append(request)
            body = {"error": {"message": f"Incorrect API key provided: {SECRET}.", "type": "invalid_request_error",
                              "code": "invalid_api_key"}}
            return httpx.Response(401, content=json.dumps(body), headers={"content-type": "application/json"})

    wire = Echo401({})
    monkeypatch.setattr(openai_provider, "HTTP_CLIENT", wire.client())
    for stream in (False, True):
        with pytest.raises(ProviderError) as error:
            result = send_request(MESSAGES, REGISTRY["openai-luna"], stream=stream, max_tokens=10, api_key=SECRET)
            list(result)
        assert error.value.kind == "auth" and SECRET not in str(error.value) and error.value.__cause__ is None


@pytest.mark.parametrize("text", ["sk-ant-api03-abcdefghij", "sk-proj-abcdefghijkl", "gsk_abcdefghijkl",
                                  "AIzaSyA1234567890abcdefghij"])
def test_redact_removes_common_key_patterns(text):
    assert redact(f"key {text} was rejected") == f"key {REDACTED} was rejected"


def test_redact_removes_exact_values_and_keeps_ordinary_text():
    assert redact("my-odd-key-value failed", ("my-odd-key-value",)) == f"{REDACTED} failed"
    assert redact("sk-learn and gsk_ are fine") == "sk-learn and gsk_ are fine"
    assert redact(None) is None


def test_byok_rows_are_mode_byok_never_training_eligible_and_store_no_vectors():
    class VectorPredictor(FakePredictor):
        def predict_messages(self, messages, task_type=None):
            p = super().predict_messages(messages, task_type)
            return Prediction(p.p_low, p.p_medium, p.raw_p_medium, p.task_type, p.task_type_guessed,
                              features=np.zeros(14), embedding=np.zeros(384, dtype="float32"))

    http, deps = app(predictor=VectorPredictor(0.95, 0.95))
    chat(http, headers=GROQ_ONLY)
    chat(http)
    byok_row, api_row = deps.log.rows()
    assert (byok_row.mode, byok_row.training_eligible, byok_row.features, byok_row.embedding) == \
        ("byok", False, None, None)
    assert api_row.mode == "api" and api_row.training_eligible is True and api_row.embedding is not None


# ---------- g: sync verification and background labels ----------

BAD_CHECK = '{"good": false, "reason": "wrong"}'
SYNC = {"X-Triage-Verify": "sync"}
GOOD = "2 + 3 = 5.\nAnswer: 5"           # passes the free format check for ASK
# p_medium 0.85: below skip_check_above (0.90), so medium's AI check isn't skipped as "confident"
MEDIUM_UNSURE = FakePredictor(0.4, 0.85)


def calls_of(deps, kind):
    return [c for c in deps.fake_models.calls if c["kind"] == kind]


def test_byok_without_the_checkers_key_skips_only_the_ai_check():
    http, deps = app({"gemini-flash": FakeBehavior(text=GOOD)}, predictor=MEDIUM_UNSURE)
    r = chat(http, headers={"x-gemini-key": SECRET, "X-Triage-Quality": "0.8"} | SYNC)
    assert r.status_code == 200
    assert (r.headers["x-triage-model"], r.headers["x-triage-check"]) == ("gemini-flash", "skipped_no_key")
    assert r.headers["x-triage-verified"] == "sync" and calls_of(deps, "check") == []
    assert providers_called(deps) == {"gemini"}
    (row,) = deps.log.rows()
    assert row.decided_by == "skipped_no_key" and row.escalated is False


def test_byok_checker_uses_the_users_own_key_and_escalates_within_their_providers():
    http, deps = app({"groq-small": FakeBehavior(text=GOOD, check_reply=BAD_CHECK)}, predictor=MEDIUM_UNSURE)
    r = chat(http, headers=GROQ_ONLY | {"X-Triage-Quality": "0.8"} | SYNC)
    assert (r.headers["x-triage-check"], r.headers["x-triage-escalated"]) == ("ai_checker", "true")
    assert r.json()["model"] == "groq-large"
    (check,) = calls_of(deps, "check")
    assert check["byok_key"] and providers_called(deps) == {"groq"}


def test_format_check_still_escalates_without_the_checkers_key():
    http, deps = app({"openai-luna": FakeBehavior(text="five"), "openai-terra": FakeBehavior(text=GOOD)},
                     predictor=FakePredictor(0.95, 0.85))
    r = chat(http, headers={"x-openai-key": SECRET, "X-Triage-Quality": "0.1"} | SYNC)
    assert r.headers["x-triage-escalated"] == "true" and r.json()["model"] == "openai-terra"
    assert calls_of(deps, "check") == [] and providers_called(deps) == {"openai"}   # no groq key: no AI check


def test_byok_escalation_with_no_higher_tier_serves_the_current_answer():
    from tests.api_helpers import routing_raw
    raw = routing_raw()
    raw["fallback_chains"]["high"] = ["groq-large"]                 # no gemini model above medium
    http, deps = app({"gemini-flash": FakeBehavior(text="five")}, raw=raw)
    r = chat(http, headers={"x-gemini-key": SECRET, "X-Triage-Quality": "0.92"} | SYNC)
    assert r.status_code == 200 and r.json()["model"] == "gemini-flash"
    assert r.headers["x-triage-escalated"] == "false"
    (row,) = deps.log.rows()
    assert row.error_kind == "no_model_for_keys:high" and row.decided_by == "format"
    assert row.saved is None and "x-triage-saved" not in r.headers      # no gemini model in high: no baseline


def test_truncated_escalation_attempts_count_in_the_cost():
    http, deps = app({"openai-luna": FakeBehavior(text="five"), "openai-terra": FakeBehavior(truncated=True),
                      "claude-sonnet": FakeBehavior(text=GOOD)}, predictor=FakePredictor(0.95, 0.95))
    r = chat(http, headers={"x-openai-key": SECRET, "x-anthropic-key": "sk-ant-test-OTHER456",
                            "X-Triage-Quality": "0.1"} | SYNC)
    assert r.status_code == 200 and r.json()["model"] == "claude-sonnet"
    (row,) = deps.log.rows()
    assert [c["model"] for c in calls_of(deps, "answer")] == ["openai-luna", "openai-terra", "claude-sonnet"]
    prompt_tokens = max(1, len(ASK[0]["content"]) // 4)
    terra = REGISTRY["openai-terra"].actual_cost(prompt_tokens, 16000)          # billed, though it wrote nothing
    luna = REGISTRY["openai-luna"].actual_cost(prompt_tokens, max(1, len("five") // 4))
    sonnet = REGISTRY["claude-sonnet"].actual_cost(row.input_tokens, row.output_tokens)
    assert row.escalation_extra_cost == pytest.approx(sonnet + terra)
    assert row.cost == pytest.approx(luna + terra + sonnet)


@pytest.mark.parametrize("stream", [False, True])
def test_byok_requests_schedule_no_background_labels(stream):
    http, deps = app(sample_rate=1.0)
    chat(http, {"model": "auto", "messages": ASK, "stream": stream}, GROQ_ONLY | {"X-Triage-Quality": "0.1"})
    assert deps.log.labels() == []
    assert [c["kind"] for c in deps.fake_models.calls] == ["answer"]       # no strong answer, no judge call
    chat(http, {"model": "auto", "messages": ASK, "stream": stream}, {"X-Triage-Quality": "0.1"})
    assert len(deps.log.labels()) == 1                                     # the same request without keys is labeled


def test_background_eligibility_rejects_byok_rows():
    from app.api.background import eligible
    from app.api.request_log import RequestRow
    row = RequestRow(request_id="r", timestamp="t", key_id="k", requested_model="auto", stream=False, tier="low",
                     fell_back=False, cache_hit=False, mode="byok")
    assert eligible(row) is False
    row.mode = "api"
    assert eligible(row) is True


# ---------- a provider rejecting the user's key is a clear 401 ----------

REJECT = FakeBehavior(fail_before="auth", echo_key=True)
SLOW = FakeBehavior(fail_before="timeout")
ANTHROPIC_ONLY = {"x-anthropic-key": SECRET}


@pytest.mark.parametrize("stream", [False, True])
def test_named_model_with_a_rejected_key_is_a_401_naming_provider_and_header(stream):
    http, deps = app({"claude-haiku": REJECT})
    r = chat(http, {"model": "claude-haiku", "messages": ASK, "stream": stream}, ANTHROPIC_ONLY)
    assert r.status_code == 401
    err = r.json()["error"]
    assert (err["type"], err["code"]) == ("invalid_request_error", "invalid_provider_key")
    assert err["message"] == ("Anthropic rejected the API key you sent in X-Anthropic-Key. "
                              "Check the key and try again.")
    assert "tier" not in r.text and SECRET not in r.text
    (row,) = deps.log.rows()
    assert (row.status, row.http_status, row.error_kind) == ("error", 401, "invalid_provider_key")


@pytest.mark.parametrize("stream", [False, True])
def test_auto_with_every_key_rejected_is_a_401_naming_each_provider(stream):
    http, _ = app({"openai-sol": REJECT, "claude-opus": REJECT})
    r = chat(http, {"model": "auto", "messages": ASK, "stream": stream},
             {"x-openai-key": SECRET, "x-anthropic-key": "sk-ant-test-OTHER456", "X-Triage-Quality": "0.99"})
    assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_provider_key"
    assert r.json()["error"]["message"] == ("OpenAI and Anthropic rejected the API keys you sent in X-OpenAI-Key "
                                            "and X-Anthropic-Key. Check the keys and try again.")
    assert SECRET not in r.text and "sk-ant-test-OTHER456" not in r.text


@pytest.mark.parametrize("stream", [False, True])
def test_mixed_failures_keep_the_503(stream):
    http, _ = app({"openai-sol": REJECT, "claude-opus": SLOW})
    r = chat(http, {"model": "auto", "messages": ASK, "stream": stream},
             {"x-openai-key": SECRET, "x-anthropic-key": "sk-ant-test-OTHER456", "X-Triage-Quality": "0.99"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "all_models_failed"


@pytest.mark.parametrize("stream", [False, True])
def test_no_key_auth_failure_keeps_the_503(stream):
    http, _ = app({"groq-small": FakeBehavior(fail_before="auth")})
    r = chat(http, {"model": "groq-small", "messages": ASK, "stream": stream})
    assert r.status_code == 503 and r.json()["error"]["code"] == "all_models_failed"


# ---------- failed requests still log their attempts and primary model ----------

@pytest.mark.parametrize("stream", [False, True])
def test_a_failed_named_model_request_logs_its_attempts_and_primary(tmp_path, stream):
    db = tmp_path / "requests.sqlite3"
    http, deps = app({"claude-haiku": REJECT}, log=SQLiteRequestLog(db))
    chat(http, {"model": "claude-haiku", "messages": ASK, "stream": stream}, ANTHROPIC_ONLY)
    (row,) = SQLiteRequestLog(db).rows()
    assert (row.primary_model, row.tier, row.served_model) == ("claude-haiku", "low", None)
    (attempt,) = row.attempts
    assert (attempt["model"], attempt["outcome"], attempt["error_kind"]) == ("claude-haiku", "error", "auth")
    assert "Incorrect API key" in attempt["message"] and SECRET not in attempt["message"]
    deps.log.close()
    for f in tmp_path.iterdir():
        assert SECRET.encode() not in f.read_bytes(), f.name


@pytest.mark.parametrize("stream", [False, True])
def test_a_failed_auto_request_logs_every_attempt_tier_and_primary(stream):
    http, deps = app({"openai-sol": REJECT, "claude-opus": SLOW})
    r = chat(http, {"model": "auto", "messages": ASK, "stream": stream},
             {"x-openai-key": SECRET, "x-anthropic-key": "sk-ant-test-OTHER456", "X-Triage-Quality": "0.99"})
    assert r.status_code == 503
    (row,) = deps.log.rows()
    assert (row.tier, row.primary_model) == ("high", "openai-sol")
    assert [(a["model"], a["error_kind"]) for a in row.attempts] == [("openai-sol", "auth"),
                                                                     ("claude-opus", "timeout")]
    assert all(SECRET not in a.get("message", "") and "OTHER456" not in a.get("message", "") for a in row.attempts)


def test_a_failed_no_key_request_logs_its_attempts_too():
    http, deps = app({"groq-small": DOWN, "gemini-flash": DOWN})
    assert chat(http, headers={"X-Triage-Quality": "0.92"}).status_code == 503
    (row,) = deps.log.rows()
    assert (row.tier, row.primary_model) == ("medium", "groq-small")
    assert [a["model"] for a in row.attempts] == ["groq-small"] * 3 + ["gemini-flash"] * 3      # with retries


def test_successful_attempts_carry_no_message():
    http, deps = app()
    chat(http)
    (row,) = deps.log.rows()
    assert row.attempts == [{"model": "groq-small", "outcome": "success", "error_kind": None}]
