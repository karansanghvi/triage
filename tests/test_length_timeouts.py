"""Length-aware timeouts and the per-request time budget (routing.yaml timeout_per_1k_input_tokens,
timeout_cap_seconds, request_time_budget_seconds). Fake sends and a fake clock; Gemini's header with a mock
transport. No network."""
import copy
import json
import math

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.auth import Auth
from app.api.fake_models import FakeBehavior
from app.api.main import create_app, providers_if_every_failure_was, timed_out_providers
from app.providers import gemini_provider
from app.providers.common import ProviderError
from app.registry import ModelConfig, load_registry
from app.reliability import (AllModelsFailed, Attempt, BreakerSettings, ReliableClient, RetryPolicy, RoutingConfig,
                             estimate_input_tokens, load_routing, routing_from_dict)
from tests.api_helpers import API_KEY, FakePredictor, make_deps, routing_raw

REGISTRY = load_registry()
ROUTING = load_routing(REGISTRY)


def words(tokens: int) -> list:
    """A one-message conversation of about `tokens` input tokens (4 characters each)."""
    return [{"role": "user", "content": "x" * (tokens * 4)}]


# ---------- the formula and the caps (the real routing.yaml) ----------

@pytest.mark.parametrize("model, tokens, expected", [
    ("groq-large", 0, 10), ("groq-large", 2_000, 11), ("groq-large", 10_000, 15),
    ("groq-large", 20_000, 20), ("groq-large", 100_000, 20),                     # capped at 2x its base
    ("groq-small", 10_000, 13), ("groq-small", 50_000, 16),
    ("ollama-llama", 5_000, 52), ("ollama-llama", 20_000, 64),
    ("gemini-flash", 0, 60), ("gemini-flash", 10_000, 110), ("gemini-flash", 20_000, 120),
    ("claude-sonnet", 10_000, 110), ("openai-sol", 30_000, 120),
])
def test_per_call_timeout_grows_with_the_input_up_to_the_cap(model, tokens, expected):
    assert ROUTING.timeout_for(model, tokens) == pytest.approx(expected)


@pytest.mark.parametrize("model, tokens, expected", [
    ("groq-large", 10_000, 15), ("gemini-flash", 10_000, 110), ("gemini-flash", 40_000, 120),
])
def test_first_token_timeout_uses_the_same_formula(model, tokens, expected):
    assert ROUTING.first_token_timeout_for(model, tokens) == pytest.approx(expected)


def test_every_chain_model_has_a_rate_and_a_cap_and_the_budget_is_150():
    names = {n for chain in ROUTING.chains.values() for n in chain}
    assert names <= set(ROUTING.model_timeout_per_1k) and names <= set(ROUTING.model_timeout_caps)
    assert ROUTING.request_time_budget_seconds == 150


def test_a_model_without_an_entry_keeps_its_fixed_timeout():
    raw = routing_raw()
    raw["timeout_per_1k_input_tokens"] = {}
    raw["timeout_cap_seconds"] = {}
    routing = routing_from_dict(raw, REGISTRY)
    assert routing.timeout_for("groq-large", 50_000) == 10
    assert routing.first_token_timeout_for("gemini-flash", 50_000) == 60


@pytest.mark.parametrize("change, message", [
    ({"timeout_cap_seconds": {"groq-large": 9}}, "below its timeout"),
    ({"timeout_per_1k_input_tokens": {"nope": 1}}, "unknown model 'nope'"),
    ({"timeout_per_1k_input_tokens": {"groq-large": -1}}, "above 0"),
    ({"timeout_cap_seconds": "120"}, "must be a mapping"),
    ({"request_time_budget_seconds": 0}, "above 0"),
])
def test_bad_settings_are_refused(change, message):
    raw = routing_raw()
    for key, value in change.items():
        raw[key] = {**raw.get(key, {}), **value} if isinstance(value, dict) else value
    with pytest.raises(ValueError, match=message):
        routing_from_dict(raw, REGISTRY)


def test_input_tokens_are_estimated_at_4_characters_each():
    messages = [{"role": "system", "content": "a" * 400}, {"role": "user", "content": "b" * 4000}]
    assert estimate_input_tokens(messages) == 1100


# ---------- through ReliableClient ----------

def model(name, provider="alpha", tier="low"):
    return ModelConfig(name=name, provider=provider, model_id=name, tier=tier, input_price=0, output_price=0)


TOY = {n: model(n, p) for n, p in [("a1", "alpha"), ("b1", "beta"), ("c1", "gamma"), ("d1", "delta")]}


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class SlowSend:
    """Each call takes `seconds` of (fake) time, then fails with `kind`; records the timeout it was given."""

    def __init__(self, clock, seconds, kind="timeout", ok=()):
        self.clock, self.seconds, self.kind, self.ok = clock, seconds, kind, set(ok)
        self.calls = []

    def __call__(self, messages, m, stream=False, **settings):
        self.calls.append((m.name, settings["timeout"]))
        self.clock.now += min(self.seconds, settings["timeout"])      # gives up at its timeout
        if m.name in self.ok:
            return f"answer from {m.name}"
        raise ProviderError(m.provider, self.kind, "fake")


def toy_client(send, clock, *, budget=150.0, timeout=60.0, per_1k=None, caps=None, max_attempts=3, sleep=None):
    routing = RoutingConfig(chains={"low": ["a1", "b1", "c1", "d1"]},
                            retry=RetryPolicy(max_attempts=max_attempts, base_delay_seconds=10, max_delay_seconds=40),
                            breaker=BreakerSettings(failure_threshold=99), timeout_seconds=timeout,
                            model_timeout_per_1k=per_1k or {}, model_timeout_caps=caps or {},
                            request_time_budget_seconds=budget)
    return ReliableClient(TOY, routing, send_fn=send, clock=clock, sleep=sleep or (lambda s: None))


def test_the_budget_cuts_each_timeout_to_the_time_left_and_stops_the_chain():
    clock = Clock()
    send = SlowSend(clock, seconds=60)
    with pytest.raises(AllModelsFailed) as info:
        toy_client(send, clock).send([{"role": "user", "content": "hi"}], "low")
    # 60 s, 60 s, then only 30 s were left; d1 is never called
    assert [(n, round(t, 3)) for n, t in send.calls] == [("a1", 60), ("b1", 60), ("c1", 30)]
    assert [(a.model_name, a.outcome, a.error_kind) for a in info.value.attempts] == [
        ("a1", "error", "timeout"), ("b1", "error", "timeout"), ("c1", "error", "timeout"),
        ("d1", "skipped", "budget_exhausted")]
    assert clock.now - 1000 == pytest.approx(150)
    # the failures so far decide the error: all timeouts
    assert timed_out_providers(info.value.attempts) == ["alpha", "beta", "gamma"]


def test_a_model_that_answers_within_the_budget_is_served():
    clock = Clock()
    send = SlowSend(clock, seconds=60, ok={"c1"})
    result = toy_client(send, clock).send([{"role": "user", "content": "hi"}], "low")
    assert result.model.name == "c1" and send.calls[-1] == ("c1", 30)


def test_the_budget_stops_retries_and_backoff_too():
    clock = Clock()
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock.now += seconds

    send = SlowSend(clock, seconds=50, kind="server")                  # retryable, so a1 is retried
    with pytest.raises(AllModelsFailed) as info:
        toy_client(send, clock, budget=120, sleep=sleep).send([{"role": "user", "content": "hi"}], "low")
    assert clock.now - 1000 <= 120 + 1e-6                              # calls and waits stayed inside the budget
    assert len(send.calls) >= 2                                        # a1 was retried while there was time
    assert info.value.attempts[-1].error_kind == "budget_exhausted"


def test_no_budget_means_the_whole_chain_is_tried_as_before():
    clock = Clock()
    send = SlowSend(clock, seconds=60)
    with pytest.raises(AllModelsFailed):
        toy_client(send, clock, budget=None).send([{"role": "user", "content": "hi"}], "low")
    assert [n for n, _ in send.calls] == ["a1", "b1", "c1", "d1"] and all(t == 60 for _, t in send.calls)


def test_a_long_input_gets_the_longer_timeout_within_the_budget():
    clock = Clock()
    send = SlowSend(clock, seconds=1, ok={"a1"})
    client = toy_client(send, clock, timeout=10, per_1k={"a1": 2}, caps={"a1": 30})
    client.send(words(5_000), "low")
    assert send.calls == [("a1", 20)]                                  # 10 + 2 x 5


def test_the_first_token_limit_is_length_aware_and_cut_by_the_budget(monkeypatch):
    from app import reliability
    limits = []
    real = reliability._first_piece

    def spy(iterator, limit, m):
        limits.append(limit)
        return real(iterator, None, m)

    monkeypatch.setattr(reliability, "_first_piece", spy)

    class Stream:
        def __iter__(self):
            yield "Hi"

    routing = RoutingConfig(chains={"low": ["a1"]}, retry=RetryPolicy(), breaker=BreakerSettings(),
                            timeout_seconds=10, first_token_timeout_seconds=8, model_timeout_per_1k={"a1": 2},
                            model_timeout_caps={"a1": 30}, request_time_budget_seconds=150)
    client = ReliableClient(TOY, routing, send_fn=lambda *a, **k: Stream(), sleep=lambda s: None)
    client.send(words(5_000), "low", stream=True)
    assert limits[0] == pytest.approx(18)                              # 8 + 2 x 5
    routing = RoutingConfig(**{**routing.__dict__, "request_time_budget_seconds": 5})
    ReliableClient(TOY, routing, send_fn=lambda *a, **k: Stream(), sleep=lambda s: None).send(
        words(5_000), "low", stream=True)
    assert limits[1] <= 5                                              # cut to the time left


def test_skipped_models_dont_change_which_error_the_failures_give():
    attempts = [Attempt("groq-small", "groq", "error", "timeout"),
                Attempt("gemini-flash", "gemini", "error", "overloaded"),
                Attempt("gemini-flash-lite", "gemini", "skipped", "budget_exhausted")]
    assert providers_if_every_failure_was(attempts, "overloaded", "timeout") == ["groq", "gemini"]
    assert timed_out_providers(attempts) == []                         # mixed: provider_unavailable, not timeout


# ---------- Gemini's X-Server-Timeout follows ----------

GEMINI_REPLY = {"candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2}}


@pytest.mark.parametrize("tokens", [0, 4_000, 10_000, 30_000])
def test_geminis_server_timeout_header_follows_the_length_aware_timeout(monkeypatch, tokens):
    monkeypatch.setenv("GEMINI_API_KEY", "server-env-key")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, content=json.dumps(GEMINI_REPLY), headers={"content-type": "application/json"})

    monkeypatch.setattr(gemini_provider, "HTTP_CLIENT", httpx.Client(transport=httpx.MockTransport(handler)))
    messages = words(tokens) if tokens else [{"role": "user", "content": ""}]
    ReliableClient(REGISTRY, ROUTING, sleep=lambda s: None).send_model(messages, "gemini-flash")
    (request,) = requests
    expected = ROUTING.timeout_for("gemini-flash", estimate_input_tokens(messages))
    assert request.extensions["timeout"]["read"] == pytest.approx(expected, abs=0.01)
    assert request.headers["x-server-timeout"] == str(math.ceil(expected))
    assert int(request.headers["x-server-timeout"]) <= 120


# ---------- the API: the usual error for the failures so far ----------

def test_the_api_cuts_later_models_to_the_time_left_and_gives_the_error_for_the_failures():
    """A real-time budget of 0.5 s: groq-small takes 0.45 s and times out, so the Gemini models get only the
    ~0.05 s left (less than they need) and time out too: every failure a timeout, so provider_timeout."""
    raw = copy.deepcopy(routing_raw())
    raw["request_time_budget_seconds"] = 0.5
    slow = FakeBehavior(first_token_delay=0.2)                         # would answer, given 0.2 s
    deps = make_deps({"groq-small": FakeBehavior(first_token_delay=0.45, fail_before="timeout"),
                      "gemini-flash": slow, "gemini-flash-lite": slow},
                     predictor=FakePredictor(0.4, 0.95), raw=raw)
    http = TestClient(create_app(deps=deps, auth=Auth([API_KEY])))
    r = http.post("/v1/chat/completions", json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
                  headers={"Authorization": f"Bearer {API_KEY}"})
    assert r.status_code == 504 and r.json()["error"]["code"] == "provider_timeout"
    assert r.json()["error"]["message"].startswith("Groq and Gemini took too long to answer.")
    answers = [(c["model"], c["timeout"]) for c in deps.fake_models.calls if c["kind"] == "answer"]
    assert [m for m, _ in answers] == ["groq-small", "gemini-flash", "gemini-flash-lite"]
    assert answers[0][1] == pytest.approx(0.5, abs=0.05)                # its 8 s cut to the budget
    assert all(t < 0.2 for _, t in answers[1:])                         # only the time left


def test_without_the_budget_the_same_request_is_answered():
    raw = copy.deepcopy(routing_raw())
    raw["request_time_budget_seconds"] = None
    deps = make_deps({"groq-small": FakeBehavior(first_token_delay=0.45, fail_before="timeout"),
                      "gemini-flash": FakeBehavior(first_token_delay=0.2)},
                     predictor=FakePredictor(0.4, 0.95), raw=raw)
    http = TestClient(create_app(deps=deps, auth=Auth([API_KEY])))
    r = http.post("/v1/chat/completions", json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
                  headers={"Authorization": f"Bearer {API_KEY}"})
    assert r.status_code == 200 and r.headers["x-triage-model"] == "gemini-flash"
