import random

import pytest

from app.providers.common import ProviderError
from app.registry import ModelConfig, load_registry
from app.reliability import (
    AllModelsFailed, BreakerSettings, ReliableClient, RetryPolicy, RoutingConfig,
    backoff_delay, load_routing,
)


def make_model(name, provider):
    return ModelConfig(name=name, provider=provider, model_id=name, tier="low",
                       input_price=0, output_price=0)


# Two models on provider "alpha", one on provider "beta"
REGISTRY = {
    "a1": make_model("a1", "alpha"),
    "a2": make_model("a2", "alpha"),
    "b1": make_model("b1", "beta"),
}
MESSAGES = [{"role": "user", "content": "hi"}]


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class FakeStream:
    """Behaves like StreamingResponse: errors appear when you start reading."""

    def __init__(self, pieces, error=None):
        self.pieces = pieces
        self.error = error
        self.response = None

    def __iter__(self):
        if self.error:
            raise self.error
        yield from self.pieces
        self.response = "".join(self.pieces)


class FakeSend:
    """Plays a script of outcomes per model. The last outcome repeats forever."""

    def __init__(self, script):
        self.script = {name: list(outcomes) for name, outcomes in script.items()}
        self.calls = []

    def __call__(self, messages, model, stream=False, **settings):
        self.calls.append(model.name)
        outcomes = self.script.get(model.name, ["ok"])
        outcome = outcomes.pop(0) if len(outcomes) > 1 else outcomes[0]
        error = None if outcome == "ok" else ProviderError(model.provider, outcome, "fake failure")
        if stream:
            return FakeStream(["Hi", " there"], error)
        if error:
            raise error
        return f"answer from {model.name}"


def make_client(script, chains=None, clock=None):
    routing = RoutingConfig(
        chains=chains or {"low": ["a1", "b1"]},
        retry=RetryPolicy(max_attempts=3, base_delay_seconds=0.5, max_delay_seconds=8),
        breaker=BreakerSettings(failure_threshold=3, cooldown_seconds=60),
        timeout_seconds=30,
    )
    fake = FakeSend(script)
    sleeps = []
    client = ReliableClient(
        REGISTRY, routing, send_fn=fake, sleep=sleeps.append,
        clock=clock or FakeClock(), rng=random.Random(42),
    )
    return client, fake, sleeps


def test_first_model_answers_without_fallback():
    client, fake, sleeps = make_client({})
    result = client.send(MESSAGES, "low")
    assert result.response == "answer from a1"
    assert not result.fell_back
    assert fake.calls == ["a1"]
    assert sleeps == []


def test_rate_limit_is_retried_then_succeeds():
    client, fake, sleeps = make_client({"a1": ["rate_limit", "ok"]})
    result = client.send(MESSAGES, "low")
    assert result.model.name == "a1"
    assert fake.calls == ["a1", "a1"]
    assert len(sleeps) == 1
    assert not result.fell_back


def test_client_error_is_not_retried_and_falls_back():
    client, fake, sleeps = make_client({"a1": ["client"]})
    result = client.send(MESSAGES, "low")
    assert result.model.name == "b1"
    assert fake.calls == ["a1", "b1"]
    assert sleeps == []
    assert client.fallback_events[0].reason == "client"


def test_falls_back_after_retries_run_out():
    client, fake, sleeps = make_client({"a1": ["server"]})
    result = client.send(MESSAGES, "low")
    assert fake.calls == ["a1", "a1", "a1", "b1"]
    assert len(sleeps) == 2
    assert result.fell_back
    event = client.fallback_events[0]
    assert (event.from_model, event.to_model, event.reason) == ("a1", "b1", "server")


def test_circuit_breaker_skips_provider_then_recovers():
    clock = FakeClock()
    client, fake, _ = make_client({"a1": ["server"]}, clock=clock)

    client.send(MESSAGES, "low")              # a1 fails 3 times in a row: breaker trips
    fake.calls.clear()

    result = client.send(MESSAGES, "low")     # alpha is skipped entirely
    assert fake.calls == ["b1"]
    assert result.attempts[0].outcome == "skipped"

    clock.now += 61                           # cooldown is over
    fake.script["a1"] = ["ok"]
    result = client.send(MESSAGES, "low")
    assert result.model.name == "a1"


def test_breaker_skips_every_model_on_that_provider():
    client, fake, _ = make_client({"a1": ["server"]}, chains={"low": ["a1", "a2", "b1"]})
    result = client.send(MESSAGES, "low")
    assert fake.calls == ["a1", "a1", "a1", "b1"]   # a2 shares alpha, so it's skipped too
    assert result.attempts[3].model_name == "a2"
    assert result.attempts[3].outcome == "skipped"


def test_all_models_failing_raises():
    client, _, _ = make_client({"a1": ["server"], "b1": ["timeout"]})
    with pytest.raises(AllModelsFailed) as info:
        client.send(MESSAGES, "low")
    kinds = {attempt.error_kind for attempt in info.value.attempts}
    assert kinds == {"server", "timeout"}
    assert client.fallback_events[-1].to_model is None


def test_backoff_grows_but_is_capped_and_random():
    policy = RetryPolicy(max_attempts=10, base_delay_seconds=1, max_delay_seconds=4)
    rng = random.Random(0)
    for attempt in range(1, 8):
        limit = min(4, 2 ** (attempt - 1))
        delay = backoff_delay(attempt, policy, rng)
        assert 0 <= delay <= limit


def test_stream_falls_back_if_it_fails_before_the_first_piece():
    client, fake, _ = make_client({"a1": ["client"]})
    result = client.send(MESSAGES, "low", stream=True)
    assert result.model.name == "b1"
    assert list(result.response) == ["Hi", " there"]
    assert result.response.response == "Hi there"


def test_routing_rejects_unknown_model(tmp_path):
    path = tmp_path / "routing.yaml"
    path.write_text("fallback_chains:\n  low: [nope]\n  medium: [a1]\n  high: [b1]\n")
    with pytest.raises(ValueError, match="unknown model 'nope'"):
        load_routing(REGISTRY, path)


def test_real_routing_config_loads():
    """Your actual routing.yaml should match your actual models.yaml."""
    registry = load_registry()
    routing = load_routing(registry)
    assert set(routing.chains) == {"low", "medium", "high"}


def test_rate_limit_only_skips_that_model_not_the_provider():
    # a1 keeps hitting its rate limit; a2 is on the same provider but is fine
    client, fake, _ = make_client({"a1": ["rate_limit"]}, chains={"low": ["a1", "a2", "b1"]})
    result = client.send(MESSAGES, "low")
    assert result.model.name == "a2"              # same provider, still used
    assert fake.calls == ["a1", "a1", "a1", "a2"]

    fake.calls.clear()
    result = client.send(MESSAGES, "low")         # now only a1 is skipped
    assert fake.calls == ["a2"]
    assert result.attempts[0].outcome == "skipped"