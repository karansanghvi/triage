import random

import pytest

from app.cache import CachedSender, ResponseCache, is_cacheable, make_cache_key
from app.providers.base import Response
from app.registry import ModelConfig
from app.reliability import BreakerSettings, ReliableClient, RetryPolicy, RoutingConfig

MODEL = ModelConfig(name="m1", provider="alpha", model_id="model-one", tier="low",
                    input_price=0, output_price=0)
OTHER_MODEL = ModelConfig(name="m2", provider="alpha", model_id="model-two", tier="low",
                          input_price=0, output_price=0)
MESSAGES = [{"role": "user", "content": "hi"}]


def make_response(model, text="hello", finish_reason="stop", reported_cost=0.002):
    return Response(
        text=text, model_name=model.name, model_id=model.model_id, provider=model.provider,
        input_tokens=5, output_tokens=3, latency_ms=400.0, cost=0.0,
        reported_cost=reported_cost, finish_reason=finish_reason,
    )


class FakeClock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


class FakeSend:
    def __init__(self, **response_kwargs):
        self.calls = 0
        self.response_kwargs = response_kwargs

    def __call__(self, messages, model, stream=False, **settings):
        self.calls += 1
        return make_response(model, **self.response_kwargs)


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def cache(tmp_path, clock):
    cache = ResponseCache(path=tmp_path / "cache.sqlite3", ttl_seconds=60, clock=clock)
    yield cache
    cache.close()


def test_second_identical_request_is_a_free_hit(cache):
    fake = FakeSend()
    sender = CachedSender(cache, send_fn=fake)

    first = sender(MESSAGES, MODEL)
    second = sender(MESSAGES, MODEL)

    assert fake.calls == 1
    assert not first.cached
    assert second.cached
    assert second.text == "hello"
    assert second.cost == 0 and second.reported_cost == 0
    assert second.cache_saved_cost == pytest.approx(0.002)
    assert (sender.hits, sender.misses) == (1, 1)


def test_key_changes_with_anything_that_affects_the_answer():
    base = make_cache_key(MESSAGES, MODEL, 0, 1024)
    assert make_cache_key([{"role": "user", "content": "hello"}], MODEL, 0, 1024) != base
    assert make_cache_key(MESSAGES, OTHER_MODEL, 0, 1024) != base
    assert make_cache_key(MESSAGES, MODEL, 0, 500) != base
    assert make_cache_key(MESSAGES, MODEL, 0, 1024, reasoning_effort="low") != base


def test_unset_reasoning_effort_keeps_the_old_key():
    assert make_cache_key(MESSAGES, MODEL, 0, 1024, reasoning_effort=None) == make_cache_key(MESSAGES, MODEL, 0, 1024)


def test_key_ignores_dict_order_and_int_vs_float():
    reordered = [{"content": "hi", "role": "user"}]
    assert make_cache_key(reordered, MODEL, 0, 1024) == make_cache_key(MESSAGES, MODEL, 0.0, 1024)


def test_only_temperature_zero_non_streaming_is_cacheable():
    assert is_cacheable(0, stream=False)
    assert not is_cacheable(0.7, stream=False)
    assert not is_cacheable(0, stream=True)


def test_higher_temperature_always_calls_the_model(cache):
    fake = FakeSend()
    sender = CachedSender(cache, send_fn=fake)
    sender(MESSAGES, MODEL, temperature=0.7)
    sender(MESSAGES, MODEL, temperature=0.7)
    assert fake.calls == 2
    assert cache.count() == 0


def test_entries_expire(cache, clock):
    fake = FakeSend()
    sender = CachedSender(cache, send_fn=fake)
    sender(MESSAGES, MODEL)
    clock.now += 61                       # past the 60-second TTL
    result = sender(MESSAGES, MODEL)
    assert fake.calls == 2
    assert not result.cached


@pytest.mark.parametrize("kwargs", [{"text": "   "}, {"finish_reason": "length"}, {"finish_reason": "MAX_TOKENS"}])
def test_empty_or_cut_off_answers_are_not_saved(cache, kwargs):
    sender = CachedSender(cache, send_fn=FakeSend(**kwargs))
    sender(MESSAGES, MODEL)
    assert cache.count() == 0


def test_cache_survives_a_restart(tmp_path, clock):
    path = tmp_path / "cache.sqlite3"
    first_cache = ResponseCache(path=path, clock=clock)
    CachedSender(first_cache, send_fn=FakeSend())(MESSAGES, MODEL)
    first_cache.close()

    reopened = ResponseCache(path=path, clock=clock)   # like restarting the program
    fake = FakeSend()
    result = CachedSender(reopened, send_fn=fake)(MESSAGES, MODEL)
    assert result.cached
    assert fake.calls == 0
    reopened.close()


def test_purge_removes_only_expired_entries(cache, clock):
    sender = CachedSender(cache, send_fn=FakeSend())
    sender(MESSAGES, MODEL)
    clock.now += 61
    sender([{"role": "user", "content": "new question"}], MODEL)
    assert cache.purge_expired() == 1
    assert cache.count() == 1


def test_works_inside_the_reliability_layer(cache):
    fake = FakeSend()
    routing = RoutingConfig(
        chains={"low": ["m1"]}, retry=RetryPolicy(), breaker=BreakerSettings(), timeout_seconds=30,
    )
    client = ReliableClient(
        {"m1": MODEL}, routing, send_fn=CachedSender(cache, send_fn=fake),
        sleep=lambda seconds: None, rng=random.Random(0),
    )
    client.send(MESSAGES, "low")
    result = client.send(MESSAGES, "low")
    assert fake.calls == 1
    assert result.response.cached