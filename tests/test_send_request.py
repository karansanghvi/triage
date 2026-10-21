import pytest

from app.providers import base
from app.providers.common import ProviderError, RawResult, Usage
from app.registry import ModelConfig, PriceAs

MODEL = ModelConfig(
    name="fake-model",
    provider="groq",
    model_id="fake-id",
    tier="low",
    input_price=0,
    output_price=0,
    price_as=PriceAs("Paid model", 1.0, 2.0),
)
MESSAGES = [{"role": "user", "content": "Hi"}]


def fake_adapter(messages, model_id, *, stream, temperature, max_tokens, timeout):
    if stream:
        def events():
            yield "Hel"
            yield "lo"
            yield Usage(10, 2)
        return events()
    return RawResult(text="Hello", usage=Usage(10, 2), finish_reason="stop")


@pytest.fixture(autouse=True)
def use_fake_adapter(monkeypatch):
    # Every test uses the fake adapter unless it swaps in something else
    monkeypatch.setitem(base.ADAPTERS, "groq", fake_adapter)


def test_normal_request_builds_response():
    response = base.send_request(MESSAGES, MODEL)
    assert response.text == "Hello"
    assert (response.input_tokens, response.output_tokens) == (10, 2)
    assert response.cost == 0
    assert response.reported_cost == pytest.approx((10 * 1.0 + 2 * 2.0) / 1_000_000)
    assert response.model_name == "fake-model"
    assert not response.tokens_estimated


def test_streaming_yields_pieces_then_response():
    stream = base.send_request(MESSAGES, MODEL, stream=True)
    assert list(stream) == ["Hel", "lo"]
    assert stream.response.text == "Hello"
    assert stream.response.output_tokens == 2
    assert stream.response.time_to_first_token_ms is not None


def test_missing_usage_is_estimated(monkeypatch):
    monkeypatch.setitem(
        base.ADAPTERS, "groq", lambda *args, **kwargs: RawResult("Hello there", None)
    )
    response = base.send_request(MESSAGES, MODEL)
    assert response.tokens_estimated
    assert response.output_tokens > 0


@pytest.mark.parametrize(
    "bad_messages",
    [
        [],
        [{"role": "robot", "content": "hi"}],
        [{"role": "user"}],
        [{"role": "system", "content": "only a system message"}],
    ],
)
def test_rejects_bad_messages(bad_messages):
    with pytest.raises(ValueError):
        base.send_request(bad_messages, MODEL)


def test_missing_key_raises_config_error(monkeypatch):
    from app.providers import groq_provider

    monkeypatch.setitem(base.ADAPTERS, "groq", groq_provider.send)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ProviderError) as info:
        base.send_request(MESSAGES, MODEL)
    assert info.value.kind == "config"
    assert not info.value.retryable


def test_only_rate_limits_and_server_errors_are_retryable():
    assert ProviderError("groq", "rate_limit", "slow down").retryable
    assert ProviderError("groq", "server", "oops").retryable
    assert not ProviderError("groq", "client", "bad key").retryable
    assert not ProviderError("groq", "timeout", "too slow").retryable

def test_reasoning_effort_is_passed_only_when_set(monkeypatch):
    seen = []

    def recording_adapter(messages, model_id, **settings):
        seen.append(settings)
        return RawResult(text="ok", usage=Usage(5, 1), finish_reason="stop")

    monkeypatch.setitem(base.ADAPTERS, "groq", recording_adapter)
    base.send_request(MESSAGES, MODEL)
    base.send_request(MESSAGES, MODEL, reasoning_effort="low")
    assert "reasoning_effort" not in seen[0]
    assert seen[1]["reasoning_effort"] == "low"


def test_reasoning_effort_rejected_for_non_openai_format_providers():
    gemini = ModelConfig(name="g", provider="gemini", model_id="m", tier="high", input_price=0, output_price=0)
    with pytest.raises(ProviderError) as error:
        base.send_request(MESSAGES, gemini, reasoning_effort="low")
    assert error.value.kind == "config"


def test_reasoning_and_reasoning_tokens_reach_the_response(monkeypatch):
    monkeypatch.setitem(
        base.ADAPTERS, "groq",
        lambda *args, **kwargs: RawResult("42", Usage(5, 30, reasoning_tokens=28), "stop", reasoning="thinking..."),
    )
    response = base.send_request(MESSAGES, MODEL)
    assert response.text == "42"
    assert response.reasoning == "thinking..."
    assert (response.output_tokens, response.reasoning_tokens) == (30, 28)


def test_response_format_is_passed_only_when_set(monkeypatch):
    seen = []

    def recording_adapter(messages, model_id, **settings):
        seen.append(settings)
        return RawResult(text="{}", usage=Usage(5, 1), finish_reason="stop")

    monkeypatch.setitem(base.ADAPTERS, "groq", recording_adapter)
    base.send_request(MESSAGES, MODEL)
    base.send_request(MESSAGES, MODEL, response_format={"type": "json_object"})
    assert "response_format" not in seen[0]
    assert seen[1]["response_format"] == {"type": "json_object"}


def test_response_format_rejected_for_non_openai_format_providers():
    gemini = ModelConfig(name="g", provider="gemini", model_id="m", tier="high", input_price=0, output_price=0)
    with pytest.raises(ProviderError) as error:
        base.send_request(MESSAGES, gemini, response_format={"type": "json_object"})
    assert error.value.kind == "config"
