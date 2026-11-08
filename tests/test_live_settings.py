"""Live calls must use the same settings as the Phase 2 collection, applied per model (Phase 4 Step 0, Part D)."""
import random

import pytest
import yaml

from app.providers.base import Response
from app.providers.common import ProviderError
from app.registry import ModelConfig, load_registry
from app.reliability import (BreakerSettings, ReliableClient, RetryPolicy, RoutingConfig, DEFAULT_ROUTING_PATH,
                             load_routing)
from eval.build_dataset import COLLECTION_CONFIG_PATH, load_collection_config

MESSAGES = [{"role": "user", "content": "hi"}]


# ---------- routing.yaml vs collection.yaml (the real files) ----------

@pytest.fixture(scope="module")
def configs():
    registry = load_registry()
    return registry, load_routing(registry), load_collection_config(registry)


def test_generation_settings_match_phase2(configs):
    _, routing, collection = configs
    assert routing.generation == {"temperature": collection.temperature, "max_tokens": collection.max_tokens}


def test_model_params_match_phase2_for_every_label_tier_model(configs):
    _, routing, collection = configs
    for tier in collection.label_tiers:
        name = collection.tiers[tier]
        assert routing.model_params.get(name, {}) == collection.model_params.get(name, {}), name


def test_raw_yaml_sections_match():
    """Same check on the raw files, so a loader default can't hide a missing section."""
    live = yaml.safe_load(DEFAULT_ROUTING_PATH.read_text())
    phase2 = yaml.safe_load(COLLECTION_CONFIG_PATH.read_text())
    assert live["generation"] == phase2["generation"]
    for name in (phase2["tiers"][t] for t in phase2["label_tiers"]):
        assert live.get("model_params", {}).get(name) == phase2.get("model_params", {}).get(name), name


def test_each_tier_calls_its_phase2_model_first(configs):
    _, routing, collection = configs
    for tier in collection.label_tiers:
        assert routing.chains[tier][0] == collection.tiers[tier]


def test_no_fallback_model_has_reasoning_effort_unless_openai_format(configs):
    registry, routing, _ = configs
    for name, params in routing.model_params.items():
        if "reasoning_effort" in params:
            assert registry[name].provider in {"openai", "groq", "ollama", "openrouter"}


# ---------- ReliableClient applies params per MODEL ----------

def model(name, provider):
    return ModelConfig(name=name, provider=provider, model_id=name, tier="medium", input_price=0, output_price=0)


REGISTRY = {"groq-small": model("groq-small", "groq"), "gemini-flash": model("gemini-flash", "gemini")}


class RecordingSend:
    def __init__(self, fail=()):
        self.fail, self.calls = set(fail), []

    def __call__(self, messages, model, stream=False, **settings):
        self.calls.append((model.name, settings))
        if model.name in self.fail:
            raise ProviderError(model.provider, "server", "down")
        return f"answer from {model.name}"


def client(send, **routing_kwargs):
    routing = RoutingConfig(
        chains={"medium": ["groq-small", "gemini-flash"]},
        retry=RetryPolicy(max_attempts=1), breaker=BreakerSettings(), timeout_seconds=30,
        **routing_kwargs)
    return ReliableClient(REGISTRY, routing, send_fn=send, sleep=lambda s: None, rng=random.Random(0))


def test_fallback_never_receives_reasoning_effort():
    send = RecordingSend(fail={"groq-small"})
    result = client(send, model_params={"groq-small": {"reasoning_effort": "low"}}).send(MESSAGES, "medium")
    assert result.model.name == "gemini-flash"
    (first, s1), (second, s2) = send.calls
    assert (first, s1["reasoning_effort"]) == ("groq-small", "low")
    assert second == "gemini-flash" and "reasoning_effort" not in s2


def test_generation_defaults_apply_to_every_model_and_caller_can_override():
    send = RecordingSend(fail={"groq-small"})
    c = client(send, generation={"temperature": 0.0, "max_tokens": 4000})
    c.send(MESSAGES, "medium")
    assert all(s["max_tokens"] == 4000 and s["temperature"] == 0.0 and s["timeout"] == 30 for _, s in send.calls)
    send.calls.clear()
    c.send(MESSAGES, "medium", temperature=0.7)
    assert all(s["temperature"] == 0.7 and s["max_tokens"] == 4000 for _, s in send.calls)


def test_reasoning_effort_per_call_is_refused():
    with pytest.raises(ValueError, match="model_params"):
        client(RecordingSend()).send(MESSAGES, "medium", reasoning_effort="low")


# ---------- load_routing validation ----------

def write(tmp_path, extra: str):
    path = tmp_path / "routing.yaml"
    path.write_text("fallback_chains:\n  low: [groq-small]\n  medium: [groq-small]\n  high: [gemini-flash]\n" + extra)
    return path


def test_load_routing_reads_generation_and_model_params(tmp_path):
    path = write(tmp_path, "generation: {temperature: 0, max_tokens: 4000}\n"
                           "model_params:\n  groq-small: {reasoning_effort: low}\n")
    routing = load_routing(REGISTRY, path)
    assert routing.generation == {"temperature": 0.0, "max_tokens": 4000}
    assert routing.model_params == {"groq-small": {"reasoning_effort": "low"}}


@pytest.mark.parametrize("extra, message", [
    ("model_params:\n  gemini-flash: {reasoning_effort: low}\n", "does not accept reasoning_effort"),
    ("model_params:\n  nope: {reasoning_effort: low}\n", "unknown model"),
    ("model_params:\n  groq-small: {top_p: 1}\n", "unsupported model_params"),
    ("generation: {top_k: 5}\n", "unsupported generation"),
    ("generation: {max_tokens: 0}\n", "max_tokens"),
    ("generation: {temperature: -1}\n", "temperature"),
])
def test_load_routing_rejects_bad_settings(tmp_path, extra, message):
    with pytest.raises(ValueError, match=message):
        load_routing(REGISTRY, write(tmp_path, extra))


# ---------- the Phase 2 collector still sends reasoning_effort, now through model_params ----------

def test_collector_sends_reasoning_effort_via_model_params(tmp_path, monkeypatch):
    import eval.build_dataset as bd
    seen = []

    def fake_send(messages, model, stream=False, **settings):
        seen.append((model.name, settings))
        return Response(text="ok", model_name=model.name, model_id=model.model_id, provider=model.provider,
                        input_tokens=1, output_tokens=1, latency_ms=1.0, cost=0.0, reported_cost=0.0,
                        finish_reason="stop")

    monkeypatch.setattr(bd, "ResponseCache", lambda: None)            # never open the real cache
    monkeypatch.setattr(bd, "CachedSender", lambda cache: fake_send)
    registry = load_registry()
    config = load_collection_config(registry)
    collector = bd.Collector(config, registry, bd.AnswerLog(tmp_path / "answers.jsonl"), run="test")
    prompt = {"id": "p1", "messages": MESSAGES}
    for tier in ("low", "medium", "high"):
        record, _ = collector.call(tier, registry[config.tiers[tier]], prompt)
        assert record["error"] is None
    sent = dict(seen)
    assert "reasoning_effort" not in sent["groq-allam"]
    assert sent["groq-small"]["reasoning_effort"] == sent["groq-large"]["reasoning_effort"] == "low"
    assert all(s["max_tokens"] == 4000 and s["temperature"] == 0 and s["timeout"] == 180 for s in sent.values())
