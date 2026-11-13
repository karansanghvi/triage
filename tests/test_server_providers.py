"""Server providers vs BYOK models (Phase 5, Step 2). No network, no keys.

routing.yaml lists openai-* and claude-* models at the END of each fallback chain, and `server_providers` drops them
for requests without their own keys. These tests pin the guarantee: a request without keys routes exactly as it did
before those models were added (same chains, same attempts, same baseline), and never tries one of them.
"""
import random

import anyio
import pytest
import yaml
from fastapi.testclient import TestClient

from app.api.auth import Auth
from app.api.config_state import ActiveConfig
from app.api.core import GenerationSettings, route_and_stream
from app.api.fake_models import FakeBehavior, fake_text
from app.api.main import create_app
from app.providers.common import PROVIDER_KEY_ENV, ProviderError, missing_key_env
from app.registry import VALID_PROVIDERS, ModelConfig, load_registry
from app.reliability import (DEFAULT_ROUTING_PATH, AllModelsFailed, BreakerSettings, ModelNotAllowed,
                             ReliableClient, RetryPolicy, RoutingConfig, filter_chain, load_routing)
from tests.api_helpers import API_KEY, ASK, FakePredictor, make_deps

BYOK_PROVIDERS = {"openai", "anthropic"}
# The chains before any BYOK model existed (Phase 4). Requests without keys must still get exactly these.
PHASE4_CHAINS = {"low": ["groq-allam", "ollama-llama"],
                 "medium": ["groq-small", "gemini-flash"],
                 "high": ["groq-large", "gemini-flash"]}
DOWN = FakeBehavior(fail_before="server")


@pytest.fixture(scope="module")
def registry():
    return load_registry()


@pytest.fixture(scope="module")
def routing(registry):
    return load_routing(registry)


def models_called(deps):
    return [c["model"] for c in deps.fake_models.calls]


def route(deps, *, tau=0.5, model="auto"):
    async def go():
        return await route_and_stream(ASK, tau=tau, task_type=None, model=model, stream=False,
                                      settings=GenerationSettings(), deps=deps, request_id="req-1")
    return anyio.run(go)


# ---------- the real config ----------

def test_server_providers_are_groq_gemini_ollama(routing):
    assert routing.server_providers == frozenset({"groq", "gemini", "ollama"})


def test_requests_without_keys_get_exactly_the_phase4_chains(routing, registry):
    for tier, chain in PHASE4_CHAINS.items():
        assert routing.chain_for(tier, registry) == chain, tier


def test_byok_models_come_after_every_server_model(routing, registry):
    for tier, names in routing.chains.items():
        providers = [registry[n].provider for n in names]
        first_byok = next(i for i, p in enumerate(providers) if p in BYOK_PROVIDERS)
        assert not set(providers[first_byok:]) - BYOK_PROVIDERS, f"{tier}: a server model after a BYOK one"
        assert names[0] == PHASE4_CHAINS[tier][0]


def test_each_byok_provider_has_one_model_per_tier(routing, registry):
    for provider in BYOK_PROVIDERS:
        for tier in ("low", "medium", "high"):
            chain = routing.chain_for(tier, registry, {provider})
            assert len(chain) == 1 and registry[chain[0]].tier == tier, (provider, tier)


def test_byok_models_are_paid_at_their_real_price(registry):
    for m in registry.values():
        if m.provider in BYOK_PROVIDERS:
            assert m.input_price > 0 and m.output_price > 0 and m.price_as is None, m.name


def test_gemini_flash_lite_is_in_the_registry_but_no_chain(routing, registry):
    assert registry["gemini-flash-lite"].model_id == "models/gemini-3.5-flash-lite"
    assert all("gemini-flash-lite" not in names for names in routing.chains.values())


# ---------- strong baseline ----------

def test_strong_baseline_is_unchanged_and_follows_the_filtered_chain():
    deps = make_deps()
    assert deps.strong_model() == "groq-large"
    assert deps.strong_model({"anthropic"}) == "claude-opus"
    assert deps.strong_model({"openai", "anthropic"}) == "openai-sol"     # chain order decides
    with pytest.raises(LookupError):
        deps.strong_model({"openrouter"})


# ---------- the filter in ReliableClient ----------

def model(name, provider, tier="low"):
    return ModelConfig(name=name, provider=provider, model_id=name, tier=tier, input_price=0, output_price=0)


REGISTRY = {"g1": model("g1", "groq"), "o1": model("o1", "openai"), "a1": model("a1", "anthropic")}


class Send:
    def __init__(self, fail=()):
        self.fail, self.calls = set(fail), []

    def __call__(self, messages, m, stream=False, **settings):
        self.calls.append(m.name)
        if m.name in self.fail:
            raise ProviderError(m.provider, "server", "down")
        return f"answer from {m.name}"


def client(send, server_providers=frozenset({"groq"})):
    routing = RoutingConfig(chains={"low": ["g1", "o1", "a1"]}, retry=RetryPolicy(max_attempts=1),
                            breaker=BreakerSettings(), timeout_seconds=8, server_providers=server_providers)
    return ReliableClient(REGISTRY, routing, send_fn=send, sleep=lambda s: None, rng=random.Random(0))


def test_without_keys_a_failing_server_model_never_falls_back_to_byok():
    send = Send(fail={"g1"})
    c = client(send)
    with pytest.raises(AllModelsFailed) as error:
        c.send(ASK, "low")
    assert send.calls == ["g1"]
    assert [a.model_name for a in error.value.attempts] == ["g1"]               # no "skipped" rows for o1 / a1
    assert [e.from_model for e in c.fallback_events] == ["g1"] and c.fallback_events[0].to_model is None


def test_byok_providers_use_only_their_own_models():
    send = Send(fail={"o1"})
    result = client(send).send(ASK, "low", providers={"openai", "anthropic"})
    assert send.calls == ["o1", "a1"] and result.model.name == "a1" and result.primary == "o1"


def test_a_tier_with_no_allowed_model_fails_without_calling_anything():
    send = Send()
    with pytest.raises(AllModelsFailed) as error:
        client(send).send(ASK, "low", providers={"openrouter"})
    assert send.calls == [] and error.value.attempts == []


def test_pass_through_to_a_byok_model_without_its_key_is_refused_before_calling():
    send = Send()
    with pytest.raises(ModelNotAllowed, match="anthropic"):
        client(send).send_model(ASK, "a1")
    assert send.calls == []
    assert client(send).send_model(ASK, "a1", providers={"anthropic"}).model.name == "a1"


def test_no_server_providers_means_no_filter():
    send = Send(fail={"g1"})
    assert client(send, server_providers=None).send(ASK, "low").model.name == "o1"


def test_filter_chain_keeps_order():
    assert filter_chain(["a1", "g1", "o1"], REGISTRY, frozenset({"openai", "anthropic"})) == ["a1", "o1"]
    assert filter_chain(["a1", "g1"], REGISTRY, None) == ["a1", "g1"]


# ---------- routing.yaml validation ----------

def write(tmp_path, extra):
    path = tmp_path / "routing.yaml"
    path.write_text("fallback_chains:\n  low: [g1, o1]\n  medium: [g1]\n  high: [g1, a1]\n" + extra)
    return path


@pytest.mark.parametrize("extra, message", [
    ("server_providers: [groq, nope]\n", "unknown provider"),
    ("server_providers: [groq, groq]\n", "twice"),
    ("server_providers: []\n", "non-empty list"),
    ("server_providers: groq\n", "non-empty list"),
    ("server_providers: [openai]\n", "no model on a server provider"),    # medium and high would have none
])
def test_bad_server_providers_are_rejected(tmp_path, extra, message):
    with pytest.raises(ValueError, match=message):
        load_routing(REGISTRY, write(tmp_path, extra))


def test_server_providers_left_out_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="'server_providers' is missing.*refuses to start"):
        load_routing(REGISTRY, write(tmp_path, ""))


def test_startup_config_without_server_providers_refuses_to_load(tmp_path, registry):
    """The path the app takes at startup (build_deps -> ActiveConfig.from_file)."""
    raw = yaml.safe_load(DEFAULT_ROUTING_PATH.read_text(encoding="utf-8"))
    del raw["server_providers"]
    path = tmp_path / "routing.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="'server_providers' is missing"):
        ActiveConfig.from_file(registry, path)


def test_live_config_update_keeps_the_filter():
    deps = make_deps()
    snap = deps.update_config({"fallback_chains": {"medium": ["groq-small", "claude-sonnet", "gemini-flash"]}})
    assert snap.routing.server_providers == frozenset({"groq", "gemini", "ollama"})
    assert snap.routing.chain_for("medium", deps.registry) == ["groq-small", "gemini-flash"]


def test_real_routing_yaml_sets_server_providers_explicitly():
    """A missing line would mean "no filter", so the real file must say it."""
    raw = yaml.safe_load(DEFAULT_ROUTING_PATH.read_text(encoding="utf-8"))
    assert sorted(raw["server_providers"]) == ["gemini", "groq", "ollama"]


# ---------- through route_and_stream and the API ----------

@pytest.mark.parametrize("tau, tier", [(0.5, "low"), (0.92, "medium"), (0.99, "high")])
def test_auto_without_keys_never_tries_a_byok_model_even_when_every_server_model_fails(tau, tier):
    deps = make_deps({name: DOWN for name in PHASE4_CHAINS[tier]})
    with pytest.raises(AllModelsFailed):
        route(deps, tau=tau)
    called = set(models_called(deps))
    assert called == set(PHASE4_CHAINS[tier]), called


def test_auto_without_keys_still_answers_from_the_phase4_backup():
    deps = make_deps({"groq-small": DOWN})
    routed = route(deps, tau=0.92)
    assert (routed.decision.primary_model, routed.decision.model) == ("groq-small", "gemini-flash")


def test_api_pass_through_to_a_byok_model_without_a_key_is_403_and_logged():
    deps = make_deps()
    http = TestClient(create_app(deps=deps, auth=Auth([API_KEY])))
    r = http.post("/v1/chat/completions", json={"model": "claude-opus", "messages": ASK},
                  headers={"Authorization": f"Bearer {API_KEY}"})
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "model_not_allowed" and r.json()["error"]["param"] == "model"
    assert "anthropic" in r.json()["error"]["message"]
    assert deps.fake_models.calls == []
    (row,) = deps.log.rows()
    assert (row.status, row.http_status, row.error_kind) == ("error", 403, "model_not_allowed")


def test_api_models_says_which_models_require_a_key():
    deps = make_deps()
    http = TestClient(create_app(deps=deps, auth=Auth([API_KEY])))
    data = {m["id"]: m for m in http.get("/v1/models", headers={"Authorization": f"Bearer {API_KEY}"}).json()["data"]}
    needs_key = {name for name, m in data.items() if name != "auto" and m["triage"]["requires_key"]}
    assert needs_key == {n for n, m in deps.registry.items() if m.provider in BYOK_PROVIDERS}
    assert data["groq-small"]["triage"]["requires_key"] is False
    assert data["claude-opus"]["triage"]["requires_key"] is True
    assert "requires_key" not in data["auto"]["triage"]


def test_api_auto_without_keys_answers_from_a_server_model():
    deps = make_deps(predictor=FakePredictor(0.4, 0.95))
    http = TestClient(create_app(deps=deps, auth=Auth([API_KEY])))
    r = http.post("/v1/chat/completions", json={"model": "auto", "messages": ASK},
                  headers={"Authorization": f"Bearer {API_KEY}"})
    assert r.status_code == 200 and r.json()["model"] == "groq-small"
    assert r.json()["choices"][0]["message"]["content"] == fake_text("groq-small")


# ---------- scripts: providers with no key are skipped, not failed ----------

def test_every_provider_has_a_key_env_entry():
    assert set(PROVIDER_KEY_ENV) == VALID_PROVIDERS


def test_missing_key_env(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "x")
    assert missing_key_env("anthropic") == "ANTHROPIC_API_KEY"
    assert missing_key_env("groq") is None
    assert missing_key_env("ollama") is None                 # needs no key


def test_try_models_skips_providers_without_a_key(monkeypatch, capsys):
    from scripts import try_models
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "x")
    sent = []

    def fake_send(messages, m):
        sent.append(m.name)
        raise ProviderError(m.provider, "server", "down")

    monkeypatch.setattr(try_models, "send_request", fake_send)
    try_models.try_every_model({"g1": REGISTRY["g1"], "o1": REGISTRY["o1"]})
    out = capsys.readouterr().out
    assert sent == ["g1"] and "[o1] skipped: no key (OPENAI_API_KEY is not set)" in out


def test_final_check_reports_no_key_as_skipped_not_failed(monkeypatch):
    from scripts import final_check
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(final_check, "results", [])
    monkeypatch.setattr(final_check, "skipped", [])
    final_check.check_every_model({"a1": REGISTRY["a1"]})
    assert final_check.results == []
    assert final_check.skipped == [("a1 answers", "no key: ANTHROPIC_API_KEY is not set")]
