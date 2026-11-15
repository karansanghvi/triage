"""BYOK preferred provider (X-Triage-Prefer-Provider): within each tier's chain, after the provider filter, the
preferred provider's models go first and the rest keep their order. No network."""
import pytest

from app.api.fake_models import FakeBehavior
from app.providers.keys import KeyContext
from app.registry import load_registry
from app.reliability import load_routing, prefer_chain
from tests.api_helpers import ASK, make_deps

REGISTRY = load_registry()
ROUTING = load_routing(REGISTRY)
SECRET = "sk-test-SECRET123"
ALL_FOUR = ("openai", "anthropic", "groq", "gemini")


def byok(*providers):
    return KeyContext("byok", {p: SECRET for p in providers})


# ---------- reordering the chains ----------

@pytest.mark.parametrize("prefer", ALL_FOUR)
@pytest.mark.parametrize("tier", ["low", "medium", "high"])
def test_the_preferred_providers_models_go_first_and_the_rest_keep_their_order(tier, prefer):
    filtered = ROUTING.chain_for(tier, REGISTRY, ALL_FOUR)
    chain = ROUTING.chain_for(tier, REGISTRY, ALL_FOUR, prefer)
    first = [n for n in filtered if REGISTRY[n].provider == prefer]
    assert first and chain == first + [n for n in filtered if REGISTRY[n].provider != prefer]
    assert sorted(chain) == sorted(filtered)                      # only reordered


def test_the_high_chain_with_each_preference():
    assert ROUTING.chain_for("high", REGISTRY, ALL_FOUR) == ["groq-large", "gemini-flash", "openai-sol",
                                                             "claude-opus"]
    assert ROUTING.chain_for("high", REGISTRY, ALL_FOUR, "anthropic") == ["claude-opus", "groq-large",
                                                                          "gemini-flash", "openai-sol"]
    assert ROUTING.chain_for("high", REGISTRY, ALL_FOUR, "openai")[0] == "openai-sol"
    assert ROUTING.chain_for("high", REGISTRY, ALL_FOUR, "gemini")[0] == "gemini-flash"
    assert ROUTING.chain_for("high", REGISTRY, ALL_FOUR, "groq") == ROUTING.chain_for("high", REGISTRY, ALL_FOUR)


def test_reordering_happens_after_the_provider_filter():
    chain = ROUTING.chain_for("low", REGISTRY, ("groq", "anthropic"), "anthropic")
    assert chain == ["claude-haiku", "groq-allam"]                 # no ollama, gemini or openai model is added
    assert ROUTING.chain_for("low", REGISTRY, ("groq",), "anthropic") == ["groq-allam"]


def test_no_preference_leaves_every_chain_unchanged():
    for tier in ("low", "medium", "high"):
        for providers in (None, ALL_FOUR, ("groq",)):
            assert ROUTING.chain_for(tier, REGISTRY, providers, None) == ROUTING.chain_for(tier, REGISTRY, providers)


def test_prefer_chain_keeps_several_preferred_models_in_order():
    names = ["groq-allam", "claude-haiku", "groq-small", "claude-sonnet"]
    assert prefer_chain(names, REGISTRY, "anthropic") == ["claude-haiku", "claude-sonnet", "groq-allam",
                                                          "groq-small"]
    assert prefer_chain(names, REGISTRY, None) == names


# ---------- ReliableClient.send ----------

def test_send_tries_the_preferred_model_first():
    deps = make_deps()
    result = deps.client.send(ASK, "high", keys=byok("groq", "anthropic"), prefer="anthropic")
    assert (result.model.name, result.primary) == ("claude-opus", "claude-opus")


def test_send_falls_back_along_the_reordered_chain():
    deps = make_deps({"claude-opus": FakeBehavior(fail_before="server")})
    result = deps.client.send(ASK, "high", keys=byok("groq", "anthropic"), prefer="anthropic")
    assert (result.primary, result.model.name) == ("claude-opus", "groq-large")
    assert [a.model_name for a in result.attempts][0] == "claude-opus"


def test_send_without_a_preference_is_unchanged():
    deps = make_deps()
    assert deps.client.send(ASK, "high", keys=byok("groq", "anthropic")).model.name == "groq-large"
