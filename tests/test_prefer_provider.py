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


# ---------- the X-Triage-Prefer-Provider header ----------

from fastapi.testclient import TestClient                          # noqa: E402

from app.api.auth import Auth                                       # noqa: E402
from app.api.main import create_app                                 # noqa: E402
from tests.api_helpers import API_KEY, FakePredictor, routing_raw   # noqa: E402

BEARER = {"Authorization": f"Bearer {API_KEY}"}
ALL_KEYS = {"x-openai-key": SECRET, "x-anthropic-key": SECRET, "x-groq-key": SECRET, "x-gemini-key": SECRET}
LOW_FIRST = {"openai": "openai-luna", "anthropic": "claude-haiku", "groq": "groq-allam",
             "gemini": "gemini-flash-lite"}


def app(predictor=None, public=True, **kw):
    deps = make_deps(predictor=predictor or FakePredictor(0.95, 0.95), **kw)
    return TestClient(create_app(deps=deps, auth=Auth([API_KEY]), public=public)), deps


def chat(http, headers=None, body=None):
    return http.post("/v1/chat/completions", json=body or {"model": "auto", "messages": ASK}, headers=headers or {})


def not_applied(r, served):
    assert r.status_code == 200 and r.headers["x-triage-preferred-provider"] == "none"
    assert r.json()["triage"]["preferred_provider"] is None
    assert "preferred provider" not in r.headers["x-triage-reason"]
    assert r.json()["model"] == served


@pytest.mark.parametrize("prefer", ALL_FOUR)
def test_byok_serves_the_preferred_providers_model_in_the_same_tier(prefer):
    http, _ = app()
    baseline = chat(http, ALL_KEYS)
    r = chat(http, ALL_KEYS | {"X-Triage-Prefer-Provider": prefer})
    assert r.status_code == 200 and r.headers["x-triage-tier"] == baseline.headers["x-triage-tier"] == "low"
    assert r.json()["model"] == LOW_FIRST[prefer]
    assert r.headers["x-triage-preferred-provider"] == prefer and r.json()["triage"]["preferred_provider"] == prefer
    assert r.headers["x-triage-reason"].endswith(f"; preferred provider {prefer}")
    assert r.headers["x-triage-fallback"] == "none"


def test_the_value_is_case_insensitive():
    http, _ = app()
    r = chat(http, ALL_KEYS | {"X-Triage-Prefer-Provider": " Anthropic "})
    assert r.headers["x-triage-preferred-provider"] == "anthropic" and r.json()["model"] == "claude-haiku"


def test_the_tier_decision_does_not_change():
    http, _ = app(FakePredictor(0.4, 0.95))
    r = chat(http, ALL_KEYS | {"X-Triage-Prefer-Provider": "openai"})
    assert (r.headers["x-triage-tier"], r.json()["model"]) == ("medium", "openai-terra")


def test_fallback_walks_the_reordered_chain():
    http, deps = app(script={"claude-haiku": FakeBehavior(fail_before="server")})
    r = chat(http, {"x-anthropic-key": SECRET, "x-groq-key": SECRET, "X-Triage-Prefer-Provider": "anthropic"})
    assert r.json()["model"] == "groq-allam" and r.headers["x-triage-fallback"] == "claude-haiku->groq-allam"
    assert r.headers["x-triage-preferred-provider"] == "anthropic"


def test_tier_fallback_works_on_the_reordered_chains():
    raw = routing_raw()
    raw["fallback_chains"]["medium"] = ["groq-small"]                  # no anthropic model in medium
    http, _ = app(FakePredictor(0.4, 0.95), raw=raw)
    r = chat(http, {"x-anthropic-key": SECRET, "X-Triage-Prefer-Provider": "anthropic"})
    assert (r.headers["x-triage-tier"], r.json()["model"]) == ("high", "claude-opus")
    assert r.headers["x-triage-reason"].endswith(
        "; no model for your keys in medium: used high; preferred provider anthropic")


def test_ignored_without_a_key_for_that_provider():
    http, _ = app()
    not_applied(chat(http, {"x-groq-key": SECRET, "X-Triage-Prefer-Provider": "anthropic"}), "groq-allam")


def test_ignored_in_api_mode():
    http, deps = app()
    r = chat(http, BEARER | {"X-Triage-Prefer-Provider": "gemini"})
    not_applied(r, "groq-allam")
    assert r.headers["x-triage-mode"] == "api"


def test_ignored_in_demo_mode_and_routed_exactly_as_without_it():
    http, _ = app()
    without = chat(http)
    r = chat(http, {"X-Triage-Prefer-Provider": "gemini"})
    not_applied(r, without.json()["model"])
    assert r.headers["x-triage-mode"] == "demo" and r.headers["x-triage-reason"] == without.headers["x-triage-reason"]


def test_ignored_for_a_named_model():
    http, _ = app()
    r = chat(http, ALL_KEYS | {"X-Triage-Prefer-Provider": "anthropic"}, {"model": "groq-small", "messages": ASK})
    not_applied(r, "groq-small")


@pytest.mark.parametrize("headers", [ALL_KEYS, BEARER, {}])
@pytest.mark.parametrize("value", ["ollama", "claude", "openai,anthropic"])
def test_an_unknown_provider_is_a_400_in_every_mode(headers, value):
    http, deps = app()
    r = chat(http, headers | {"X-Triage-Prefer-Provider": value})
    assert r.status_code == 400
    error = r.json()["error"]
    assert (error["code"], error["param"], error["type"]) == ("invalid_value", "X-Triage-Prefer-Provider",
                                                              "invalid_request_error")
    assert deps.fake_models.calls == [] and deps.log.rows() == []


def test_a_blank_value_is_no_preference():
    http, _ = app()
    not_applied(chat(http, ALL_KEYS | {"X-Triage-Prefer-Provider": " "}), "groq-allam")


def test_streams_report_it_too():
    http, _ = app()
    with http.stream("POST", "/v1/chat/completions", headers=ALL_KEYS | {"X-Triage-Prefer-Provider": "openai"},
                     json={"model": "auto", "messages": ASK, "stream": True}) as r:
        "".join(r.iter_text())
        assert r.headers["x-triage-preferred-provider"] == "openai" and r.headers["x-triage-model"] == "openai-luna"


# ---------- the savings baseline and verification follow the preference ----------

def strong_cost_of(deps, name, row):
    return deps.registry[name].reported_cost(row.input_tokens, row.output_tokens)


def test_strong_model_uses_the_reordered_high_chain():
    deps = make_deps()
    assert deps.strong_model(("groq", "anthropic"), "anthropic") == "claude-opus"
    assert deps.strong_model(("groq", "anthropic"), "openai") == "groq-large"      # no key: nothing to move
    assert deps.strong_model(("groq", "anthropic")) == "groq-large"
    assert deps.strong_model() == "groq-large"                                     # server baseline unchanged


def test_the_baseline_follows_the_preference():
    http, deps = app()
    keys = {"x-anthropic-key": SECRET, "x-groq-key": SECRET}
    r = chat(http, keys | {"X-Triage-Prefer-Provider": "anthropic"})
    plain = chat(http, keys)
    preferred_row, plain_row = deps.log.rows()
    assert preferred_row.strong_cost == pytest.approx(strong_cost_of(deps, "claude-opus", preferred_row))
    assert plain_row.strong_cost == pytest.approx(strong_cost_of(deps, "groq-large", plain_row))
    assert float(r.headers["x-triage-strong-cost"]) == pytest.approx(preferred_row.strong_cost, abs=1e-8)
    assert r.json()["triage"]["saved"] == pytest.approx(preferred_row.strong_cost - preferred_row.price_as_cost,
                                                        abs=1e-8)
    assert float(plain.headers["x-triage-strong-cost"]) == pytest.approx(plain_row.strong_cost, abs=1e-8)


def test_a_streams_baseline_follows_the_preference_too():
    http, deps = app()
    with http.stream("POST", "/v1/chat/completions",
                     headers={"x-anthropic-key": SECRET, "x-groq-key": SECRET, "X-Triage-Prefer-Provider": "anthropic"},
                     json={"model": "auto", "messages": ASK, "stream": True}) as r:
        "".join(r.iter_text())
    (row,) = deps.log.rows()
    assert row.strong_cost == pytest.approx(strong_cost_of(deps, "claude-opus", row))


def test_an_ignored_preference_keeps_the_usual_baseline():
    http, deps = app()
    chat(http, BEARER | {"X-Triage-Prefer-Provider": "anthropic"})               # api mode: ignored
    (row,) = deps.log.rows()
    assert row.strong_cost == pytest.approx(strong_cost_of(deps, "groq-large", row))


GOOD = "2 + 3 = 5.\nAnswer: 5"           # passes the free format check for ASK
SYNC = {"X-Triage-Verify": "sync", "X-Triage-Quality": "0.1"}


def test_escalation_follows_the_preference():
    http, deps = app(FakePredictor(0.95, 0.85), script={"claude-haiku": FakeBehavior(text="five"),
                                                        "claude-sonnet": FakeBehavior(text=GOOD)})
    keys = {"x-anthropic-key": SECRET, "x-groq-key": SECRET}
    r = chat(http, keys | SYNC | {"X-Triage-Prefer-Provider": "anthropic"})
    assert r.status_code == 200 and r.headers["x-triage-escalated"] == "true"
    assert r.json()["model"] == "claude-sonnet" and r.headers["x-triage-preferred-provider"] == "anthropic"
    (row,) = deps.log.rows()
    assert (row.escalated_from, row.escalated_to) == ("low", "medium")
    assert row.strong_cost == pytest.approx(strong_cost_of(deps, "claude-opus", row))


def test_escalation_without_a_preference_is_unchanged():
    http, _ = app(FakePredictor(0.95, 0.85), script={"groq-allam": FakeBehavior(text="five"),
                                                     "groq-small": FakeBehavior(text=GOOD)})
    r = chat(http, {"x-anthropic-key": SECRET, "x-groq-key": SECRET} | SYNC)
    assert r.headers["x-triage-escalated"] == "true" and r.json()["model"] == "groq-small"
