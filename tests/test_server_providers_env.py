"""TRIAGE_SERVER_PROVIDERS: a per-environment override of routing.yaml's server_providers (Phase 5, demo mode).

A free host can't run Ollama, so the deployed server sets TRIAGE_SERVER_PROVIDERS=groq,gemini. No network.
"""
import pytest

from app.api import deps as deps_module
from app.api.deps import load_config, server_providers_from_env
from app.registry import load_registry
from tests.api_helpers import ASK, make_deps, routing_raw

REGISTRY = load_registry()


@pytest.mark.parametrize("value, expected", [(None, None), ("", None), (" , ", None),
                                             ("groq,gemini", ["groq", "gemini"]),
                                             (" groq , gemini ", ["groq", "gemini"])])
def test_parsing(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("TRIAGE_SERVER_PROVIDERS", raising=False)
    else:
        monkeypatch.setenv("TRIAGE_SERVER_PROVIDERS", value)
    assert server_providers_from_env() == expected


def test_unset_keeps_routing_yaml(monkeypatch):
    monkeypatch.delenv("TRIAGE_SERVER_PROVIDERS", raising=False)
    routing = load_config(REGISTRY).snapshot().routing
    assert routing.server_providers == frozenset({"groq", "gemini", "ollama"})


def test_override_excludes_ollama_from_every_chain(monkeypatch):
    monkeypatch.setenv("TRIAGE_SERVER_PROVIDERS", "groq,gemini")
    config = load_config(REGISTRY)
    routing = config.snapshot().routing
    assert routing.server_providers == frozenset({"groq", "gemini"})
    for tier in ("low", "medium", "high"):
        chain = routing.chain_for(tier, REGISTRY)
        assert chain and all(REGISTRY[n].provider in {"groq", "gemini"} for n in chain), (tier, chain)


def test_override_routes_low_tier_traffic_without_ollama(monkeypatch):
    from app.api.fake_models import FakeBehavior
    raw = routing_raw()
    raw["server_providers"] = ["groq", "gemini"]
    deps = make_deps({"groq-allam": FakeBehavior(fail_before="server")}, raw=raw)
    from app.reliability import AllModelsFailed
    try:
        deps.client.send(ASK, "low")
    except AllModelsFailed:
        pass
    called = {c["model"] for c in deps.fake_models.calls}
    assert "ollama-llama" not in called and "groq-allam" in called


@pytest.mark.parametrize("value, message", [("groq,nope", "unknown provider"),
                                            ("openrouter", "no model on a server provider")])
def test_bad_override_refuses_to_start(monkeypatch, value, message):
    monkeypatch.setenv("TRIAGE_SERVER_PROVIDERS", value)
    with pytest.raises(RuntimeError, match=f"refusing to start: TRIAGE_SERVER_PROVIDERS={value}.*{message}"):
        load_config(REGISTRY)


def test_models_endpoint_requires_key_follows_the_override(monkeypatch):
    from fastapi.testclient import TestClient
    from app.api.auth import Auth
    from app.api.main import create_app
    from tests.api_helpers import API_KEY
    raw = routing_raw()
    raw["server_providers"] = ["groq", "gemini"]
    deps = make_deps(raw=raw)
    http = TestClient(create_app(deps=deps, auth=Auth([API_KEY])))
    data = {m["id"]: m for m in http.get("/v1/models", headers={"Authorization": f"Bearer {API_KEY}"}).json()["data"]}
    assert data["ollama-llama"]["triage"]["requires_key"] is True
    assert data["groq-allam"]["triage"]["requires_key"] is False


def test_load_deps_uses_the_override(monkeypatch):
    """load_deps builds its config through load_config (the override), not routing.yaml alone."""
    seen = {}
    monkeypatch.setattr(deps_module, "load_config", lambda registry: seen.setdefault("called", True) or None)
    monkeypatch.setattr(deps_module, "load_predictor_checked", lambda: (_ for _ in ()).throw(SystemExit("stop")))
    monkeypatch.setenv("TRIAGE_LOG", "memory")
    monkeypatch.setenv("TRIAGE_CACHE", "off")
    with pytest.raises(SystemExit):
        deps_module.load_deps()
    assert seen == {"called": True}


def test_deployed_low_chain_is_groq_allam_then_gemini_flash_lite(monkeypatch):
    monkeypatch.setenv("TRIAGE_SERVER_PROVIDERS", "groq,gemini")
    routing = load_config(REGISTRY).snapshot().routing
    assert routing.chain_for("low", REGISTRY) == ["groq-allam", "gemini-flash-lite"]
    assert routing.chain_for("medium", REGISTRY) == ["groq-small", "gemini-flash"]
    assert routing.chain_for("high", REGISTRY) == ["groq-large", "gemini-flash"]
