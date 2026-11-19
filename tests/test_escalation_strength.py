"""Escalation never serves a weaker model: an escalation only tries models whose own tier (models.yaml) is at
least the tier of the answer it replaces. gemini-flash-lite (low) is the last resort of medium and high for normal
routing, never for an escalation away from a medium answer. Fake models only."""
import copy
import random

import pytest
from fastapi.testclient import TestClient

from app.api.auth import Auth
from app.api.fake_models import FakeBehavior
from app.api.main import create_app
from app.api.verification import DEFAULT_SECTION, escalation_tier
from app.providers.common import ProviderError
from app.providers.keys import KeyContext
from app.registry import load_registry
from app.reliability import AllModelsFailed, ReliableClient, at_least, load_routing
from tests.api_helpers import API_KEY, FakePredictor, make_deps, routing_raw

REGISTRY = load_registry()
MESSAGES = [{"role": "user", "content": "Hi"}]
RAIN = [{"role": "user", "content": "Write two sentences about rain."}]
ASK = [{"role": "user", "content": "What is 2 + 3? End with: Answer: <number>"}]
BAD_CHECK = '{"good": false, "reason": "the answer is wrong"}'
MEDIUM, LOW = FakePredictor(0.4, 0.95), FakePredictor(0.95, 0.97)
DOWN = FakeBehavior(fail_before="server")


def app_with(script, predictor=MEDIUM, medium_skip_check_above=None):
    raw = routing_raw()
    raw["verification"] = copy.deepcopy(DEFAULT_SECTION)
    # medium always runs the AI check unless a test says otherwise
    raw["verification"]["sync"]["tiers"]["medium"]["skip_check_above"] = medium_skip_check_above
    deps = make_deps(script, predictor=predictor, raw=raw)
    return TestClient(create_app(deps=deps, auth=Auth([API_KEY]))), deps


def post(http, messages=RAIN):
    return http.post("/v1/chat/completions", json={"model": "auto", "messages": messages},
                     headers={"Authorization": f"Bearer {API_KEY}", "X-Triage-Verify": "sync"})


def answer_calls(deps):
    return [c["model"] for c in deps.fake_models.calls if c["kind"] == "answer"]


# ---------- through the API ----------

def test_a_bad_medium_answer_is_never_replaced_by_flash_lite():
    """High's strong models fail; Flash-Lite would answer, but it is weaker than the medium answer: the
    original is served and Flash-Lite is never called."""
    http, deps = app_with({"groq-small": FakeBehavior(check_reply=BAD_CHECK),
                           "groq-large": DOWN, "gemini-flash": DOWN})
    r = post(http)
    assert r.status_code == 200 and r.json()["model"] == "groq-small"
    assert r.headers["x-triage-escalated"] == "false"
    assert "gemini-flash-lite" not in answer_calls(deps)
    row = deps.log.rows()[-1]
    assert row.error_kind == "all_models_failed:high" and row.checker_verdict == "bad"


def test_the_escalation_still_falls_back_among_strong_enough_models():
    http, deps = app_with({"groq-small": FakeBehavior(check_reply=BAD_CHECK), "groq-large": DOWN})
    r = post(http)
    assert r.json()["model"] == "gemini-flash" and r.headers["x-triage-escalated"] == "true"


def test_flash_lite_stays_a_normal_fallback_outside_escalation():
    http, deps = app_with({"groq-small": DOWN, "gemini-flash": DOWN})
    r = http.post("/v1/chat/completions", json={"model": "auto", "messages": RAIN},
                  headers={"Authorization": f"Bearer {API_KEY}"})
    assert r.status_code == 200 and r.json()["model"] == "gemini-flash-lite"


def test_a_low_answer_may_escalate_to_a_model_of_the_same_tier():
    """"At least as strong": a bad low answer's escalation to medium may end on Flash-Lite (low too)."""
    # groq-small is also the AI checker, so medium's AI check is skipped here (LOW's p_medium 0.97 is above 0.5): this test is
    # about which models the escalation may use, not about the checker failing
    http, deps = app_with({"groq-allam": FakeBehavior(text="five"), "groq-small": DOWN, "gemini-flash": DOWN,
                           "gemini-flash-lite": FakeBehavior(text="2 + 3 = 5.\nAnswer: 5")}, predictor=LOW,
                          medium_skip_check_above=0.5)
    r = post(http, ASK)
    assert r.json()["model"] == "gemini-flash-lite" and r.headers["x-triage-escalated"] == "true"


# ---------- the client ----------

class Send:
    def __init__(self, down=()):
        self.down, self.calls = set(down), []

    def __call__(self, messages, model, stream=False, **settings):
        self.calls.append(model.name)
        if model.name in self.down:
            raise ProviderError(model.provider, "server", "down")
        return f"answer from {model.name}"


def client(send):
    return ReliableClient(REGISTRY, load_routing(REGISTRY), send_fn=send, sleep=lambda s: None,
                          rng=random.Random(0))


def test_min_tier_drops_weaker_models_before_anything_is_tried():
    send = Send(down={"groq-large", "gemini-flash"})
    with pytest.raises(AllModelsFailed):
        client(send).send(MESSAGES, "high", min_tier="medium")
    assert "gemini-flash-lite" not in send.calls
    send = Send(down={"groq-large", "gemini-flash"})
    assert client(send).send(MESSAGES, "high").model.name == "gemini-flash-lite"        # no floor: as before


@pytest.mark.parametrize("floor, expected", [
    (None, ["groq-small", "gemini-flash", "gemini-flash-lite"]),
    ("low", ["groq-small", "gemini-flash", "gemini-flash-lite"]),
    ("medium", ["groq-small", "gemini-flash"]),
    ("high", ["gemini-flash"]),
])
def test_at_least_uses_each_models_own_tier(floor, expected):
    assert at_least(["groq-small", "gemini-flash", "gemini-flash-lite"], REGISTRY, floor) == expected


# ---------- BYOK: the nearest tier with a strong-enough model ----------

class Deps:
    registry = REGISTRY

    class client:
        routing = load_routing(REGISTRY)


def byok(*providers):
    return KeyContext("byok", {p: "sk-test" for p in providers})


def test_byok_escalation_tier_needs_a_strong_enough_model():
    # Gemini only, from medium to high: gemini-flash (high) qualifies
    assert escalation_tier("high", Deps, byok("gemini"), min_tier="medium") == "high"
    # Groq only, a high answer escalating "to medium": medium's groq-small is weaker, so the high tier is used
    assert escalation_tier("medium", Deps, byok("groq"), min_tier="high") == "high"
    # no floor: as before
    assert escalation_tier("medium", Deps, byok("groq")) == "medium"
