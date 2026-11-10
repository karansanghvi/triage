"""route_and_stream (Phase 4 Step 5). These fail until app/api/core.py's route_and_stream is written by hand.

Fake models only (app/api/fake_models.py) behind the real ReliableClient; a fake predictor.
"""
import time

import anyio
import pytest

from app.api.core import GenerationSettings, RoutedStream, UnknownModel, route_and_stream
from app.api.fake_models import FakeBehavior, fake_text
from app.cache import ResponseCache
from app.providers.common import ProviderError
from app.reliability import AllModelsFailed
from app.router.quality import route
from tests.api_helpers import ASK, FakePredictor, make_deps

DEFAULT = GenerationSettings()


def call(deps, *, tau=0.5, task=None, model="auto", stream=True, settings=DEFAULT, request_id="req-1"):
    async def go():
        return await route_and_stream(ASK, tau=tau, task_type=task, model=model, stream=stream, settings=settings,
                                      deps=deps, request_id=request_id)
    return anyio.run(go)


def call_and_read(deps, **kw):
    """(RoutedStream, pieces, response seen before reading, response after)."""
    async def go():
        routed = await route_and_stream(ASK, tau=kw.get("tau", 0.5), task_type=kw.get("task"),
                                        model=kw.get("model", "auto"), stream=kw.get("stream", True),
                                        settings=kw.get("settings", DEFAULT), deps=deps, request_id="req-1")
        before = routed.response
        pieces = [p async for p in routed.pieces]
        return routed, pieces, before, routed.response
    return anyio.run(go)


def models_called(deps):
    return [c["model"] for c in deps.fake_models.calls]


# ---------- model="auto": the route follows tau ----------

@pytest.mark.parametrize("tau,tier,model", [(0.5, "low", "groq-allam"), (0.92, "medium", "groq-small"),
                                            (0.99, "high", "groq-large")])
def test_auto_picks_the_tier_from_tau(tau, tier, model):
    deps = make_deps(predictor=FakePredictor(p_low=0.9, p_medium=0.95))
    d = call(deps, tau=tau).decision
    assert (d.tier, d.primary_model, d.model) == (tier, model, model)
    assert models_called(deps) == [model]


def test_auto_decision_records_the_prediction_and_reason():
    deps = make_deps(predictor=FakePredictor(p_low=0.4, p_medium=0.8))
    d = call(deps, tau=0.7, request_id="req-42").decision
    assert d.request_id == "req-42" and d.requested_model == "auto"
    assert (d.tau, d.p_low, d.p_medium) == (0.7, 0.4, 0.8)
    assert d.reason == route(0.4, 0.8, 0.7).reason
    assert not d.fell_back and d.attempts and d.attempts[-1].outcome == "success"


def test_predictor_gets_the_messages_and_the_task_type_once():
    pred = FakePredictor()
    call(make_deps(predictor=pred), task="coding")
    assert pred.calls == [(ASK, "coding")]


def test_given_and_guessed_task_types_are_recorded():
    d = call(make_deps(), task="coding").decision
    assert (d.task_type, d.task_type_guessed) == ("coding", False)
    d = call(make_deps(predictor=FakePredictor(guessed_task="reasoning")), task=None).decision
    assert (d.task_type, d.task_type_guessed) == ("reasoning", True)


# ---------- pass-through and unknown models ----------

def test_pass_through_skips_the_predictor_and_calls_only_that_model():
    pred = FakePredictor()
    deps = make_deps(predictor=pred)
    d = call(deps, model="groq-large", tau=None).decision
    assert pred.calls == []
    assert (d.requested_model, d.tier, d.primary_model, d.model) == ("groq-large", "high", "groq-large", "groq-large")
    assert d.tau is None and d.p_low is None and d.p_medium is None
    assert "groq-large" in d.reason
    assert models_called(deps) == ["groq-large"]


def test_pass_through_never_falls_back_to_another_model():
    deps = make_deps({"groq-small": FakeBehavior(fail_before="server")})
    with pytest.raises(AllModelsFailed):
        call(deps, model="groq-small", tau=None)
    assert set(models_called(deps)) == {"groq-small"}              # gemini-flash (its tier's backup) never tried


def test_unknown_model_raises_before_calling_anything():
    pred = FakePredictor()
    deps = make_deps(predictor=pred)
    with pytest.raises(UnknownModel):
        call(deps, model="gpt-17-ultra", tau=None)
    assert pred.calls == [] and models_called(deps) == []


# ---------- fallback only before the first token ----------

def test_fallback_before_the_first_token_changes_the_model():
    deps = make_deps({"groq-small": FakeBehavior(fail_before="server")})
    routed, pieces, _, _ = call_and_read(deps, tau=0.92)
    d = routed.decision
    assert (d.primary_model, d.model, d.tier) == ("groq-small", "gemini-flash", "medium") and d.fell_back
    assert any(a.model_name == "groq-small" and a.outcome == "error" for a in d.attempts)
    assert "".join(pieces) == fake_text("gemini-flash")


def test_first_token_limit_falls_back_without_waiting_for_the_slow_model():
    deps = make_deps({"groq-small": FakeBehavior(first_token_delay=1.0)}, first_token=0.1)
    started = time.perf_counter()
    d = call(deps, tau=0.92).decision
    assert time.perf_counter() - started < 0.8
    assert d.model == "gemini-flash"
    assert any(a.model_name == "groq-small" and a.error_kind == "timeout" for a in d.attempts)


def test_a_steady_slow_stream_is_not_cut_off():
    """Only the first token has a limit: a stream that starts in time may take longer than it overall."""
    deps = make_deps({"groq-small": FakeBehavior(first_token_delay=0.01, piece_delay=0.03)}, first_token=0.1)
    routed, pieces, _, _ = call_and_read(deps, tau=0.92)
    assert routed.decision.model == "groq-small" and "".join(pieces) == fake_text("groq-small")


def test_an_error_after_the_first_token_propagates_with_no_fallback():
    deps = make_deps({"groq-small": FakeBehavior(fail_after=2)})

    async def go():
        routed = await route_and_stream(ASK, tau=0.92, task_type=None, model="auto", stream=True, settings=DEFAULT,
                                        deps=deps, request_id="r")
        got = []
        with pytest.raises(ProviderError):
            async for p in routed.pieces:
                got.append(p)
        return routed, got
    routed, got = anyio.run(go)
    assert routed.decision.model == "groq-small" and len(got) == 2
    assert models_called(deps) == ["groq-small"]


def test_every_model_failing_raises_before_returning():
    deps = make_deps({"groq-small": FakeBehavior(fail_before="server"),
                      "gemini-flash": FakeBehavior(fail_before="timeout")})
    with pytest.raises(AllModelsFailed):
        call(deps, tau=0.92)


# ---------- what comes back ----------

def test_routing_is_decided_before_any_piece_is_read():
    deps = make_deps()
    routed = call(deps, tau=0.92)
    assert isinstance(routed, RoutedStream)
    assert routed.decision.model == "groq-small" and models_called(deps) == ["groq-small"]
    assert hasattr(routed.pieces, "__aiter__")


def test_response_appears_only_after_the_pieces_run_out():
    routed, pieces, before, after = call_and_read(make_deps(), tau=0.92)
    assert before is None
    assert after is not None and after.text == "".join(pieces) == fake_text("groq-small")
    assert after.model_name == "groq-small" and after.output_tokens > 0 and after.reported_cost > 0
    assert len(pieces) > 1                                          # it really streamed


def test_non_streaming_yields_the_whole_answer_once():
    deps = make_deps()
    routed, pieces, before, after = call_and_read(deps, tau=0.92, stream=False)
    assert pieces == [fake_text("groq-small")] and before is None and after.text == pieces[0]
    assert deps.fake_models.calls[0]["stream"] is False


def test_non_streaming_goes_through_the_cache(tmp_path):
    deps = make_deps(cache=ResponseCache(tmp_path / "cache.sqlite3"))
    first = call_and_read(deps, tau=0.92, stream=False)[3]
    second = call_and_read(deps, tau=0.92, stream=False)[3]
    assert not first.cached and second.cached and second.reported_cost == 0
    assert models_called(deps) == ["groq-small"]                    # the second answer came from the cache


def test_non_streaming_falls_back_too():
    deps = make_deps({"groq-small": FakeBehavior(fail_before="server")})
    routed, pieces, _, after = call_and_read(deps, tau=0.92, stream=False)
    assert routed.decision.model == "gemini-flash" and after.model_name == "gemini-flash"


# ---------- generation settings ----------

def test_caller_settings_reach_the_model():
    deps = make_deps()
    call_and_read(deps, tau=0.92, settings=GenerationSettings(temperature=0.7, max_tokens=123))
    c = deps.fake_models.calls[0]
    assert (c["temperature"], c["max_tokens"]) == (0.7, 123)
    assert c["reasoning_effort"] == "low"                           # groq-small's model_params still apply


def test_without_caller_settings_the_phase2_settings_are_used():
    deps = make_deps()
    call_and_read(deps, tau=0.92)
    c = deps.fake_models.calls[0]
    assert (c["temperature"], c["max_tokens"]) == (0.0, 4000)


def test_pass_through_honours_caller_settings_too():
    deps = make_deps()
    call_and_read(deps, model="groq-allam", tau=None, settings=GenerationSettings(max_tokens=50))
    assert deps.fake_models.calls[0]["max_tokens"] == 50
