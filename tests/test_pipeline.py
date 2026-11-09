"""Verification pipeline order: format, then agreement, then the AI checker (Phase 4 Step 1). No API calls."""
import hashlib
import json

import pytest
import yaml

from app.providers.base import Response
from app.providers.common import ProviderError
from app.verifier.ai_checker import CheckerSettings
from app.verifier.pipeline import verify

SETTINGS = CheckerSettings(model_name="groq-small")
MATH = [{"role": "user", "content": "Tom has 2 apples and buys 3. How many?\n\n"
                                    "Work through it step by step, then give the final answer on the last line "
                                    "in the form: Answer: <number>"}]
ESSAY = [{"role": "user", "content": "Write a two-line poem about rain."}]


class FakeSend:
    def __init__(self, text='{"good": true, "reason": "Fine."}', finish="stop", raises=None):
        self.text, self.finish, self.raises, self.calls = text, finish, raises, 0

    def __call__(self, messages, model_name, settings):
        self.calls += 1
        if self.raises:
            raise self.raises
        return Response(text=self.text, model_name=model_name, model_id="m", provider="groq", input_tokens=250,
                        output_tokens=70, latency_ms=1.0, cost=0.0, reported_cost=0.00004,
                        finish_reason=self.finish, reasoning_tokens=30)


def never_called(*args, **kwargs):
    raise AssertionError("the AI checker must not be called")


def run(messages, answer, send=never_called, **kw):
    return verify(messages, answer, send=send, settings=SETTINGS, **kw)


# ---------- (a) format ----------

@pytest.mark.parametrize("answer, kw", [
    ("2 + 3 = 5", {}),                                   # 'Answer:' line requested but missing
    ("", {}),
    ("Answer: 5", {"truncated": True}),
    ("Answer: 5", {"error": {"kind": "server"}}),
])
def test_format_flag_stops_before_agreement_and_ai(answer, kw):
    result = run(MATH, answer, other_answer="Answer: 5", **kw)
    assert (result.good, result.decided_by) == (False, "format")
    assert result.reasons and result.ai is None and result.api_calls == 0


# ---------- (b) agreement ----------

def test_agreement_stops_before_the_ai_checker():
    result = run(MATH, "2+3=5\nAnswer: 5", other_answer="Answer: $5.00")
    assert (result.good, result.decided_by, result.ai, result.input_tokens) == (True, "agreement", None, 0)


def test_disagreement_is_bad_without_the_ai_checker():
    result = run(MATH, "Answer: 5", other_answer="Answer: 6")
    assert (result.good, result.decided_by) == (False, "agreement")


def test_other_answer_without_a_final_answer_counts_as_disagreement():
    assert run(MATH, "Answer: 5", other_answer="I think five").decided_by == "agreement"
    assert run(MATH, "Answer: 5", other_answer="Answer: 5", other_truncated=True).good is False


# ---------- (c) AI checker only when needed ----------

def test_ai_checker_runs_when_no_second_answer():
    send = FakeSend()
    result = run(MATH, "Answer: 5", send=send)
    assert (result.good, result.decided_by, send.calls) == (True, "ai_checker", 1)


def test_ai_checker_runs_when_the_prompt_is_not_comparable():
    send = FakeSend('{"good": false, "reason": "Not a poem."}')
    result = run(ESSAY, "Rain is wet.", other_answer="Drops fall.", send=send)
    assert (result.good, result.decided_by, send.calls) == (False, "ai_checker", 1)
    assert result.reasons == ("Not a poem.",)


def test_ai_result_carries_cost_fields():
    result = run(ESSAY, "Rain taps the glass,\nthe street forgets the sun.", send=FakeSend())
    assert (result.checker_model, result.checker_version) == ("groq-small", 1)
    assert (result.input_tokens, result.output_tokens, result.reasoning_tokens, result.api_calls) == (250, 70, 30, 1)
    assert result.reported_cost == pytest.approx(0.00004)
    assert result.ai.status == "ok"


def test_checker_model_can_be_overridden():
    result = run(ESSAY, "Rain.", send=FakeSend(), checker_model="groq-large")
    assert result.checker_model == "groq-large"


@pytest.mark.parametrize("send", [
    FakeSend(text="", finish="length"),
    FakeSend(text="not json at all"),
    FakeSend(text='{"good": "true", "reason": "x"}'),
    FakeSend(raises=ProviderError("groq", "timeout", "slow")),
])
def test_checker_error_maps_to_unknown(send):
    result = run(ESSAY, "Rain.", send=send)
    assert (result.good, result.decided_by) == (None, "ai_checker_error")
    assert result.ai.status == "error" and result.reasons


# ---------- (d) nothing applies ----------

@pytest.mark.parametrize("messages, other", [(ESSAY, None), (ESSAY, "Drops."), (MATH, None)])
def test_ai_check_false_never_calls_send(messages, other):
    answer = "Answer: 5" if messages is MATH else "Rain."
    result = run(messages, answer, other_answer=other, ai_check=False)
    assert (result.good, result.decided_by, result.ai, result.api_calls) == (None, "none", None, 0)


def test_default_settings_come_from_checker_yaml(monkeypatch):
    from app.verifier import ai_checker
    seen = []
    monkeypatch.setattr(ai_checker, "default_send", lambda m, name, s: seen.append((name, s.checker_version)) or
                        FakeSend()(m, name, s))
    assert verify(ESSAY, "Rain.").decided_by == "ai_checker"
    assert seen == [("groq-small", 2)]


# ---------- the Phase 2 judge is unchanged (its v3 verdicts are ground truth) ----------

JUDGE_PROMPT_SHA256 = "2b57be60ac277c75c446e4bab7018e49e947eaa84e63045e73ead8dcbd2897b1"
JUDGE_MESSAGES_SHA256 = "7c7b3de14e24c40b58d248e3f1a7272e49faa1f4b86ad32130bc09c99b1b8039"


def test_judge_prompt_messages_and_version_are_unchanged():
    from app.registry import PROJECT_ROOT
    from app.verifier.judge import JUDGE_SYSTEM_PROMPT, build_judge_messages
    assert hashlib.sha256(JUDGE_SYSTEM_PROMPT.encode()).hexdigest() == JUDGE_PROMPT_SHA256
    messages = build_judge_messages([{"role": "user", "content": "Write a tagline."}], ["under 5 words"],
                                    "Fresh meals, zero stress.", "Dinner, done.")
    assert hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest() == JUDGE_MESSAGES_SHA256
    assert yaml.safe_load((PROJECT_ROOT / "config" / "judge.yaml").read_text())["judge_version"] == 3
