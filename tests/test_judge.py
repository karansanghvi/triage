import json

import pytest

from app.verifier.judge import (
    JUDGE_SYSTEM_PROMPT, JudgeSettings, build_judge_messages, judge, load_judge_settings,
)

SETTINGS = JudgeSettings(model_name="fake", temperature=0, reasoning_effort="low", max_tokens=500,
                         max_attempts=2, pass_threshold=4)
MESSAGES = [{"role": "system", "content": "You write taglines."},
            {"role": "user", "content": "Write a tagline under 8 words. Don't say 'delicious'."}]
CONSTRAINTS = ["under 8 words", "does not use 'delicious'"]


class FakeJudge:
    """Returns scripted replies in order and records what it was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, messages, settings):
        self.calls.append(messages)
        return self.replies.pop(0)


def verdict(score, reason="ok", met=(True, True)):
    return json.dumps({"constraints": [{"constraint": c, "met": m} for c, m in zip(CONSTRAINTS, met)],
                       "score": score, "reason": reason})


def run(fake, constraints=CONSTRAINTS, reference="Fresh meals, zero stress.", candidate="Dinner, done.", **kw):
    return judge(MESSAGES, constraints, reference, candidate, SETTINGS, send=fake, **kw)


# ---------- parsing and the pass threshold ----------

@pytest.mark.parametrize("score, passed", [(5, True), (4, True), (3, False), (2, False), (1, False)])
def test_score_parsing_and_pass_threshold(score, passed):
    result = run(FakeJudge(verdict(score, "clear and on brand")))
    assert (result.status, result.score, result.passed) == ("scored", score, passed)
    assert result.reason == "clear and on brand"
    assert result.attempts == 1


def test_verdict_inside_a_json_code_fence_is_accepted():
    result = run(FakeJudge("```json\n" + verdict(5) + "\n```"))
    assert result.score == 5


def test_pass_threshold_comes_from_settings():
    strict = JudgeSettings(**{**SETTINGS.__dict__, "pass_threshold": 5})
    result = judge(MESSAGES, CONSTRAINTS, "ref", "cand", strict, send=FakeJudge(verdict(4)))
    assert result.passed is False


def test_settings_load_threshold_from_thresholds_yaml():
    settings = load_judge_settings()
    assert settings.pass_threshold == 4
    assert settings.model_name == "groq-large"
    assert (settings.temperature, settings.reasoning_effort) == (0, "low")


# ---------- malformed verdicts ----------

@pytest.mark.parametrize("bad", [
    "The candidate is pretty good, I'd say 4.",                     # not JSON
    json.dumps({"score": 4}),                                        # no reason, no constraint checks
    json.dumps({"constraints": [], "score": 4, "reason": "ok"}),     # wrong number of constraint checks
    verdict(6),                                                      # out of range
    verdict(4.5),                                                    # not a whole number
    verdict(True),                                                   # boolean, not a score
    verdict(4, reason=""),                                           # empty reason
    json.dumps([4, "ok"]),                                           # JSON, but not an object
])
def test_malformed_verdict_is_retried_once(bad):
    fake = FakeJudge(bad, verdict(4))
    result = run(fake)
    assert (result.status, result.score, result.attempts) == ("scored", 4, 2)
    assert len(fake.calls) == 2
    assert fake.calls[0] == fake.calls[1]          # the retry sends the same prompt


def test_two_malformed_verdicts_record_an_error():
    fake = FakeJudge("nope", "{broken json")
    result = run(fake)
    assert result.status == "error"
    assert result.score is None and result.passed is None
    assert result.attempts == 2
    assert "unparseable" in result.reason
    assert result.raw == "{broken json"


# ---------- constraint cap ----------

def test_breaking_a_constraint_caps_the_score_at_3():
    result = run(FakeJudge(verdict(5, "great but uses the banned word", met=(True, False))))
    assert result.judge_score == 5
    assert result.score == 3
    assert result.passed is False
    assert result.broken_constraints == ("does not use 'delicious'",)


def test_cap_never_raises_a_lower_score():
    result = run(FakeJudge(verdict(2, met=(False, False))))
    assert result.score == 2


def test_prompt_without_constraints_needs_an_empty_list():
    fake = FakeJudge(json.dumps({"constraints": [], "score": 5, "reason": "good"}))
    result = run(fake, constraints=[])
    assert (result.status, result.score) == ("scored", 5)


# ---------- no_reference ----------

@pytest.mark.parametrize("reference, truncated", [(None, False), ("", False), ("   \n", False), ("long answer...", True)])
def test_missing_empty_or_truncated_reference_is_not_judged(reference, truncated):
    fake = FakeJudge()                             # would fail if it were called
    result = run(fake, reference=reference, reference_truncated=truncated)
    assert result.status == "no_reference"
    assert result.score is None and result.passed is None
    assert fake.calls == []


# ---------- prompt ----------

def test_judge_prompt_contains_rubric_constraints_and_both_answers():
    messages = build_judge_messages(MESSAGES, CONSTRAINTS, "REF TEXT", "CAND TEXT")
    assert messages[0] == {"role": "system", "content": JUDGE_SYSTEM_PROMPT}
    user = messages[1]["content"]
    assert "1. under 8 words" in user and "2. does not use 'delicious'" in user
    assert "<reference_answer>\nREF TEXT\n</reference_answer>" in user
    assert "<candidate_answer>\nCAND TEXT\n</candidate_answer>" in user
    assert "[system]\nYou write taglines." in user
    for phrase in ["5 = as good as or better", "4 = slightly worse", "3 = noticeably worse",
                   "2 = mostly fails", "1 = wrong, off-topic, a refusal, or not in English", "at most 3"]:
        assert phrase in JUDGE_SYSTEM_PROMPT
