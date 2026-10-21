import os

import pytest

from app.providers.common import ProviderError
from app.verifier.judge import JudgeResult
from eval import run_judge
from eval.run_judge import JudgeRunner, JudgmentLog, judge_prompt

PROMPT = {"id": "own_creative_010", "scoring": "judge", "messages": [{"role": "user", "content": "Write a tagline."}],
          "constraints": []}


def answer(tier, text="An answer.", truncated=False, error=None):
    return {"prompt_id": PROMPT["id"], "tier": tier, "answer": text, "truncated": truncated, "error": error}


def answers(low="Low answer.", medium="Medium answer.", high="High answer.", **flags):
    out = {}
    for tier, text in (("low", low), ("medium", medium), ("high", high)):
        if text is not None:
            out[(PROMPT["id"], tier)] = answer(tier, text, truncated=flags.get(f"{tier}_truncated", False),
                                               error=flags.get(f"{tier}_error"))
    return out


class FakeJudge:
    """Scores candidates by a scripted table: candidate text -> score (or an exception to raise)."""

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def __call__(self, prompt, candidate, reference):
        self.calls.append((candidate, reference))
        outcome = self.scores[candidate]
        if isinstance(outcome, Exception):
            raise outcome
        return (JudgeResult(status="scored", score=outcome, passed=outcome >= 4, reason=f"scored {outcome}",
                            judge_score=outcome, attempts=1),
                {"input_tokens": 1000, "output_tokens": 100, "api_calls": 1})


def by_tier(records):
    return {r["tier"]: r for r in records}


# ---------- option C ----------

def test_low_pass_means_medium_is_assumed_not_judged():
    fake = FakeJudge({"Low answer.": 5})
    records = by_tier(judge_prompt(PROMPT, answers(), {}, fake))
    assert fake.calls == [("Low answer.", "High answer.")]          # only low was judged, against high
    assert (records["low"]["status"], records["low"]["method"]) == ("pass", "judged")
    assert records["medium"]["status"] == "assumed_pass"
    assert records["medium"]["method"] == "assumed (option C)"
    assert records["medium"]["api_calls"] == 0 and records["medium"]["score"] is None
    assert (records["high"]["status"], records["high"]["method"]) == ("pass", "auto_high")


def test_low_fail_means_medium_is_judged_second():
    fake = FakeJudge({"Low answer.": 2, "Medium answer.": 4})
    records = by_tier(judge_prompt(PROMPT, answers(), {}, fake))
    assert [c for c, _ in fake.calls] == ["Low answer.", "Medium answer."]   # low first, then medium
    assert (records["low"]["status"], records["medium"]["status"]) == ("fail", "pass")
    assert records["medium"]["method"] == "judged"
    assert records["medium"]["score"] == 4


@pytest.mark.parametrize("flags, reason", [
    ({"low_truncated": True}, "truncated"),
    ({"low": "   "}, "empty"),
    ({"low_error": {"kind": "client", "message": "Error code: 413 - Request too large"}}, "input too large"),
])
def test_pre_failed_low_fails_without_a_call_and_medium_is_judged(flags, reason):
    low_text = flags.pop("low", "Low answer.")
    fake = FakeJudge({"Medium answer.": 5})
    records = by_tier(judge_prompt(PROMPT, answers(low=low_text, **flags), {}, fake))
    assert (records["low"]["status"], records["low"]["method"], records["low"]["reason"]) == ("fail", "pre_failed", reason)
    assert fake.calls == [("Medium answer.", "High answer.")]
    assert records["medium"]["status"] == "pass"


def test_pre_failed_medium_is_failed_without_a_call():
    fake = FakeJudge({"Low answer.": 1})
    records = by_tier(judge_prompt(PROMPT, answers(medium_truncated=True), {}, fake))
    assert len(fake.calls) == 1
    assert (records["medium"]["status"], records["medium"]["method"]) == ("fail", "pre_failed")


@pytest.mark.parametrize("flags", [{"high_truncated": True}, {"high": ""}])
def test_truncated_or_empty_high_is_no_reference_for_every_tier(flags):
    fake = FakeJudge({})
    records = by_tier(judge_prompt(PROMPT, answers(**flags), {}, fake))
    assert fake.calls == []
    assert {t: r["status"] for t, r in records.items()} == {t: "no_reference" for t in ("low", "medium", "high")}


def test_missing_high_answer_leaves_everything_pending():
    records = by_tier(judge_prompt(PROMPT, answers(high=None), {}, FakeJudge({})))
    assert {r["status"] for r in records.values()} == {"pending"}


def test_provider_error_marks_the_tier_pending():
    fake = FakeJudge({"Low answer.": ProviderError("groq", "rate_limit", "429", 429)})
    records = by_tier(judge_prompt(PROMPT, answers(), {}, fake))
    assert records["low"]["status"] == "pending"
    assert records["low"]["error"]["kind"] == "rate_limit"
    assert records["medium"]["status"] == "pending"            # waits for the low result


def test_unparseable_verdict_is_an_error_not_a_fail():
    def unparseable(prompt, candidate, reference):
        return JudgeResult(status="error", reason="unparseable verdict after 2 attempts", attempts=2), {"api_calls": 2}
    records = by_tier(judge_prompt(PROMPT, answers(), {}, unparseable))
    assert records["low"]["status"] == "error"
    assert records["medium"]["status"] == "pending"


# ---------- resuming ----------

def test_rerun_skips_finished_and_retries_pending(tmp_path):
    second = dict(PROMPT, id="own_creative_011")
    all_answers = answers()
    all_answers.update({(second["id"], t): dict(a, prompt_id=second["id"]) for (_, t), a in answers().items()})
    log = JudgmentLog(tmp_path / "judgments.jsonl")

    flaky = FakeJudge({"Low answer.": 2, "Medium answer.": ProviderError("groq", "server", "503", 503)})
    JudgeRunner([PROMPT, second], all_answers, log, judge_call=flaky, out_dir=tmp_path).run(passes=1, pause=0)
    assert len(flaky.calls) == 4                                 # low + medium for each prompt
    assert {r["status"] for (_, t), r in log.latest().items() if t == "medium"} == {"pending"}

    fixed = FakeJudge({"Low answer.": 2, "Medium answer.": 5})
    reloaded = JudgmentLog(tmp_path / "judgments.jsonl")          # as if the process restarted
    JudgeRunner([PROMPT, second], all_answers, reloaded, judge_call=fixed, out_dir=tmp_path).run(passes=1, pause=0)
    assert [c for c, _ in fixed.calls] == ["Medium answer.", "Medium answer."]   # lows were not re-judged
    assert {r["status"] for (_, t), r in reloaded.latest().items() if t == "medium"} == {"pass"}

    untouched = FakeJudge({})
    JudgeRunner([PROMPT, second], all_answers, JudgmentLog(tmp_path / "judgments.jsonl"),
                judge_call=untouched, out_dir=tmp_path).run(passes=1, pause=0)
    assert untouched.calls == []                                   # everything finished: no calls at all


def test_only_judge_scored_prompts_are_run(tmp_path):
    exact = dict(PROMPT, id="own_qa_001", scoring="exact_match")
    runner = JudgeRunner([PROMPT, exact], answers(), JudgmentLog(tmp_path / "j.jsonl"),
                         judge_call=FakeJudge({"Low answer.": 5}), out_dir=tmp_path)
    assert [p["id"] for p in runner.prompts] == [PROMPT["id"]]


# ---------- refuses to start while the collector runs ----------

def test_refuses_to_start_while_the_collector_is_running(tmp_path, monkeypatch):
    collector_dir = tmp_path / "answers"
    collector_dir.mkdir()
    (collector_dir / "run.pid").write_text(str(os.getpid()))      # a live process
    monkeypatch.setattr(run_judge, "FULL_DIR", collector_dir)
    monkeypatch.setattr(run_judge, "JUDGMENTS_DIR", tmp_path / "judgments")
    monkeypatch.setattr("sys.argv", ["run_judge"])
    with pytest.raises(SystemExit, match="collector is still running"):
        run_judge.main()
    assert not (tmp_path / "judgments").exists()                    # nothing was started
