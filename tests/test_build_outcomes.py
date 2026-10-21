import pytest

from eval.build_outcomes import build_all, build_outcome

AUTO = {"id": "gsm8k_0002", "source": "gsm8k", "task_type": "reasoning", "scoring": "final_answer",
        "difficulty_guess": "unrated", "messy": False}
JUDGED = {"id": "arena_0010", "source": "arena", "task_type": "coding", "scoring": "judge",
          "difficulty_guess": "unrated"}


def score(tier, status, reason="numbers match", scorer="final_answer"):
    return {"prompt_id": AUTO["id"], "tier": tier, "status": status, "scorer": scorer, "reason": reason}


def scores(low, medium, high, extra="fail"):
    return {t: score(t, s) for t, s in zip(("low", "medium", "high", "extra"), (low, medium, high, extra))}


def judgment(tier, status, method="judged", score_value=None, reason="ok"):
    return {"prompt_id": JUDGED["id"], "tier": tier, "status": status, "method": method,
            "score": score_value, "reason": reason}


# ---------- label cases ----------

@pytest.mark.parametrize("low, medium, high, label", [
    ("pass", "pass", "pass", "low"),
    ("pass", "fail", "fail", "low"),          # low passing is enough, whatever medium/high did
    ("fail", "pass", "pass", "medium"),
    ("fail", "pass", "fail", "medium"),
    ("fail", "fail", "pass", "high"),
])
def test_label_is_the_cheapest_passing_tier(low, medium, high, label):
    outcome = build_outcome(AUTO, scores(low, medium, high), {})
    assert (outcome["label"], outcome["label_status"], outcome["none_passed"]) == (label, "labeled", False)
    assert outcome["low_pass"] is (low == "pass")


def test_none_passed_is_still_labeled_high():
    outcome = build_outcome(AUTO, scores("fail", "fail", "fail"), {})
    assert (outcome["label"], outcome["label_status"], outcome["none_passed"]) == ("high", "labeled", True)


@pytest.mark.parametrize("extra", ["pass", "fail", "pending"])
def test_extra_never_affects_the_label(extra):
    assert build_outcome(AUTO, scores("fail", "fail", "pass", extra=extra), {})["label"] == "high"
    assert build_outcome(AUTO, scores("fail", "pass", "pass", extra=extra), {})["label"] == "medium"


def test_scorer_name_and_pre_failed_methods():
    rows = scores("fail", "pass", "pass")
    rows["low"] = score("low", "fail", reason="truncated")
    tiers = build_outcome(AUTO, rows, {})["tiers"]
    assert tiers["low"]["method"] == "pre_failed"
    assert tiers["medium"]["method"] == "final_answer"


# ---------- incomplete ----------

def test_pending_needed_tier_is_incomplete():
    outcome = build_outcome(AUTO, scores("fail", "pending", "pass"), {})
    assert (outcome["label"], outcome["label_status"]) == (None, "incomplete")
    assert outcome["medium_pass"] is None


def test_pending_tier_that_isnt_needed_does_not_block_the_label():
    outcome = build_outcome(AUTO, scores("pass", "pending", "pending"), {})
    assert (outcome["label"], outcome["label_status"]) == ("low", "labeled")


def test_missing_score_rows_are_incomplete():
    outcome = build_outcome(AUTO, {}, {})
    assert outcome["label_status"] == "incomplete"


# ---------- judge-scored prompts ----------

def test_assumed_pass_counts_as_a_medium_pass_with_a_flag():
    judged = {"low": judgment("low", "fail", score_value=2), "medium": judgment("medium", "assumed_pass", "assumed (option C)"),
              "high": judgment("high", "pass", "auto_high")}
    outcome = build_outcome(JUDGED, {}, judged)
    assert outcome["medium_pass"] is True and outcome["medium_pass_assumed"] is True
    assert outcome["tiers"]["medium"]["method"] == "assumed"
    assert outcome["tiers"]["low"]["judge_score"] == 2
    assert outcome["label"] == "medium"


def test_judged_low_pass_labels_low_and_medium_is_assumed():
    judged = {"low": judgment("low", "pass", score_value=5), "medium": judgment("medium", "assumed_pass", "assumed (option C)"),
              "high": judgment("high", "pass", "auto_high")}
    outcome = build_outcome(JUDGED, {}, judged)
    assert outcome["label"] == "low"
    assert outcome["low_pass_assumed"] is False


def test_judged_low_and_medium_fail_labels_high():
    judged = {"low": judgment("low", "fail", score_value=2), "medium": judgment("medium", "fail", score_value=3),
              "high": judgment("high", "pass", "auto_high")}
    outcome = build_outcome(JUDGED, {}, judged)
    assert (outcome["label"], outcome["none_passed"]) == ("high", False)


def test_no_reference_is_excluded_for_every_tier():
    judged = {t: judgment(t, "no_reference", "no_reference") for t in ("low", "medium", "high")}
    outcome = build_outcome(JUDGED, {"extra": {"prompt_id": JUDGED["id"], "tier": "extra", "status": "needs_judge",
                                               "scorer": None, "reason": "judge-scored prompt"}}, judged)
    assert (outcome["label"], outcome["label_status"]) == (None, "excluded_no_reference")
    assert outcome["tiers"]["extra"]["status"] == "no_reference"


def test_missing_judgments_leave_judge_prompts_incomplete():
    outcome = build_outcome(JUDGED, {}, {})
    assert outcome["label_status"] == "incomplete"
    assert {outcome["tiers"][t]["status"] for t in ("low", "medium", "high")} == {"pending"}
    assert outcome["tiers"]["extra"]["status"] == "not_judged"


def test_partial_judgments_incomplete_until_the_needed_tier_is_judged():
    partial = {"low": judgment("low", "fail", score_value=1), "high": judgment("high", "pass", "auto_high")}
    assert build_outcome(JUDGED, {}, partial)["label_status"] == "incomplete"     # medium not judged yet
    partial["low"] = judgment("low", "pass", score_value=4)
    assert build_outcome(JUDGED, {}, partial)["label"] == "low"                   # medium not needed


def test_judge_error_on_a_needed_tier_is_incomplete():
    judged = {"low": judgment("low", "error"), "high": judgment("high", "pass", "auto_high")}
    assert build_outcome(JUDGED, {}, judged)["label_status"] == "incomplete"


# ---------- whole dataset ----------

def test_one_line_per_prompt_in_order_and_unknown_ids_warn():
    rows = list(scores("pass", "pass", "pass").values()) + [dict(score("low", "pass"), prompt_id="gsm8k_9999")]
    stray = [dict(judgment("low", "pass"), prompt_id="arena_9999")]
    outcomes, warnings = build_all([JUDGED, AUTO], rows, stray)
    assert [o["id"] for o in outcomes] == ["arena_0010", "gsm8k_0002"]
    assert any("scores.jsonl" in w and "gsm8k_9999" in w for w in warnings)
    assert any("judgments.jsonl" in w and "arena_9999" in w for w in warnings)


def test_rebuilding_gives_identical_output():
    rows = list(scores("fail", "pass", "pass").values())
    assert build_all([AUTO, JUDGED], rows, []) == build_all([AUTO, JUDGED], rows, [])


def test_duplicate_prompt_ids_are_rejected():
    with pytest.raises(ValueError):
        build_all([AUTO, AUTO], [], [])
